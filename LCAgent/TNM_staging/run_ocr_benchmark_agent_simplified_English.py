import argparse
import ast
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm
from zhipuai import ZhipuAI
import yaml

try:
    from bert_score import BERTScorer
except ImportError:
    BERTScorer = None

RUNNER_BASE_URL = os.getenv("RUNNER_BASE_URL", "https://api.siliconflow.cn/v1")
RUNNER_MODEL = os.getenv("RUNNER_MODEL", "Qwen/Qwen3-VL-32B-Thinking")
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "openai").strip().lower()
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")

AGENT_EXTRACT_MODEL = os.getenv("AGENT_EXTRACT_MODEL", RUNNER_MODEL)
AGENT_STAGE_MODEL = os.getenv("AGENT_STAGE_MODEL", RUNNER_MODEL)
AGENT_PARALLEL_BRANCHES = os.getenv("AGENT_PARALLEL_BRANCHES", "0") == "1"
MAX_EXTRACT_ATTEMPTS = max(1, int(os.getenv("MAX_EXTRACT_ATTEMPTS", "2")))
MAX_STAGE_ATTEMPTS = max(1, int(os.getenv("MAX_STAGE_ATTEMPTS", "2")))

DEFAULT_BENCHMARK_DIR = os.getenv("BENCHMARK_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
DEFAULT_OUTPUT_FILE = os.getenv(
    "OUTPUT_FILE",
    os.path.join(DEFAULT_BENCHMARK_DIR, "my_framework", "results_TNMstaging", "results_ocr_Agent", f"{RUNNER_MODEL.split('/')[-1]}_English_agent_ocr_simplified.json"),
)
MAX_RUNNER_ATTEMPTS = max(1, int(os.getenv("MAX_RUNNER_ATTEMPTS", "2")))
MAX_JUDGE_ATTEMPTS = max(1, int(os.getenv("MAX_JUDGE_ATTEMPTS", "1")))
BERT_BATCH_SIZE = max(1, int(os.getenv("BERT_BATCH_SIZE", "16")))

SIMPLIFIED_KEYS = {
    "T_stage",
    "T_reasoning",
    "N_stage",
    "N_reasoning",
    "M_stage",
    "M_reasoning",
    "Final_TNM",
}
JUDGE_SCORE_KEYS = {"T_score", "N_score", "M_score"}

_bert_scorer = None
_bert_scorer_unavailable = False
_dify_prompt_cache: Dict[str, Dict[str, str]] = {}

DIFY_WORKFLOW_FILE = os.getenv("DIFY_WORKFLOW_FILE", os.path.join(DEFAULT_BENCHMARK_DIR, "my_framework", "LCAgent", "TNM_staging", "tnm_staging_assistant_en_0327_fixed.yml"))


def make_runner_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=RUNNER_BASE_URL)


def make_judge_client(api_key: str):
    if JUDGE_PROVIDER == "openai" or JUDGE_BASE_URL:
        base_url = JUDGE_BASE_URL if JUDGE_BASE_URL else RUNNER_BASE_URL
        return OpenAI(api_key=api_key, base_url=base_url)
    return ZhipuAI(api_key=api_key)


def atomic_save_json(data: Dict[str, Any], output_file: str) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    temp_file = f"{output_file}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, output_file)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    def _parse_obj(candidate: str) -> Optional[Dict[str, Any]]:
        candidate = candidate.strip()
        if not candidate:
            return None
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        try:
            obj = ast.literal_eval(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _balanced_json_objects(s: str) -> List[str]:
        objs: List[str] = []
        start = -1
        depth = 0
        in_str = False
        quote = ""
        escape = False
        for i, ch in enumerate(s):
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_str = False
                continue

            if ch in ('"', "'"):
                in_str = True
                quote = ch
                continue

            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    objs.append(s[start : i + 1])
                    start = -1
        return objs

    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if code_block:
        parsed = _parse_obj(code_block.group(1))
        if parsed:
            return parsed

    for candidate in _balanced_json_objects(cleaned):
        parsed = _parse_obj(candidate)
        if parsed:
            return parsed

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = _parse_obj(cleaned[start : end + 1])
        if parsed:
            return parsed

    kv: Dict[str, str] = {}
    for key in ["T_stage", "T_reasoning", "N_stage", "N_reasoning", "M_stage", "M_reasoning", "Final_TNM"]:
        m = re.search(rf"(?im)^\s*{re.escape(key)}\s*[:：]\s*(.+?)\s*$", cleaned)
        if m:
            v = m.group(1).strip().strip('"').strip("'").strip()
            if v:
                kv[key] = v
    if len(kv) >= 2:
        return kv
    return None


def normalize_stage_token(value: Any, kind: str) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().upper().replace(" ", "")
    if kind == "T":
        m = re.search(r"T(?:IS|X|[0-4][ABC]?)", text)
        return m.group(0) if m else ""
    if kind == "N":
        m = re.search(r"N(?:X|[0-3])", text)
        return m.group(0) if m else ""
    if kind == "M":
        m = re.search(r"M(?:X|0|1[ABC]?)", text)
        return m.group(0) if m else ""
    return ""


def contains_cjk(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def normalize_simplified_output(candidate: Dict[str, Any]) -> Dict[str, str]:
    out = {k: "" for k in SIMPLIFIED_KEYS}
    for key in SIMPLIFIED_KEYS:
        value = candidate.get(key)
        out[key] = value.strip() if isinstance(value, str) else ""

    out["T_stage"] = normalize_stage_token(out["T_stage"], "T")
    out["N_stage"] = normalize_stage_token(out["N_stage"], "N")
    out["M_stage"] = normalize_stage_token(out["M_stage"], "M")

    parts = [p for p in [out["T_stage"], out["N_stage"], out["M_stage"]] if p]
    out["Final_TNM"] = " ".join(parts)

    return out


def validate_simplified_output(data: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "top-level is not an object"
    missing = SIMPLIFIED_KEYS - set(data.keys())
    if missing:
        return False, f"missing keys: {sorted(missing)}"

    for key in SIMPLIFIED_KEYS:
        if not isinstance(data.get(key), str) or not data.get(key).strip():
            return False, f"{key} must be a non-empty string"

    t = normalize_stage_token(data.get("T_stage"), "T")
    n = normalize_stage_token(data.get("N_stage"), "N")
    m = normalize_stage_token(data.get("M_stage"), "M")
    if not t or not n or not m:
        return False, "T_stage/N_stage/M_stage must be valid TNM tokens"

    final_text = str(data.get("Final_TNM", ""))
    final_t = normalize_stage_token(final_text, "T")
    final_n = normalize_stage_token(final_text, "N")
    final_m = normalize_stage_token(final_text, "M")
    if final_t != t or final_n != n or final_m != m:
        return False, "Final_TNM must include T_stage, N_stage, and M_stage"

    for key in ["T_reasoning", "N_reasoning", "M_reasoning"]:
        if contains_cjk(str(data.get(key, ""))):
            return False, f"{key} must be written in English"

    return True, "ok"


def validate_judge_json(judge_json: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(judge_json, dict):
        return False, "top-level is not an object"
    if "scores" not in judge_json or "justification" not in judge_json:
        return False, "judge JSON must contain scores and justification"

    scores = judge_json.get("scores")
    if not isinstance(scores, dict):
        return False, "scores must be an object"

    missing_scores = JUDGE_SCORE_KEYS - set(scores.keys())
    if missing_scores:
        return False, f"missing score keys: {sorted(missing_scores)}"

    for key in JUDGE_SCORE_KEYS:
        value = scores.get(key)
        if not isinstance(value, int) or not (1 <= value <= 5):
            return False, f"{key} must be int in 1..5"

    justification = judge_json.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        return False, "justification must be non-empty string"
    return True, "ok"


def normalize_judge_json(judge_json: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(judge_json, dict):
        return {}

    out = dict(judge_json)
    scores = out.get("scores") if isinstance(out.get("scores"), dict) else {}
    norm_scores: Dict[str, int] = {}
    for key in JUDGE_SCORE_KEYS:
        value = scores.get(key)
        if isinstance(value, int):
            norm_scores[key] = max(1, min(5, value))
    out["scores"] = norm_scores

    justification = out.get("justification")
    if isinstance(justification, str):
        out["justification"] = justification.strip()
    elif isinstance(justification, (dict, list)):
        out["justification"] = json.dumps(justification, ensure_ascii=False)
    elif justification is None:
        out["justification"] = ""
    else:
        out["justification"] = str(justification).strip()

    return out


def get_bert_scorer() -> Optional[Any]:
    global _bert_scorer, _bert_scorer_unavailable
    if BERTScorer is None or _bert_scorer_unavailable:
        return None
    if _bert_scorer is None:
        try:
            _bert_scorer = BERTScorer(lang="en", rescale_with_baseline=False)
        except Exception:
            _bert_scorer_unavailable = True
            return None
    return _bert_scorer


def simplified_to_text(simplified: Dict[str, Any]) -> str:
    if not isinstance(simplified, dict):
        return ""
    ordered_keys = [
        "T_stage",
        "T_reasoning",
        "N_stage",
        "N_reasoning",
        "M_stage",
        "M_reasoning",
        "Final_TNM",
    ]
    lines: List[str] = []
    for key in ordered_keys:
        value = simplified.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value.strip()}")
    return "\n".join(lines)


def compute_bert_f1_batch(pred_texts: List[str], gt_texts: List[str], batch_size: int = BERT_BATCH_SIZE) -> List[Optional[float]]:
    if not pred_texts or len(pred_texts) != len(gt_texts):
        return []

    scorer = get_bert_scorer()
    if scorer is None:
        return [None] * len(pred_texts)

    outputs: List[Optional[float]] = []
    for i in range(0, len(pred_texts), batch_size):
        pred_batch = pred_texts[i : i + batch_size]
        gt_batch = gt_texts[i : i + batch_size]
        try:
            _, _, f1 = scorer.score(pred_batch, gt_batch)
            outputs.extend(round(float(x.item()), 4) for x in f1)
        except Exception:
            outputs.extend([None] * len(pred_batch))
    return outputs


def fill_bert_f1_for_results(results: Dict[str, Any], target_case_ids: List[str]) -> None:
    case_ids: List[str] = []
    pred_texts: List[str] = []
    gt_texts: List[str] = []

    for case_id in target_case_ids:
        record = results.get(case_id)
        if not isinstance(record, dict) or "error" in record:
            continue

        gt_s = record.get("GT_Simplified") if isinstance(record.get("GT_Simplified"), dict) else None
        pred_s = record.get("Pred_Simplified") if isinstance(record.get("Pred_Simplified"), dict) else None
        if gt_s is None or pred_s is None:
            continue

        case_ids.append(case_id)
        gt_texts.append(simplified_to_text(gt_s))
        pred_texts.append(simplified_to_text(pred_s))

    if not case_ids:
        return

    scores = compute_bert_f1_batch(pred_texts, gt_texts)
    for case_id, score in zip(case_ids, scores):
        record = results.get(case_id)
        if not isinstance(record, dict):
            continue
        post = record.get("Postprocess") if isinstance(record.get("Postprocess"), dict) else {}
        post["BERTScore_F1"] = score
        record["Postprocess"] = post


def get_case_pdf_name(labels: Dict[str, Any]) -> str:
    for key in ["English_file_name", "file_name", "Chinese_file_name"]:
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_ocr_dir(benchmark_dir: str, ocr_root: str, seed: int, ocr_dir: str) -> str:
    if ocr_dir:
        return ocr_dir
    if ocr_root:
        return os.path.join(ocr_root, f"OCR_English_seed{seed}")
    return os.path.join(benchmark_dir, "470_md", f"OCR_English_seed{seed}")


def resolve_ocr_md_path(case_id: str, pdf_name: str, ocr_dir: str) -> str:
    candidates: List[str] = []
    if case_id:
        candidates.append(os.path.join(ocr_dir, f"{case_id}.md"))
    if pdf_name:
        pdf_base = os.path.splitext(os.path.basename(pdf_name))[0]
        candidates.append(os.path.join(ocr_dir, f"{pdf_base}.md"))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"No OCR markdown found for case_id={case_id}, pdf_name={pdf_name}, searched in {ocr_dir}")


def load_ocr_markdown(md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"OCR markdown is empty: {md_path}")
    return text


def strip_think_tags(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE).strip()


def parse_json_relaxed(text: str) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    raw = str(text or "").strip()
    if not raw:
        return {}
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    parsed = extract_json(raw)
    return parsed if isinstance(parsed, dict) else {}


def format_context_list(data_list: Any, default_msg: str) -> str:
    if not isinstance(data_list, list) or not data_list:
        return default_msg
    valid_items = [f"- {str(item)}" for item in data_list if str(item).strip()]
    if not valid_items:
        return default_msg
    return "\n".join(valid_items)


def load_dify_prompts() -> Dict[str, Dict[str, str]]:
    global _dify_prompt_cache
    if _dify_prompt_cache:
        return _dify_prompt_cache

    with open(DIFY_WORKFLOW_FILE, "r", encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    nodes = workflow.get("workflow", {}).get("graph", {}).get("nodes", [])
    cache: Dict[str, Dict[str, str]] = {}
    for node in nodes:
        data = node.get("data", {}) if isinstance(node, dict) else {}
        title = data.get("title")
        prompt_template = data.get("prompt_template")
        if not isinstance(title, str) or not isinstance(prompt_template, list):
            continue
        role_map: Dict[str, str] = {}
        for item in prompt_template:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            text = item.get("text")
            if isinstance(role, str) and isinstance(text, str):
                role_map[role] = text
        if role_map:
            cache[title] = role_map

    _dify_prompt_cache = cache
    return _dify_prompt_cache


def get_dify_prompt(title: str, role: str) -> str:
    prompts = load_dify_prompts()
    text = prompts.get(title, {}).get(role)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Missing Dify prompt: title={title}, role={role}")
    return text


def get_dify_prompt_with_fallback(titles: List[str], role: str) -> str:
    prompts = load_dify_prompts()
    for title in titles:
        text = prompts.get(title, {}).get(role)
        if isinstance(text, str) and text.strip():
            return text
    raise RuntimeError(f"Missing Dify prompt: titles={titles}, role={role}")


def render_dify_template(template: str, replacements: Dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def build_agent_extract_messages(ocr_text: str) -> List[Dict[str, Any]]:
    title_candidates = ["Extract Info", "Information Extraction", "提取信息", "信息提取节点"]
    user_template = get_dify_prompt_with_fallback(title_candidates, "user")
    user_text = render_dify_template(
        user_template,
        {
            "{{#sys.query#}}": "",
            "{{#1756804402464.text#}}": ocr_text,
        },
    )

    return [
        {"role": "system", "content": get_dify_prompt_with_fallback(title_candidates, "system")},
        {"role": "user", "content": user_text},
    ]


def build_agent_t_split_messages(extracted_text: str) -> List[Dict[str, Any]]:
    title_candidates = ["T Routing", "T Routing Node", "T提取", "T分流节点"]
    user_template = get_dify_prompt_with_fallback(title_candidates, "user")
    user_text = render_dify_template(user_template, {"{{#1764323445709.result#}}": extracted_text})
    return [
        {"role": "system", "content": get_dify_prompt_with_fallback(title_candidates, "system")},
        {"role": "user", "content": user_text},
    ]


def build_agent_nm_split_messages(extracted_text: str) -> List[Dict[str, Any]]:
    title_candidates = ["N/M Routing", "N/M Routing Node", "N.M分流节点"]
    user_template = get_dify_prompt_with_fallback(title_candidates, "user")
    user_text = render_dify_template(user_template, {"{{#1764323445709.result#}}": extracted_text})
    return [
        {"role": "system", "content": get_dify_prompt_with_fallback(title_candidates, "system")},
        {"role": "user", "content": user_text},
    ]


def build_t_descriptions_from_split(split_text: str) -> str:
    data = parse_json_relaxed(split_text)
    return format_context_list(data.get("t_assessment_context", []), "No explicit primary-tumor staging evidence mentioned.")


def build_nm_descriptions_from_split(split_text: str) -> Tuple[str, str]:
    data = parse_json_relaxed(split_text)
    n_context = format_context_list(data.get("n_assessment_context", []), "No explicit regional nodal abnormality mentioned.")
    m_context = format_context_list(data.get("m_assessment_context", []), "No explicit distant-metastasis or non-regional nodal abnormality mentioned.")
    return n_context, m_context


def build_agent_stage_messages(dimension: str, stage_context_text: str) -> List[Dict[str, Any]]:
    if dimension not in {"T", "N", "M"}:
        raise ValueError(f"Unsupported dimension: {dimension}")

    title_map = {
        "T": ["T Staging", "T Stage", "T分期", "T分期节点"],
        "N": ["N Staging", "N Stage", "N分期", "N分期节点"],
        "M": ["M Staging", "M Stage", "M分期", "M分期节点"],
    }
    placeholder_map = {
        "T": "{{#1770127481088.t_descriptions#}}",
        "N": "{{#1769604169192.n_descriptions#}}",
        "M": "{{#1769604169192.m_descriptions#}}",
    }
    title_candidates = title_map[dimension]
    user_template = get_dify_prompt_with_fallback(title_candidates, "user")
    user_text = render_dify_template(user_template, {placeholder_map[dimension]: stage_context_text})

    return [
        {"role": "system", "content": get_dify_prompt_with_fallback(title_candidates, "system")},
        {"role": "user", "content": user_text},
    ]


def build_judge_messages(ocr_text: str, gt_simplified: Dict[str, str], pred_simplified: Dict[str, str]) -> List[Dict[str, Any]]:
    gt_simplified_json = json.dumps(gt_simplified, ensure_ascii=False, indent=2)
    pred_simplified_json = json.dumps(pred_simplified, ensure_ascii=False, indent=2)

    judge_prompt = f"""
You are a strict lung-cancer TNM benchmark judge. Evaluate only based on the case OCR markdown text, GT_Simplified, Pred_Simplified, and AJCC 8th-edition lung-cancer TNM rules. Use the OCR text only to cross-check whether the prediction contradicts the source report; scoring should still primarily follow GT_Simplified versus Pred_Simplified.

Case OCR markdown text:
```markdown
{ocr_text}
```

GT_Simplified:
{gt_simplified_json}

Pred_Simplified:
{pred_simplified_json}

Your task:
Score T, N, and M independently on a 1-5 scale.
For each dimension, evaluate:
1. Whether the predicted stage matches the GT stage.
2. Whether the predicted reasoning follows the key evidence logic and workflow requirements for that dimension.

General principles:
- Stage mismatch must be penalized clearly.
- A correct stage does not deserve a 5 if the reasoning is non-compliant or weak.
- Do not be lenient just because the overall answer feels broadly correct.
- Do not credit evidence that is not actually present in GT.
- If uncertain findings are presented as confirmed facts, downgrade by at least one level.
- Output valid JSON only. No extra text.

--------------------------------
1. T_score rubric
--------------------------------
Evaluate:
A. Whether Pred T_stage exactly matches GT T_stage.
B. Whether T_reasoning identifies the decisive evidence for T staging.
C. Whether T_reasoning follows these rules:
   - Prefer label-level evidence wording. Do not reward reasoning that mainly relies on exact measurements or SUV values without translating them into staging-level evidence.
   - If multiple T criteria are satisfied, the reasoning should reflect the workflow: identify all satisfied criteria first, then land on the highest final T stage.
   - Do not mention only size if a higher-level invasion or intrapulmonary spread criterion is what actually determines the GT T stage.
   - Do not use M-level evidence (for example pleural metastasis, chest wall metastasis, contralateral lung metastasis) as T evidence.
   - If GT T_stage is driven by same-lobe spread, different-lobe ipsilateral spread, carcinomatous lymphangitis, or invasion of adjacent structures, the reasoning must explicitly mention that decisive evidence.

T_score scale:
- 5: T_stage fully correct; reasoning captures the decisive evidence, reflects highest-stage logic, and contains no major misuse of evidence or overclaiming.
- 4: T_stage correct; reasoning is mostly correct but slightly incomplete, such as omitting a secondary satisfied criterion or being somewhat weakly phrased.
- 3: T_stage correct but reasoning is clearly weak, or the stage is close with a minor subtype mismatch and partially reasonable evidence.
- 2: T_stage is wrong but reasoning includes partially relevant evidence.
- 1: Severe T-stage error, hallucinated evidence, or major evidence misassignment such as treating M evidence as T evidence.

--------------------------------
2. N_score rubric
--------------------------------
Evaluate:
A. Whether Pred N_stage exactly matches GT N_stage.
B. Whether N_reasoning follows the workflow of full evidence scan plus explicit landing on the highest confirmed N stage.
C. Whether N_reasoning follows these rules:
   - It should reflect scanning of N1, N2, and N3 evidence rather than jumping from one node directly to a conclusion.
   - The reasoning must explicitly land on the final N stage (for example: "therefore the highest confirmed nodal stage is N2").
   - Uncertain / indeterminate / merely enlarged nodes must not be counted as confirmed evidence for the final N stage.
   - Uncertain nodes may be discussed as possible upgrades, but the final stage must be based on confirmed evidence only.
   - Non-regional lymph nodes must not be used as N evidence.

N_score scale:
- 5: N_stage fully correct; reasoning clearly presents the key nodal evidence and explicitly lands on the highest confirmed final N stage; confirmed versus uncertain evidence is handled correctly.
- 4: N_stage correct; reasoning is mostly complete but somewhat weak in full-scan structure or final landing phrasing.
- 3: N_stage correct but reasoning is clearly incomplete, such as listing only partial nodal evidence or failing to land clearly on the final stage; or stage is off by one level with broadly relevant evidence.
- 2: N_stage is wrong but reasoning still contains partially relevant nodal evidence.
- 1: Severe N-stage error, or reasoning incorrectly treats uncertain nodes, non-regional nodes, or wrong laterality as confirmed N evidence.

--------------------------------
3. M_score rubric
--------------------------------
Evaluate:
A. Whether Pred M_stage exactly matches GT M_stage.
B. Whether M_reasoning follows the M0/M1a/M1b/M1c decision path.
C. Whether M_reasoning follows these rules:
   - It should separately check M1a features first: contralateral lung metastasis, pleural metastasis, malignant pleural effusion, malignant pericardial effusion, etc.
   - It should correctly distinguish extra-thoracic organ count and lesion count.
   - Single-organ single-lesion metastasis should map to M1b; single-organ multi-lesion or multi-organ metastasis should map to M1c.
   - Regional lymph nodes must not be used as M evidence.
   - Indeterminate findings, follow-up-only statements, or enlarged/metabolically active findings without metastatic wording must not be treated as confirmed metastasis.

M_score scale:
- 5: M_stage fully correct; reasoning explicitly states the decision path leading to M0, M1a, M1b, or M1c, and handles organ-count / lesion-count logic correctly.
- 4: M_stage correct; reasoning is mostly correct but somewhat incomplete in explaining the decision path or misses minor supporting evidence.
- 3: M_stage correct but reasoning is weak, or stage is close but the M1b versus M1c logic is not clearly expressed.
- 2: M_stage is wrong but reasoning still includes some relevant distant-metastasis evidence.
- 1: Severe M-stage error, or reasoning wrongly treats regional nodes, indeterminate findings, or clearly non-metastatic findings as confirmed metastasis.

--------------------------------
4. Output format
--------------------------------
Output exactly one JSON object:

{{
  "scores": {{
    "T_score": 0,
    "N_score": 0,
    "M_score": 0
  }},
  "justification": {{
    "T": "Explain why T received this score, whether the stage matches, whether decisive evidence was captured, and the main deduction points.",
    "N": "Explain why N received this score, whether the reasoning explicitly lands on the final stage, and whether uncertain or non-regional nodes were mishandled.",
    "M": "Explain why M received this score, whether the M0/M1a/M1b/M1c pathway was handled correctly, and the main deduction points."
  }}
}}
""".strip()

    return [
        {
            "role": "system",
            "content": "You are a strict and objective medical evaluator. Output valid JSON only.",
        },
        {"role": "user", "content": judge_prompt},
    ]


def call_text_until_nonempty(client: OpenAI, model: str, messages: List[Dict[str, Any]], max_attempts: int) -> str:
    last_error = "unknown error"
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
            )
            raw = resp.choices[0].message.content
            text = raw if isinstance(raw, str) else str(raw)
            if text and text.strip():
                return text.strip()
            last_error = "empty content"
        except Exception as e:
            last_error = str(e)
        if attempt < max_attempts:
            time.sleep(min(8, 2 * attempt))
    raise RuntimeError(f"Model call failed after {max_attempts} attempts: {last_error}")


def parse_stage_branch(dimension: str, raw_text: str) -> Dict[str, str]:
    parsed = extract_json(raw_text) if isinstance(raw_text, str) else None
    stage_key = f"{dimension}_stage"
    reason_key = f"{dimension}_reasoning"

    stage = ""
    reasoning = ""

    def _criteria_to_text(items: Any) -> str:
        if not isinstance(items, list):
            return ""
        parts: List[str] = []
        for item in items:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            stage_level = str(item.get("stage_level", "")).strip()
            desc = str(item.get("description", "")).strip()
            if not desc:
                node_name = str(item.get("node_name", "")).strip()
                impact = str(item.get("impact_description") or item.get("potential_impact") or item.get("reason") or "").strip()
                desc = f"{node_name}: {impact}".strip(": ")
            if not desc:
                continue
            parts.append(f"[{stage_level}] {desc}" if stage_level else desc)
        return " ; ".join(parts)

    if isinstance(parsed, dict):
        stage_candidates = [stage_key, f"final_{dimension.lower()}_stage", "stage", "final_stage"]
        reason_candidates = [reason_key, "logic_analysis", "reasoning", "analysis", "rationale"]
        saw_explicit_null_stage = False
        for k in stage_candidates:
            if k not in parsed:
                continue
            v = parsed.get(k)
            if v is None:
                saw_explicit_null_stage = True
                continue
            if isinstance(v, str):
                v_clean = v.strip()
                if not v_clean:
                    continue
                if v_clean.lower() in {"null", "none", "unknown", "na", "n/a"}:
                    saw_explicit_null_stage = True
                    continue
                stage = v_clean
                break
        for k in reason_candidates:
            v = parsed.get(k)
            if isinstance(v, str) and v.strip():
                reasoning = v.strip()
                break

        if not reasoning:
            criteria_text = _criteria_to_text(parsed.get("matched_criteria"))
            uncertain_text = _criteria_to_text(parsed.get("uncertain_nodes") or parsed.get("uncertain_findings"))
            pieces: List[str] = []
            if criteria_text:
                pieces.append(f"Matched criteria: {criteria_text}.")
            if uncertain_text:
                pieces.append(f"Uncertain findings: {uncertain_text}.")
            if pieces:
                reasoning = " ".join(pieces)

        if not stage and saw_explicit_null_stage:
            stage = f"{dimension}x"
            if not reasoning:
                reasoning = f"No valid {dimension} staging evidence was identified from the provided branch context."

    if not stage:
        m = re.search(rf"(?im)^\s*{re.escape(stage_key)}\s*[:：]\s*(.+?)\s*$", raw_text)
        if m:
            stage = m.group(1).strip().strip('"').strip("'").strip()
    if not reasoning:
        m = re.search(rf"(?im)^\s*{re.escape(reason_key)}\s*[:：]\s*(.+?)\s*$", raw_text)
        if m:
            reasoning = m.group(1).strip().strip('"').strip("'").strip()

    if not stage:
        lower_text = raw_text.lower()
        no_evidence_markers = [
            f"no valid {dimension.lower()} evidence",
            f"no {dimension.lower()} evidence",
            f"cannot determine {dimension.lower()}",
            f"unable to determine {dimension.lower()}",
            f"{dimension.lower()} stage cannot be determined",
            f"{dimension.lower()} cannot be assessed",
        ]
        if any(marker in lower_text for marker in no_evidence_markers):
            stage = f"{dimension}x"
            if not reasoning:
                reasoning = f"No valid {dimension} staging evidence was identified from the provided branch context."

    if stage and not reasoning:
        reasoning = f"Final {dimension} stage determined as {stage} based on extracted criteria."

    return {stage_key: stage, reason_key: reasoning}


def run_stage_branch(client: OpenAI, dimension: str, stage_context_text: str) -> Dict[str, str]:
    messages = build_agent_stage_messages(dimension, stage_context_text)
    raw_text = call_text_until_nonempty(client, AGENT_STAGE_MODEL, messages, MAX_STAGE_ATTEMPTS)
    clean_text = strip_think_tags(raw_text)
    branch = parse_stage_branch(dimension, clean_text)
    stage_key = f"{dimension}_stage"
    reason_key = f"{dimension}_reasoning"
    if not branch.get(stage_key):
        raise RuntimeError(f"{dimension} branch output invalid after parsing: parsed={branch}, raw={clean_text[:240]}")
    return branch


def call_runner_until_valid(client: OpenAI, ocr_text: str) -> Dict[str, str]:
    last_error = "unknown error"

    for attempt in range(1, MAX_RUNNER_ATTEMPTS + 1):
        try:
            # Node 1: extraction
            extract_msgs = build_agent_extract_messages(ocr_text)
            extract_raw = call_text_until_nonempty(client, AGENT_EXTRACT_MODEL, extract_msgs, MAX_EXTRACT_ATTEMPTS)
            extract_clean = strip_think_tags(extract_raw)
            if not extract_clean.strip():
                raise RuntimeError("Extractor output is empty after think-tag stripping")

            # Node 2A: T routing + think filter + code node
            t_split_raw = call_text_until_nonempty(client, AGENT_EXTRACT_MODEL, build_agent_t_split_messages(extract_clean), MAX_EXTRACT_ATTEMPTS)
            t_split_clean = strip_think_tags(t_split_raw)
            t_descriptions = build_t_descriptions_from_split(t_split_clean)

            # Node 2B: NM routing + think filter + code node
            nm_split_raw = call_text_until_nonempty(client, AGENT_EXTRACT_MODEL, build_agent_nm_split_messages(extract_clean), MAX_EXTRACT_ATTEMPTS)
            nm_split_clean = strip_think_tags(nm_split_raw)
            n_descriptions, m_descriptions = build_nm_descriptions_from_split(nm_split_clean)

            # Node 3: T/N/M staging
            merged: Dict[str, str] = {}
            stage_inputs = {"T": t_descriptions, "N": n_descriptions, "M": m_descriptions}
            dims = ["T", "N", "M"]
            if AGENT_PARALLEL_BRANCHES:
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = {pool.submit(run_stage_branch, client, dim, stage_inputs[dim]): dim for dim in dims}
                    for future in futures:
                        merged.update(future.result())
            else:
                for dim in dims:
                    merged.update(run_stage_branch(client, dim, stage_inputs[dim]))

            normalized = normalize_simplified_output(merged)
            ok, err = validate_simplified_output(normalized)
            if ok:
                return normalized
            last_error = err
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_RUNNER_ATTEMPTS:
            time.sleep(min(8, 2 * attempt))

    raise RuntimeError(f"Runner-Agent failed after {MAX_RUNNER_ATTEMPTS} attempts: {last_error}")


def call_judge_until_valid(client: OpenAI, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    last_error = "unknown error"
    for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content
            parsed = extract_json(raw if isinstance(raw, str) else str(raw))
            if not parsed:
                last_error = "judge output is not valid JSON"
                continue
            parsed = normalize_judge_json(parsed)
            ok, err = validate_judge_json(parsed)
            if ok:
                return parsed
            last_error = err
        except Exception as e:
            last_error = str(e)
        if attempt < MAX_JUDGE_ATTEMPTS:
            time.sleep(2)

    raise RuntimeError(f"Judge failed after {MAX_JUDGE_ATTEMPTS} attempts: {last_error}")


def resolve_gt_file(user_supplied: str, benchmark_dir: str) -> str:
    if user_supplied and os.path.exists(user_supplied):
        return user_supplied

    candidates = [
        os.path.join(benchmark_dir, "gt", "benchmark_gt_English.json"),
        os.path.join(benchmark_dir, "gt", "benchmark_gt_English_seed42.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    globbed = sorted(glob.glob(os.path.join(benchmark_dir, "gt", "benchmark_gt_English_seed*.json")))
    if globbed:
        return globbed[0]

    raise FileNotFoundError("No English GT file found under 32s/gt")


def summarize_results(results: Dict[str, Any], target_case_ids: List[str]) -> Dict[str, Any]:
    t_correct = 0
    n_correct = 0
    m_correct = 0
    final_correct = 0
    score_acc = {"T_score": [], "N_score": [], "M_score": []}
    bert_f1_scores: List[float] = []
    evaluated = 0

    for case_id in target_case_ids:
        record = results.get(case_id)
        if not isinstance(record, dict):
            continue
        if "error" in record:
            continue

        gt_s = record.get("GT_Simplified")
        pred_s = record.get("Pred_Simplified")
        judge = record.get("Evaluation_Report")
        if not isinstance(gt_s, dict) or not isinstance(pred_s, dict) or not isinstance(judge, dict):
            continue

        evaluated += 1
        t_correct += int(str(gt_s.get("T_stage", "")).upper() == str(pred_s.get("T_stage", "")).upper())
        n_correct += int(str(gt_s.get("N_stage", "")).upper() == str(pred_s.get("N_stage", "")).upper())
        m_correct += int(str(gt_s.get("M_stage", "")).upper() == str(pred_s.get("M_stage", "")).upper())
        final_correct += int(str(gt_s.get("Final_TNM", "")).upper() == str(pred_s.get("Final_TNM", "")).upper())

        scores = judge.get("scores") if isinstance(judge.get("scores"), dict) else {}
        for key in score_acc:
            val = scores.get(key)
            if isinstance(val, int):
                score_acc[key].append(val)

        post = record.get("Postprocess") if isinstance(record.get("Postprocess"), dict) else {}
        bert_f1 = post.get("BERTScore_F1")
        if isinstance(bert_f1, (int, float)):
            bert_f1_scores.append(float(bert_f1))

    def avg(values: List[int]) -> Optional[float]:
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    return {
        "total_cases_targeted": len(target_case_ids),
        "total_cases_evaluated": evaluated,
        "T_accuracy": round(t_correct / evaluated, 4) if evaluated else None,
        "N_accuracy": round(n_correct / evaluated, 4) if evaluated else None,
        "M_accuracy": round(m_correct / evaluated, 4) if evaluated else None,
        "Final_TNM_accuracy": round(final_correct / evaluated, 4) if evaluated else None,
        "Average_T_score": avg(score_acc["T_score"]),
        "Average_N_score": avg(score_acc["N_score"]),
        "Average_M_score": avg(score_acc["M_score"]),
        "Average_BERTScore_F1": round(sum(bert_f1_scores) / len(bert_f1_scores), 4) if bert_f1_scores else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=32, help="Number of cases to run. Use -1 for all.")
    parser.add_argument("--benchmark-dir", type=str, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--gt-file", type=str, default="")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--runner-api-key", type=str, default=os.getenv("RUNNER_API_KEY", ""))
    parser.add_argument("--judge-api-key", type=str, default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--ocr-dir", type=str, default=os.getenv("OCR_DIR", ""))
    parser.add_argument("--ocr-root", type=str, default=os.getenv("OCR_ROOT", os.path.join(DEFAULT_BENCHMARK_DIR, "470_md")))
    parser.add_argument("--ocr-seed", type=int, default=int(os.getenv("OCR_SEED", "42")))
    args = parser.parse_args()

    if not args.runner_api_key:
        raise ValueError("Missing RUNNER_API_KEY")
    if not args.judge_api_key:
        raise ValueError("Missing JUDGE_API_KEY")

    gt_file = resolve_gt_file(args.gt_file, args.benchmark_dir)
    print(f"Using GT file: {gt_file}")
    ocr_dir = resolve_ocr_dir(args.benchmark_dir, args.ocr_root, args.ocr_seed, args.ocr_dir)
    if not os.path.isdir(ocr_dir):
        raise FileNotFoundError(f"OCR directory not found: {ocr_dir}")
    print(f"Using OCR dir: {ocr_dir}")

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    case_items = [(k, v) for k, v in gt_data.items() if isinstance(v, dict) and k.startswith("LC_")]
    selected = case_items if args.num == -1 else case_items[: args.num]
    target_case_ids = [cid for cid, _ in selected]

    results: Dict[str, Any] = {}
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded existing output: {args.output_file}")
        except Exception:
            results = {}

    runner_client = make_runner_client(args.runner_api_key)
    judge_client = make_judge_client(args.judge_api_key)

    todo = [
        (cid, item)
        for cid, item in selected
        if cid not in results or not isinstance(results.get(cid), dict) or "error" in results.get(cid, {})
    ]
    print(f"Cases to run: {len(todo)}")

    for case_id, item in tqdm(todo):
        gt_raw = item.get("Ground_Truth_Raw", {}) if isinstance(item.get("Ground_Truth_Raw"), dict) else {}
        gt_simplified = item.get("Simplified_GT", {}) if isinstance(item.get("Simplified_GT"), dict) else {}

        pdf_name = get_case_pdf_name(gt_raw)

        try:
            md_path = resolve_ocr_md_path(case_id, pdf_name, ocr_dir)
            ocr_text = load_ocr_markdown(md_path)
            pred_simplified = call_runner_until_valid(runner_client, ocr_text)

            judge_msgs = build_judge_messages(ocr_text, gt_simplified, pred_simplified)
            judge_result = call_judge_until_valid(judge_client, judge_msgs)

            results[case_id] = {
                "GT_Simplified": gt_simplified,
                "Pred_Simplified": pred_simplified,
                "Evaluation_Report": judge_result,
                "Postprocess": {},
            }
        except Exception as e:
            results[case_id] = {"error": str(e)}

        atomic_save_json(results, args.output_file)

    fill_bert_f1_for_results(results, target_case_ids)
    atomic_save_json(results, args.output_file)

    results["_Summary"] = summarize_results(results, target_case_ids)
    atomic_save_json(results, args.output_file)
    print(json.dumps(results["_Summary"], ensure_ascii=False, indent=2))
    print(f"Done. Output: {args.output_file}")


if __name__ == "__main__":
    main()
