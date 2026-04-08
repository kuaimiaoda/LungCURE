import argparse
import ast
import base64
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pdf2image import convert_from_path
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
    os.path.join(DEFAULT_BENCHMARK_DIR, "my_framework", "results_TNMstaging", "results_Agent", f"{RUNNER_MODEL.split('/')[-1]}_Chinese_agent_simplified.json"),
)
DEFAULT_MAX_PAGES = int(os.getenv("MAX_PAGES", "6"))
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

DIFY_WORKFLOW_FILE = os.path.join(DEFAULT_BENCHMARK_DIR, "my_framework", "LCAgent", "TNM_staging", "tnm分期助手_分开_格式化_20260327.yml")


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
        if not contains_cjk(str(data.get(key, ""))):
            return False, f"{key} must be written in Chinese"

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
            _bert_scorer = BERTScorer(lang="zh", rescale_with_baseline=False)
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


def encode_image(image) -> str:
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_pdf_to_images(pdf_path: str, max_pages: int = 0) -> List[Any]:
    try:
        images = convert_from_path(pdf_path, dpi=200)
        if max_pages > 0:
            return images[:max_pages]
        return images
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return []


def get_case_pdf_name(labels: Dict[str, Any]) -> str:
    for key in ["Chinese_file_name", "file_name", "English_file_name"]:
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


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


def render_dify_template(template: str, replacements: Dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def build_agent_extract_messages(images: List[Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for img in images:
        base64_image = encode_image(img)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    user_template = get_dify_prompt("提取信息", "user")
    user_text = render_dify_template(
        user_template,
        {
            "{{#sys.query#}}": "",
            "{{#1756804402464.text#}}": "请根据上传的报告图像内容进行提取。",
        },
    )
    content.append({"type": "text", "text": user_text})

    return [
        {"role": "system", "content": get_dify_prompt("提取信息", "system")},
        {"role": "user", "content": content},
    ]


def build_agent_t_split_messages(extracted_text: str) -> List[Dict[str, Any]]:
    user_template = get_dify_prompt("T提取", "user")
    user_text = render_dify_template(user_template, {"{{#1764323445709.result#}}": extracted_text})
    return [
        {"role": "system", "content": get_dify_prompt("T提取", "system")},
        {"role": "user", "content": user_text},
    ]


def build_agent_nm_split_messages(extracted_text: str) -> List[Dict[str, Any]]:
    user_template = get_dify_prompt("N.M分流节点", "user")
    user_text = render_dify_template(user_template, {"{{#1764323445709.result#}}": extracted_text})
    return [
        {"role": "system", "content": get_dify_prompt("N.M分流节点", "system")},
        {"role": "user", "content": user_text},
    ]


def build_t_descriptions_from_split(split_text: str) -> str:
    data = parse_json_relaxed(split_text)
    return format_context_list(data.get("t_assessment_context", []), "未提及原发灶关键分期信息。")


def build_nm_descriptions_from_split(split_text: str) -> Tuple[str, str]:
    data = parse_json_relaxed(split_text)
    n_context = format_context_list(data.get("n_assessment_context", []), "未提及区域淋巴结相关异常。")
    m_context = format_context_list(data.get("m_assessment_context", []), "未提及远处转移或非区域淋巴结异常。")
    return n_context, m_context


def build_agent_stage_messages(dimension: str, stage_context_text: str) -> List[Dict[str, Any]]:
    if dimension not in {"T", "N", "M"}:
        raise ValueError(f"Unsupported dimension: {dimension}")

    title_map = {"T": "T分期", "N": "N分期", "M": "M分期"}
    placeholder_map = {
        "T": "{{#1770127481088.t_descriptions#}}",
        "N": "{{#1769604169192.n_descriptions#}}",
        "M": "{{#1769604169192.m_descriptions#}}",
    }
    title = title_map[dimension]
    user_template = get_dify_prompt(title, "user")
    user_text = render_dify_template(user_template, {placeholder_map[dimension]: stage_context_text})

    return [
        {"role": "system", "content": get_dify_prompt(title, "system")},
        {"role": "user", "content": user_text},
    ]

def build_judge_messages(images: List[Any], gt_simplified: Dict[str, str], pred_simplified: Dict[str, str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for img in images:
        base64_image = encode_image(img)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    gt_simplified_json = json.dumps(gt_simplified, ensure_ascii=False, indent=2)
    pred_simplified_json = json.dumps(pred_simplified, ensure_ascii=False, indent=2)

    judge_prompt = f"""
你是肺癌 TNM benchmark 的严格评测裁判。请只基于以下 GT_Simplified、Pred_Simplified 以及 AJCC 第八版肺癌 TNM 分期规则进行评估；可结合图像仅用于核对是否存在明显与报告相矛盾的情况，但评分以 GT_Simplified 和 Pred_Simplified 为主。

GT_Simplified:
{gt_simplified_json}

Pred_Simplified:
{pred_simplified_json}

你的任务：
分别对 T、N、M 三个维度独立打分，每项 1-5 分。
请重点评估两件事：
1. Pred 的 stage 是否与 GT 一致。
2. Pred 的 reasoning 是否符合该维度的关键证据逻辑与工作流要求。

总原则：
- stage 错误必须明显扣分。
- stage 正确但 reasoning 不符合该维度规则，不能给 5 分。
- 不得因为整体看起来“差不多”而放松对单维度规则的要求。
- 不得把 GT 中不存在的证据说成 Pred 正确引用的依据。
- 若 Pred 把 uncertain / 待定信息当作 confirmed 事实，至少降 1 档。
- 输出必须是合法 JSON，不要输出任何额外文字。

--------------------------------
一、T_score 评分规则
--------------------------------
评估重点：
A. T_stage 是否与 GT 完全一致。
B. T_reasoning 是否抓住决定 T 分期的关键证据。
C. T_reasoning 是否符合以下规则：
   - 应基于标签级证据表达，不要因为 Pred 复述精确测量值/SUV 就给高分；若 reasoning 主要依赖精确数值复述而缺少标签级结论，需降分。
   - 如果病例同时满足多个 T 标准，reasoning 应体现“先识别全部已满足标准，再按最高级别确定 final T_stage”的思路。
   - 不应只说大小而漏掉更高级别的侵犯/播散证据。
   - 不应把属于 M 的证据（如胸膜转移、胸壁转移、对侧肺转移）误当成 T 依据。
   - 若 GT 的 T_stage 是由同侧同肺叶播散 / 同侧不同肺叶播散 / 癌性淋巴管炎 / 邻近结构侵犯等决定，reasoning 必须明确提到该决定性证据，而不只是泛泛说“肿瘤较大”。

T_score 细则：
- 5分：T_stage 完全正确；T_reasoning 命中决定性依据，体现最高级别原则，无明显误用证据、过度推断或格式性缺陷。
- 4分：T_stage 正确；T_reasoning 基本正确，但略有不完整，例如漏掉一个次要已满足标准，或表达偏弱但仍抓住决定性依据。
- 3分：T_stage 正确但 reasoning 明显薄弱，或 stage 只差一个子级且 reasoning 有部分合理依据。
- 2分：T_stage 错误，但 reasoning 仍有部分相关证据或方向接近。
- 1分：T_stage 严重错误，或 reasoning 存在明显幻觉、关键证据错配、把 M 证据当 T 证据等严重问题。

--------------------------------
二、N_score 评分规则
--------------------------------
评估重点：
A. N_stage 是否与 GT 完全一致。
B. N_reasoning 是否符合“全量扫描 + 最高 confirmed stage 落点”的规则。
C. N_reasoning 是否符合以下规则：
   - 应体现对 N1/N2/N3 证据的扫描，而不是只提一个节点就直接跳结论。
   - reasoning 最终必须明确落到最终 N_stage（例如“因此最高已确认淋巴结分期为 N2”）。
   - 不应把 uncertain / 性质待定 / 稍大但未明确转移 的节点计入最终 confirmed N_stage。
   - 若存在 uncertain node，可以讨论其潜在升级方向，但 final stage 必须基于 confirmed evidence。
   - 不应把非区域淋巴结误算为 N 分期依据。

N_score 细则：
- 5分：N_stage 完全正确；reasoning 明确呈现关键淋巴结证据，并明确落到最终最高 confirmed N_stage；confirmed 与 uncertain 区分清楚。
- 4分：N_stage 正确；reasoning 基本完整，但全量扫描意识稍弱，或 final landing 不够有力但仍可清楚判断其落点。
- 3分：N_stage 正确但 reasoning 明显不完整，例如只列局部证据、未清楚落到最终 stage；或 stage 只差一级但证据方向基本对。
- 2分：N_stage 错误，但 reasoning 仍提到部分相关淋巴结证据。
- 1分：N_stage 严重错误，或将 uncertain / 非区域淋巴结 / 错误侧别直接当作 confirmed N 依据。

--------------------------------
三、M_score 评分规则
--------------------------------
评估重点：
A. M_stage 是否与 GT 完全一致。
B. M_reasoning 是否符合 M1a / M1b / M1c 的判定路径。
C. M_reasoning 是否符合以下规则：
   - 应先识别 M1a 相关证据：对侧肺、胸膜转移、恶性胸腔积液、恶性心包积液等。
   - 应明确区分肺外器官/非区域淋巴结的器官数与灶数。
   - 单器官单灶应对应 M1b；单器官多灶或多器官应对应 M1c。
   - 不应把区域淋巴结误当作 M 证据。
   - 不应把“待定”“随诊观察”“仅肿大/代谢增高但无转移表述”的病灶直接算作 confirmed metastasis。

M_score 细则：
- 5分：M_stage 完全正确；M_reasoning 明确说明决定该 stage 的路径（M0 / M1a / M1b / M1c），器官数/灶数逻辑正确，无明显误判。
- 4分：M_stage 正确；reasoning 基本正确，但对判定路径说明不够完整，或遗漏次要伴随证据。
- 3分：M_stage 正确但 reasoning 较弱，或 stage 接近正确但对 M1b/M1c 的逻辑表达不清。
- 2分：M_stage 错误，但 reasoning 中仍包含部分相关远处转移证据。
- 1分：M_stage 严重错误，或把区域淋巴结/待定病灶/无转移表述错误算作 confirmed metastasis。

--------------------------------
四、输出要求
--------------------------------
只输出一个 JSON 对象，格式如下：

{{
    "scores": {{
        "T_score": 0,
        "N_score": 0,
        "M_score": 0
    }},
    "justification": {{
        "T": "说明 T 为什么给这个分，指出 stage 是否一致、关键证据是否命中、主要扣分点。",
        "N": "说明 N 为什么给这个分，指出是否明确落到最终 stage、是否错误使用 uncertain 或非区域节点。",
        "M": "说明 M 为什么给这个分，指出是否正确区分 M0/M1a/M1b/M1c，以及主要扣分点。"
    }}
}}
""".strip()
    content.append({"type": "text", "text": judge_prompt})

    return [
        {
            "role": "system",
            "content": "你是严格客观的医学评测裁判，只输出合法 JSON。",
        },
        {"role": "user", "content": content},
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
                desc = f"{node_name}：{impact}".strip("：")
            if not desc:
                continue
            parts.append(f"[{stage_level}] {desc}" if stage_level else desc)
        return "；".join(parts)

    if isinstance(parsed, dict):
        stage_candidates = [stage_key, f"final_{dimension.lower()}_stage", "stage", "final_stage"]
        reason_candidates = [reason_key, "logic_analysis", "reasoning", "analysis", "rationale"]
        for k in stage_candidates:
            v = parsed.get(k)
            if isinstance(v, str) and v.strip():
                stage = v.strip()
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
                pieces.append(f"匹配依据：{criteria_text}。")
            if uncertain_text:
                pieces.append(f"不确定项：{uncertain_text}。")
            if pieces:
                reasoning = "".join(pieces)

    if not stage:
        m = re.search(rf"(?im)^\s*{re.escape(stage_key)}\s*[:：]\s*(.+?)\s*$", raw_text)
        if m:
            stage = m.group(1).strip().strip('"').strip("'").strip()
    if not reasoning:
        m = re.search(rf"(?im)^\s*{re.escape(reason_key)}\s*[:：]\s*(.+?)\s*$", raw_text)
        if m:
            reasoning = m.group(1).strip().strip('"').strip("'").strip()

    if stage and not reasoning:
        reasoning = f"依据提取证据，最终判定 {dimension} 分期为 {stage}。"

    return {stage_key: stage, reason_key: reasoning}


def run_stage_branch(client: OpenAI, dimension: str, stage_context_text: str) -> Dict[str, str]:
    messages = build_agent_stage_messages(dimension, stage_context_text)
    raw_text = call_text_until_nonempty(client, AGENT_STAGE_MODEL, messages, MAX_STAGE_ATTEMPTS)
    branch = parse_stage_branch(dimension, strip_think_tags(raw_text))
    stage_key = f"{dimension}_stage"
    reason_key = f"{dimension}_reasoning"
    if not branch.get(stage_key):
        raise RuntimeError(f"{dimension} branch output invalid: {raw_text[:240]}")
    return branch


def call_runner_until_valid(client: OpenAI, images: List[Any]) -> Dict[str, str]:
    last_error = "unknown error"

    for attempt in range(1, MAX_RUNNER_ATTEMPTS + 1):
        try:
            # 节点1: 提取信息
            extract_msgs = build_agent_extract_messages(images)
            extract_raw = call_text_until_nonempty(client, AGENT_EXTRACT_MODEL, extract_msgs, MAX_EXTRACT_ATTEMPTS)
            extract_clean = strip_think_tags(extract_raw)
            if not extract_clean.strip():
                raise RuntimeError("Extractor output is empty after think-tag stripping")

            # 节点2A: T分流 + 过滤think + 代码节点9
            t_split_raw = call_text_until_nonempty(client, AGENT_EXTRACT_MODEL, build_agent_t_split_messages(extract_clean), MAX_EXTRACT_ATTEMPTS)
            t_split_clean = strip_think_tags(t_split_raw)
            t_descriptions = build_t_descriptions_from_split(t_split_clean)

            # 节点2B: NM分流 + 过滤think + 代码节点7
            nm_split_raw = call_text_until_nonempty(client, AGENT_EXTRACT_MODEL, build_agent_nm_split_messages(extract_clean), MAX_EXTRACT_ATTEMPTS)
            nm_split_clean = strip_think_tags(nm_split_raw)
            n_descriptions, m_descriptions = build_nm_descriptions_from_split(nm_split_clean)

            # 节点3: T/N/M 分期
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
        os.path.join(benchmark_dir, "gt", "benchmark_gt_Chinese.json"),
        os.path.join(benchmark_dir, "gt", "benchmark_gt_Chinese_seed42.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    globbed = sorted(glob.glob(os.path.join(benchmark_dir, "gt", "benchmark_gt_Chinese_seed*.json")))
    if globbed:
        return globbed[0]

    raise FileNotFoundError("No Chinese GT file found under 32s/gt")


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
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    args = parser.parse_args()

    if not args.runner_api_key:
        raise ValueError("Missing RUNNER_API_KEY")
    if not args.judge_api_key:
        raise ValueError("Missing JUDGE_API_KEY")

    gt_file = resolve_gt_file(args.gt_file, args.benchmark_dir)
    print(f"Using GT file: {gt_file}")

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
        pdf_path = os.path.join(args.benchmark_dir, pdf_name)
        images = process_pdf_to_images(pdf_path, max_pages=args.max_pages)

        if not images:
            results[case_id] = {"error": f"PDF cannot be converted to images: {pdf_path}"}
            atomic_save_json(results, args.output_file)
            continue

        try:
            pred_simplified = call_runner_until_valid(runner_client, images)

            judge_msgs = build_judge_messages(images, gt_simplified, pred_simplified)
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
