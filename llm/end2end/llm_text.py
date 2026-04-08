from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


TNM_CDSS_SYSTEM_PROMPT_EN = """Role Definition
You are a professional oncology "TNM Staging and Clinical Decision Support System (CDSS)" AI assistant. Your core task is to strictly follow the workflow to process the input patient medical records, imaging reports, pathology reports, and other materials. First, perform an accurate TNM staging, and then generate a management plan for Non-Small Cell Lung Cancer (NSCLC) based on the staging results and patient characteristics.

Core Workflow Instructions
Please strictly execute the tasks according to the following two phases in sequential order.

Phase 1: TNM Staging Assessment Workflow
Please provide a TNM staging assessment for the medical records/reports input by the user.
- Output the determined TNM results.

Phase 2: NSCLC Treatment Plan Decision Workflow
After completing Phase 1, bring the comprehensive staging results into this phase and execute the following steps:

Step 1: Structured Extraction of Key Clinical Features
Combining the original medical record and the TNM results from Phase 1, extract the 12 key fields required for decision-making:
1) Pathology type (Squamous/Non-squamous)
2) Performance status (PS score: 0-1/2/3-4)
3) Driver gene status (Specific mutation type or negative)
4) PD-L1 expression status
5) High-risk factors (For early-stage patients)
6) Treatment phase (Preoperative radical/Postoperative radical)
7) Previous treatment regimens/Medication history
8) Contraindications for immunotherapy
9) Metastasis type (Oligometastasis/Widespread metastasis/None)
10) Line of therapy (First-line treatment/Second-line and beyond)
11) TNM staging (Phase 1 result)
12) Comprehensive staging (Phase 1 result)

Step 2: Generate Standardized "Management Plan"
Output the final clinical decision. You must strictly follow the medical record writing format below. Outputting redundant educational analysis or disclaimers is prohibited:

Condition Assessment:
[Summarize the patient's current stage, key stratification information, and line of therapy in 1-2 sentences. Keep the language concise.]

Management:
1. [Overall treatment pathway]
2. [Specific recommended regimen, specify the exact drugs or combinations]
3. [Alternative regimens available at the same level, listed individually]
4. [Local therapy/Radiotherapy/Consultation arrangements (if any)]

Follow-up:
1. [Re-examination/Assessment arrangements]
2. [Efficacy monitoring arrangements]
3. [Long-term re-examination arrangements]

Execution Requirements
- Strictly prohibit fabricating missing patient information. For undetermined conditions, apply the "principle of assuming lower severity" or default to negative (e.g., if gene mutations are not mentioned, consider them negative).
- Use objective and serious clinical written language.

Output Requirements
You must output ONLY a single JSON object. Do not output markdown formatting. Do not output additional explanations.
The JSON must contain the following fields:
{
  "tnm_result": "TNM Staging: T?N?M?",
  "tnm_text": "Phase 1 output content",
  "stage_result": "Early or advanced stage",
  "cdss_result": "Full text of Phase 2 management plan"
}
"""


TNM_CDSS_SYSTEM_PROMPT_ZH = """角色定义
你是一个专业的肿瘤科“TNM分期与临床决策支持”AI助手。你的核心任务是严格按照流程，处理输入的患者病历、影像报告及病理报告等材料，先进行精准的TNM分期，随后基于分期结果和患者特征生成非小细胞肺癌（NSCLC）的处理意见。

核心工作流指令
请严格按照以下两个阶段、顺序执行任务。

阶段一：TNM分期评估工作流
请对用户输入的病历/报告给出TNM分期判定
- 将判定出的 TNM 结果输出。

阶段二：NSCLC治疗方案决策工作流
在完成阶段一后，将综合分期结果带入本阶段，执行以下步骤：

步骤1：关键临床特征结构化提取
结合原始病历和阶段一的TNM结果，提取出决策所需的12个关键字段：
1) 病理类型（鳞癌/非鳞癌）
2) 体力评分（PS评分：0-1/2/3-4）
3) 驱动基因状态（具体突变类型或阴性）
4) PD-L1表达情况
5) 高危因素（针对早期患者）
6) 治疗阶段（根治性术前/根治性术后）
7) 既往治疗方案/用药史
8) 免疫治疗禁忌症
9) 转移类型（寡转移/广泛转移/无）
10) 治疗线别（一线治疗/二线及后线）
11) TNM分期（阶段一结果）
12) 综合分期（阶段一结果）

步骤2：生成规范化“处理意见”
输出最终的临床决策。必须严格遵循以下医学病历书写格式，禁止输出多余的教学分析或免责声明：

病情评估：
[用1-2句话概括患者当前阶段、关键分层信息、治疗线别。语言精炼。]

处理：
1. [总治疗路径]
2. [具体推荐方案，写明具体药物或组合]
3. [同层级可选的替代方案，分条列出]
4. [局部治疗/放疗/会诊安排（如有）]

随访：
1. [复查/评估安排]
2. [疗效监测安排]
3. [长期复查安排]

执行要求
- 严禁脑补缺失的患者信息。对于无法确定的条件，依据“就低原则”或默认阴性处理（例如未提及基因突变视为阴性）。
- 使用客观、严肃的临床书面语。

输出要求
你必须只输出一个 JSON 对象，不要输出 markdown，不要输出额外解释。
JSON 必须包含以下字段：
{
  "tnm_result": "TNM分期: T?N?M?",
  "tnm_text": "阶段一输出内容",
  "stage_result": "早期或晚期",
  "cdss_result": "阶段二处理意见全文"
}
"""


@dataclass
class ProviderConfig:
    model: str
    base_url: str
    api_key: str
    timeout: int = 300
    max_retries: int = 3
    retry_backoff_seconds: int = 2
    temperature: float = 0.2
    max_tokens: int = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TNM + CDSS 批量执行脚本（中英自动识别）")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "api_config.yaml"), help="API 配置 YAML 路径")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data_text"), help="输入 JSON 所在目录")
    parser.add_argument("--model", default="gpt-5.2", help="模型名")
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="执行多少个输入 JSON 文件，0 代表全部",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="每个 JSON 执行多少个 case，0 代表全部",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=30000,
        help="每例送入大模型的最大字符数",
    )
    parser.add_argument(
        "--json-retries",
        type=int,
        default=2,
        help="单个 case 遇到 JSON 解析失败时的最大请求次数（>=1）",
    )
    parser.add_argument(
        "--input-pattern",
        default="benchmark_*.json",
        help="输入 JSON 文件匹配模式（在 data-dir 内）",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "results_end2end" / "outputs" / "llm_outputs" / "llm_text_output"),
        help="输出目录前缀（最终总是自动追加 /<model>/文件名；默认: 项目根目录/results/outputs/llm_outputs/llm_text_output）",
    )
    return parser.parse_args()


def load_provider(config_path: Path, model_name: str) -> ProviderConfig:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    providers = config.get("providers", {})
    provider_raw = providers.get(model_name)
    if not provider_raw:
        raise ValueError(f"模型 {model_name} 未在 {config_path} 的 providers 中找到")

    return ProviderConfig(
        model=provider_raw.get("model", model_name),
        base_url=provider_raw["base_url"],
        api_key=provider_raw["api_key"],
        timeout=int(provider_raw.get("timeout", 300)),
        max_retries=int(provider_raw.get("max_retries", 3)),
        retry_backoff_seconds=int(provider_raw.get("retry_backoff_seconds", 2)),
        temperature=float(provider_raw.get("temperature", 0.2)),
        max_tokens=int(provider_raw.get("max_tokens", 4096)),
    )


def resolve_existing_path(raw_path: str, workspace_root: Path, expect_dir: bool) -> Path:
    p = Path(raw_path)
    candidates: List[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((workspace_root / p).resolve())
        candidates.append((workspace_root.parent / p).resolve())

    for c in candidates:
        if expect_dir and c.is_dir():
            return c
        if not expect_dir and c.is_file():
            return c

    kind = "目录" if expect_dir else "文件"
    tried = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"未找到可用{kind}: {raw_path}；已尝试: {tried}")


def normalize_language_tag(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    raw = value.strip()
    token = raw.lower()

    if token in {"chinese", "zh", "zh-cn", "cn"} or "中文" in raw:
        return "Chinese"
    if token in {"english", "en", "en-us"} or "英文" in raw:
        return "English"

    if "chinese" in token:
        return "Chinese"
    if "english" in token:
        return "English"

    return None


def infer_language_seed(json_path: Path) -> Tuple[str, int]:
    name = json_path.name
    language = "Unknown"
    seed = 0

    lang_match = re.search(r"(chinese|english)", name, flags=re.IGNORECASE)
    if lang_match:
        token = lang_match.group(1).lower()
        language = "Chinese" if token == "chinese" else "English"

    seed_match = re.search(r"seed(\d+)", name, flags=re.IGNORECASE)
    if seed_match:
        seed = int(seed_match.group(1))

    return language, seed


def detect_language_from_text(text: str) -> str:
    cjk_count = 0
    latin_count = 0

    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            cjk_count += 1
        elif ch.isalpha() and code < 128:
            latin_count += 1

    if cjk_count == 0 and latin_count == 0:
        return "Chinese"

    ratio = cjk_count / max(cjk_count + latin_count, 1)
    if cjk_count >= 40 and ratio >= 0.12:
        return "Chinese"

    return "English"


def discover_input_files(data_dir: Path, pattern: str, file_limit: int) -> List[Path]:
    files = sorted(data_dir.glob(pattern))
    if file_limit and file_limit > 0:
        files = files[:file_limit]
    return files


def load_cases(json_path: Path) -> Tuple[Optional[str], Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and isinstance(raw.get("cases"), dict):
        language = raw.get("language")
        cases = raw.get("cases", {})
        return normalize_language_tag(language), cases

    if isinstance(raw, dict):
        return None, raw

    raise ValueError(f"无法解析输入 JSON 结构: {json_path}")


def _resolve_md_candidate(candidate: str, workspace_root: Path) -> Optional[Path]:
    p = Path(candidate)
    checks: List[Path] = []
    search_roots = [workspace_root, workspace_root.parent]

    if p.is_absolute():
        checks.append(p)
    else:
        for root in search_roots:
            checks.append((root / p).resolve())
            checks.append((root / p.name).resolve())

            # data2 中记录的路径可能是 470_md/Chinese/*.md；当前仓库实际为 470_md/*.md。
            if "470_md" in p.parts:
                checks.append((root / "470_md" / p.name).resolve())

    seen: set[str] = set()
    for c in checks:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_file():
            return c

    return None


def _expand_md_candidates(raw_value: str, case_id: Optional[str]) -> List[str]:
    raw = (raw_value or "").strip()
    if not raw:
        return []

    p = Path(raw)
    stem = p.stem if p.stem else raw

    candidates: List[str] = [raw]

    # 兼容输入里提供 PDF 路径（English_file_name/Chinese_file_name），自动映射到 md 候选路径。
    if p.suffix.lower() == ".pdf":
        candidates.extend(
            [
                str(p.with_suffix(".md")),
                f"{stem}.md",
                f"470_md/{stem}.md",
                f"470_md/English/{stem}.md",
                f"470_md/Chinese/{stem}.md",
            ]
        )

    if p.suffix.lower() != ".md":
        candidates.extend(
            [
                f"{stem}.md",
                f"470_md/{stem}.md",
                f"470_md/English/{stem}.md",
                f"470_md/Chinese/{stem}.md",
            ]
        )

    if case_id:
        candidates.extend(
            [
                f"{case_id}.md",
                f"470_md/{case_id}.md",
                f"470_md/English/{case_id}.md",
                f"470_md/Chinese/{case_id}.md",
            ]
        )

    deduped: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def resolve_md_path(
    case_info: Dict[str, Any],
    workspace_root: Path,
    language: Optional[str],
) -> Tuple[Path, str]:
    if language == "English":
        preferred_fields = [
            "english_md_name",
            "English_md_name",
            "english_file_name",
            "English_file_name",
            "chinese_md_name",
            "Chinese_md_name",
            "chinese_file_name",
            "Chinese_file_name",
        ]
    else:
        preferred_fields = [
            "chinese_md_name",
            "Chinese_md_name",
            "chinese_file_name",
            "Chinese_file_name",
            "english_md_name",
            "English_md_name",
            "english_file_name",
            "English_file_name",
        ]

    missing_fields: List[str] = []
    case_id = case_info.get("case_id") if isinstance(case_info.get("case_id"), str) else None

    for field in preferred_fields:
        val = case_info.get(field)
        if not isinstance(val, str) or not val.strip():
            missing_fields.append(field)
            continue

        for candidate in _expand_md_candidates(val.strip(), case_id=case_id):
            resolved = _resolve_md_candidate(candidate, workspace_root)
            if not resolved:
                continue

            try:
                if resolved.stat().st_size <= 0:
                    continue
            except OSError:
                continue

            return resolved, field

    # 最后兜底：只用 case_id 推断 md 文件。
    if case_id:
        for candidate in _expand_md_candidates(case_id, case_id=case_id):
            resolved = _resolve_md_candidate(candidate, workspace_root)
            if resolved:
                try:
                    if resolved.stat().st_size > 0:
                        return resolved, "case_id_fallback"
                except OSError:
                    pass

    raise FileNotFoundError(
        "未找到可用 md 文件，优先字段: "
        f"{preferred_fields}；空字段: {missing_fields}"
    )


def read_md_text(md_path: Path) -> str:
    try:
        return md_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return md_path.read_text(encoding="utf-8", errors="replace")


def build_messages(case_id: str, language: str, case_text: str) -> List[Dict[str, str]]:
    if language == "Chinese":
        user_prompt = (
            f"病例ID: {case_id}\n"
            "请严格按系统指令完成 TNM+CDSS 两阶段流程，并仅输出中文 JSON。\n"
            "JSON 结构强约束（必须全部满足）：\n"
            "1) 顶层必须且只能是一个 JSON 对象。\n"
            "2) 所有键名和字符串值必须使用双引号。\n"
            "3) 字符串中的换行必须写成 \\n，不要写成原始换行。\n"
            "4) 禁止任何尾逗号。\n"
            "5) 禁止输出 markdown、代码块、解释性文字、前后缀文本。\n"
            "6) 必须完整包含要求的所有字段。\n\n"
            f"{case_text}"
        )
        system_prompt = TNM_CDSS_SYSTEM_PROMPT_ZH
    else:
        user_prompt = (
            f"Case ID: {case_id}\n"
            "Please strictly follow the system instructions to complete the two-phase TNM+CDSS process and output only an English JSON.\n"
            "Strong Constraints for JSON Structure (All must be met):\n"
            "1) The top level must and can only be a single JSON object.\n"
            "2) All key names and string values must use double quotes.\n"
            "3) Newlines within strings must be written as \\n, do not use raw newlines.\n"
            "4) No trailing commas allowed.\n"
            "5) No output of markdown, code blocks, explanatory text, prefixes, or suffixes.\n"
            "6) Must fully include all required fields.\n\n"
            f"{case_text}"
        )
        system_prompt = TNM_CDSS_SYSTEM_PROMPT_EN

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_repair_hint(language: str) -> str:
    if language == "Chinese":
        return (
            "你上一条输出不是可解析的 JSON 对象。"
            "请严格只输出一个完整 JSON 对象，不要输出数字、数组、markdown、解释文本。"
            "必须包含字段: tnm_text, stage_result, cdss_result。"
        )

    return (
        "Your previous output was not a parseable JSON object. "
        "Return exactly one complete JSON object only; do not output numbers, arrays, markdown, or explanations. "
        "Required fields: tnm_text, stage_result, cdss_result."
    )


def call_chat_api(provider: ProviderConfig, messages: List[Dict[str, str]]) -> str:
    url = provider.base_url.rstrip("/") + "/chat/completions"

    payload = {
        "model": provider.model,
        "messages": messages,
        "temperature": provider.temperature,
        "max_tokens": provider.max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, provider.max_retries + 1):
        req = urlrequest.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=provider.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
            content = _extract_content_from_response(obj)
            if not content.strip():
                raise ValueError("模型返回内容为空")
            return content
        except (urlerror.HTTPError, urlerror.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt < provider.max_retries:
                time.sleep(provider.retry_backoff_seconds * attempt)

    raise RuntimeError(f"API 调用失败: {last_err}")


def _extract_content_from_response(resp_obj: Dict[str, Any]) -> str:
    choices = resp_obj.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts)

    return ""


def _strip_code_fence(text: str) -> str:
    txt = (text or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s*```$", "", txt)
    return txt.strip()


def _escape_newlines_inside_json_strings(text: str) -> str:
    out: List[str] = []
    in_string = False
    escaped = False

    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue

            if ch == "\\":
                out.append(ch)
                escaped = True
                continue

            if ch == '"':
                out.append(ch)
                in_string = False
                continue

            if ch == "\n":
                out.append("\\n")
                continue

            if ch == "\r":
                out.append("\\r")
                continue

            out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_string = True

    return "".join(out)


def _repair_common_broken_json(text: str) -> str:
    fixed = (text or "").strip()
    if not fixed:
        return fixed

    fixed = fixed.replace("\ufeff", "")
    fixed = fixed.replace("“", '"').replace("”", '"')
    fixed = fixed.replace("‘", "'").replace("’", "'")
    fixed = fixed.replace("\u00a0", " ")
    fixed = _escape_newlines_inside_json_strings(fixed)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    return fixed


def _close_unterminated_json(text: str) -> str:
    """Best-effort repair for truncated JSON fragments from model outputs."""
    s = (text or "").strip()
    if not s:
        return s

    out: List[str] = []
    stack: List[str] = []
    in_string = False
    escaped = False

    for ch in s:
        out.append(ch)

        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in {"}", "]"} and stack and stack[-1] == ch:
            stack.pop()

    # If output was truncated while inside a string, close the quote first.
    if in_string:
        out.append('"')

    while stack:
        out.append(stack.pop())

    return "".join(out)


def _coerce_json_object(value: Any) -> Dict[str, Any]:
    """Normalize parsed JSON into a dict so downstream logic can safely access keys."""
    if isinstance(value, dict):
        return value

    if isinstance(value, list):
        # Common fallback: model wraps the object in a single-element list.
        if len(value) == 1 and isinstance(value[0], dict):
            return value[0]

        # Another possible shape: list of key-value pairs.
        if all(
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], str)
            for item in value
        ):
            return {str(k): v for k, v in value}

        raise ValueError("模型输出顶层是 JSON 数组，且无法转换为对象")

    if isinstance(value, str):
        inner = value.strip()
        if inner and (inner.startswith("{") or inner.startswith("[")):
            try:
                return _coerce_json_object(json.loads(inner))
            except json.JSONDecodeError:
                pass
        raise ValueError("模型输出顶层是字符串，且不是可解析的 JSON 对象")

    raise ValueError(f"模型输出顶层类型无效: {type(value).__name__}")


def _iter_json_candidates(raw_text: str) -> List[str]:
    seeds = [raw_text, _strip_code_fence(raw_text)]
    candidates: List[str] = []
    seen: set[str] = set()

    for seed in seeds:
        s = (seed or "").strip()
        if not s:
            continue

        for item in (s, _repair_common_broken_json(s)):
            t = item.strip()
            if t and t not in seen:
                seen.add(t)
                candidates.append(t)

            closed = _close_unterminated_json(t)
            if closed and closed not in seen:
                seen.add(closed)
                candidates.append(closed)

            start = t.find("{")
            end = t.rfind("}")
            if start >= 0 and end > start:
                snippet = t[start : end + 1].strip()
                for item2 in (
                    snippet,
                    _repair_common_broken_json(snippet),
                    _close_unterminated_json(snippet),
                ):
                    u = item2.strip()
                    if u and u not in seen:
                        seen.add(u)
                        candidates.append(u)

    return candidates


def parse_json_from_model_output(raw_text: str) -> Dict[str, Any]:
    txt = (raw_text or "").strip()
    if not txt:
        raise ValueError("模型输出为空")

    last_err: Optional[Exception] = None
    for candidate in _iter_json_candidates(txt):
        try:
            parsed = json.loads(candidate)
            return _coerce_json_object(parsed)
        except (json.JSONDecodeError, ValueError) as exc:
            last_err = exc

    raise ValueError(f"无法从模型输出解析 JSON: {last_err}")


def normalize_tnm_result(raw_obj: Any) -> Dict[str, Any]:
    if not isinstance(raw_obj, dict):
        raw_obj = _coerce_json_object(raw_obj)

    if "tnm_result" in raw_obj and isinstance(raw_obj["tnm_result"], dict):
        raw_obj = raw_obj["tnm_result"]

    normalized = {
        "tnm_text": str(raw_obj.get("tnm_text", "")).strip(),
        "stage_result": str(raw_obj.get("stage_result", "")).strip(),
    }
    return normalized


def extract_cdss_result(raw_obj: Any) -> str:
    if not isinstance(raw_obj, dict):
        try:
            raw_obj = _coerce_json_object(raw_obj)
        except ValueError:
            return ""

    nested_tnm = raw_obj.get("tnm_result")
    if not isinstance(nested_tnm, dict):
        nested_tnm = {}

    for value in (
        raw_obj.get("cdss_results"),
        raw_obj.get("cdss_result"),
        nested_tnm.get("cdss_results"),
        nested_tnm.get("cdss_result"),
    ):
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        elif value is not None:
            text = str(value).strip()
            if text:
                return text

    return ""


def safe_model_token(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_") or "model"


def take_items(d: Dict[str, Any], limit: int) -> Iterable[Tuple[str, Any]]:
    items = list(d.items())
    if limit and limit > 0:
        items = items[:limit]
    return items


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT
    workspace_root = project_root

    config_path = resolve_existing_path(args.config, workspace_root=workspace_root, expect_dir=False)
    data_dir = resolve_existing_path(args.data_dir, workspace_root=workspace_root, expect_dir=True)

    provider = load_provider(config_path, args.model)
    json_files = discover_input_files(data_dir, args.input_pattern, args.max_files)

    if not json_files:
        print(f"未找到输入文件: {data_dir / args.input_pattern}")
        return 1

    model_token = safe_model_token(args.model)
    if isinstance(args.output_dir, str) and args.output_dir.strip():
        custom_prefix = Path(args.output_dir.strip())
        if custom_prefix.is_absolute():
            out_prefix_dir = custom_prefix
        else:
            # 相对路径统一按项目根目录解析，便于跨脚本目录使用。
            out_prefix_dir = (project_root / custom_prefix).resolve()
    else:
        out_prefix_dir = project_root / "results" / "outputs" / "llm_outputs" / "llm_text_output"

    # 无论 output-dir 如何设置，都强制在末尾追加模型目录。
    out_dir = out_prefix_dir / model_token
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始执行，模型: {args.model}")
    print(f"输入文件数量: {len(json_files)}")

    for file_index, json_file in enumerate(json_files, start=1):
        inferred_language, seed = infer_language_seed(json_file)
        explicit_language, cases = load_cases(json_file)
        file_language = explicit_language or inferred_language
        if file_language == "Unknown":
            file_language = ""

        output: Dict[str, Any] = {
            "source_input": str(json_file).replace("\\", "/"),
            "language": file_language or "Auto",
            "seed": seed,
            "model": args.model,
            "cases": {},
        }

        selected_cases = list(take_items(cases, args.max_cases))
        print(f"[{file_index}/{len(json_files)}] 处理 {json_file.name}，cases: {len(selected_cases)}")

        observed_languages: set[str] = set()

        for case_idx, (case_key, case_info) in enumerate(selected_cases, start=1):
            cid = str(case_info.get("case_id") or case_key)
            case_result: Dict[str, Any] = {"case_id": cid}
            raw_output = ""
            try:
                if not isinstance(case_info, dict):
                    raise ValueError("case 数据不是对象")

                case_lang = file_language or normalize_language_tag(case_info.get("language"))
                md_path, md_source = resolve_md_path(case_info, workspace_root=workspace_root, language=case_lang)
                case_result["md_path"] = str(md_path).replace("\\", "/")
                case_result["md_source"] = md_source

                text = read_md_text(md_path)
                if not text.strip():
                    raise RuntimeError("MD 文本为空，无法提供给模型")

                if not case_lang:
                    case_lang = detect_language_from_text(text)
                if case_lang not in {"Chinese", "English"}:
                    case_lang = "Chinese"

                observed_languages.add(case_lang)
                case_result["prompt_language"] = case_lang

                trimmed_text = text[: args.max_input_chars]
                max_json_retries = max(1, int(args.json_retries))
                model_json: Optional[Dict[str, Any]] = None
                parse_err: Optional[Exception] = None
                generation_prompt_language = case_lang
                prompt_candidates: List[Tuple[str, List[Dict[str, str]]]] = [
                    (case_lang, build_messages(case_id=cid, language=case_lang, case_text=trimmed_text))
                ]

                if case_lang == "English":
                    prompt_candidates.append(
                        ("Chinese", build_messages(case_id=cid, language="Chinese", case_text=trimmed_text))
                    )

                for candidate_lang, base_messages in prompt_candidates:
                    retry_messages = base_messages

                    for json_try in range(1, max_json_retries + 1):
                        raw_output = call_chat_api(provider, retry_messages)
                        try:
                            model_json = parse_json_from_model_output(raw_output)
                            parse_err = None
                            generation_prompt_language = candidate_lang
                            break
                        except ValueError as exc:
                            parse_err = exc
                            if json_try < max_json_retries:
                                retry_messages = retry_messages + [
                                    {"role": "assistant", "content": raw_output[:4000]},
                                    {"role": "user", "content": build_repair_hint(candidate_lang)},
                                ]
                                time.sleep(1)

                    if model_json is not None:
                        break

                if model_json is None:
                    raise ValueError(f"JSON 解析失败（重试 {max_json_retries} 次）: {parse_err}")

                if generation_prompt_language != case_lang:
                    case_result["generation_prompt_language"] = generation_prompt_language

                case_result["tnm_result"] = normalize_tnm_result(model_json)
                cdss_text = extract_cdss_result(model_json)
                case_result["cdss_result"] = cdss_text
                print(f"  - ({case_idx}/{len(selected_cases)}) {cid}: 成功 [{case_lang}]")
            except Exception as exc:
                case_result["error"] = str(exc)
                if raw_output.strip():
                    case_result["raw_model_output"] = raw_output
                print(f"  - ({case_idx}/{len(selected_cases)}) {cid}: 失败 -> {exc}")

            output["cases"][cid] = case_result

        if not file_language:
            if len(observed_languages) == 1:
                output["language"] = next(iter(observed_languages))
            elif len(observed_languages) > 1:
                output["language"] = "Mixed"

        out_lang = output["language"] if output["language"] else "Auto"
        out_file = out_dir / f"{model_token}_output_{out_lang}_seed{seed}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"已保存: {out_file}")

    print("全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
