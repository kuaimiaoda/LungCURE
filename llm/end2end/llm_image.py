from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None


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
    max_tokens: int = 10240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TNM + CDSS 批量执行脚本（多模态PDF输入）")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "api_config.yaml"), help="API 配置 YAML 路径")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data_image"), help="输入 JSON 所在目录")
    parser.add_argument("--model", default="gpt-5.2", help="模型名")
    parser.add_argument("--max-files", type=int, default=0, help="执行多少个输入 JSON 文件，0 代表全部")
    parser.add_argument("--max-cases", type=int, default=0, help="每个 JSON 执行多少个 case，0 代表全部")
    parser.add_argument("--json-retries", type=int, default=2, help="单个 case 的最大 JSON 重试次数")
    parser.add_argument("--input-pattern", default="benchmark_*.json", help="输入 JSON 文件匹配模式")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "results_end2end" / "outputs" / "llm_outputs" / "llm_image_output"),
        help="输出目录前缀（最终总是自动追加 /<model>/文件名；默认: 项目根目录/results/outputs/llm_outputs/llm_image_output）",
    )
    parser.add_argument("--max-pages", type=int, default=8, help="每个 PDF 最多发送页数")
    parser.add_argument("--dpi", type=int, default=170, help="PDF 渲染 DPI")
    parser.add_argument("--max-image-size", type=int, default=1400, help="单页图片最大边长")
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
        max_tokens=int(provider_raw.get("max_tokens", 10240)),
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
        language = "Chinese" if lang_match.group(1).lower() == "chinese" else "English"

    seed_match = re.search(r"seed(\d+)", name, flags=re.IGNORECASE)
    if seed_match:
        seed = int(seed_match.group(1))

    return language, seed


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


def safe_model_token(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_") or "model"


def take_items(d: Dict[str, Any], limit: int) -> Iterable[Tuple[str, Any]]:
    items = list(d.items())
    if limit and limit > 0:
        items = items[:limit]
    return items


def _resolve_pdf_candidate(candidate: str, workspace_root: Path) -> Optional[Path]:
    p = Path(candidate)
    checks: List[Path] = []

    if p.is_absolute():
        checks.append(p)
    else:
        checks.append((workspace_root / p).resolve())
        checks.append((workspace_root.parent / p).resolve())
        checks.append((workspace_root / p.name).resolve())
        checks.append((workspace_root.parent / p.name).resolve())

    seen: set[str] = set()
    for c in checks:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_file():
            return c
    return None


def resolve_pdf_path(case_info: Dict[str, Any], workspace_root: Path, language: Optional[str]) -> Tuple[Path, str]:
    if language == "English":
        preferred_fields = ["English_file_name", "english_file_name", "pdf_path", "file_name", "Chinese_file_name", "chinese_file_name"]
    else:
        preferred_fields = ["Chinese_file_name", "chinese_file_name", "pdf_path", "file_name", "English_file_name", "english_file_name"]

    case_id = str(case_info.get("case_id", "")).strip()

    for field in preferred_fields:
        val = case_info.get(field)
        if not isinstance(val, str) or not val.strip():
            continue
        resolved = _resolve_pdf_candidate(val.strip(), workspace_root)
        if resolved is not None:
            return resolved, field

    if case_id:
        for candidate in [
            f"470/Chinese/{case_id}.pdf",
            f"470/English/{case_id}.pdf",
            f"{case_id}.pdf",
        ]:
            resolved = _resolve_pdf_candidate(candidate, workspace_root)
            if resolved is not None:
                return resolved, "case_id_fallback"

    raise FileNotFoundError("未找到可用 PDF 文件（检查 JSON 中的 *file_name/pdf_path 字段）")


def _pick_images(images: List[Any], max_pages: int) -> List[Any]:
    if max_pages <= 0 or len(images) <= max_pages:
        return images
    head = max_pages // 2
    tail = max_pages - head
    return images[:head] + images[-tail:]


def _encode_image_to_jpeg_b64(image: Any, max_image_size: int) -> str:
    if max(image.size) > max_image_size:
        image.thumbnail((max_image_size, max_image_size))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pdf_to_data_urls(pdf_path: Path, max_pages: int, dpi: int, max_image_size: int) -> List[str]:
    if convert_from_path is None:
        raise RuntimeError("未安装 pdf2image，无法进行 PDF 多模态读取。请先安装: pip install pdf2image")

    images = convert_from_path(str(pdf_path), dpi=dpi)
    images = _pick_images(images, max_pages=max_pages)
    if not images:
        raise RuntimeError("PDF 渲染后没有可用页面")

    urls: List[str] = []
    for img in images:
        b64 = _encode_image_to_jpeg_b64(img, max_image_size=max_image_size)
        urls.append(f"data:image/jpeg;base64,{b64}")
    return urls


def build_messages(case_id: str, language: str, image_urls: List[str], retry_hint: str = "") -> List[Dict[str, Any]]:
    if language == "Chinese":
        text_prompt = (
            f"病例ID: {case_id}。请阅读以下PDF页面图像，严格按系统指令完成TNM+CDSS两阶段流程，并仅输出中文JSON。"
            "JSON 结构强约束（必须全部满足）：\n"
            "1) 顶层必须且只能是一个 JSON 对象。\n"
            "2) 所有键名和字符串值必须使用双引号。\n"
            "3) 字符串中的换行必须写成 \\n，不要写成原始换行。\n"
            "4) 禁止任何尾逗号。\n"
            "5) 禁止输出 markdown、代码块、解释性文字、前后缀文本。\n"
            "6) 必须完整包含要求的所有字段。\n\n"
        )
        if retry_hint:
            text_prompt += f"\n\n上一轮问题：{retry_hint}"
        system_prompt = TNM_CDSS_SYSTEM_PROMPT_ZH
    else:
        text_prompt = (
            f"Case ID: {case_id}. Please read the following PDF page images and strictly follow the two-phase TNM+CDSS workflow in system instructions."
            "Strong Constraints for JSON Structure (All must be met):\n"
            "1) The top level must and can only be a single JSON object.\n"
            "2) All key names and string values must use double quotes.\n"
            "3) Newlines within strings must be written as \\n, do not use raw newlines.\n"
            "4) No trailing commas allowed.\n"
            "5) No output of markdown, code blocks, explanatory text, prefixes, or suffixes.\n"
            "6) Must fully include all required fields.\n\n"
        )
        if retry_hint:
            text_prompt += f"\n\nPrevious issue: {retry_hint}"
        system_prompt = TNM_CDSS_SYSTEM_PROMPT_EN

    content: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


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


def _coerce_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    raise ValueError(f"模型输出顶层类型无效: {type(value).__name__}")


def parse_json_from_model_output(raw_text: str) -> Dict[str, Any]:
    txt = (raw_text or "").strip()
    if not txt:
        raise ValueError("模型输出为空")

    txt = txt.replace("\ufeff", "")
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s*```$", "", txt)

    start = txt.find("{")
    end = txt.rfind("}")
    if start >= 0 and end > start:
        txt = txt[start : end + 1]

    parsed = json.loads(txt)
    return _coerce_json_object(parsed)


def normalize_tnm_result(raw_obj: Any) -> Dict[str, Any]:
    if not isinstance(raw_obj, dict):
        raw_obj = _coerce_json_object(raw_obj)

    return {
        "tnm_text": str(raw_obj.get("tnm_text", "")).strip(),
        "stage_result": str(raw_obj.get("stage_result", "")).strip(),
    }


def extract_cdss_result(raw_obj: Any) -> str:
    if not isinstance(raw_obj, dict):
        return ""

    for value in (raw_obj.get("cdss_result"), raw_obj.get("cdss_results")):
        if isinstance(value, str) and value.strip():
            return value.strip()

    nested = raw_obj.get("tnm_result")
    if isinstance(nested, dict):
        for value in (nested.get("cdss_result"), nested.get("cdss_results")):
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _is_multimodal_messages(messages: List[Dict[str, Any]]) -> bool:
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            return True
    return False


def _should_disable_response_format_for_provider(provider: ProviderConfig, messages: List[Dict[str, Any]]) -> bool:
    """
    SiliconFlow 上部分模型在多模态 content(list) 下不支持 response_format：
    - Qwen/Qwen3.5-397B-A17B: 会返回 messages illegal
    - zai-org/GLM-4.6V: 会返回 Json mode is not supported
    仅对这些模型做兼容分支，避免影响其他模型。
    """
    base = str(provider.base_url).strip().lower()
    model = str(provider.model).strip()
    no_json_mode_models = {
        "Qwen/Qwen3.5-397B-A17B",
        "zai-org/GLM-4.6V",
    }
    return (
        base.startswith("https://api.siliconflow.cn")
        and model in no_json_mode_models
        and _is_multimodal_messages(messages)
    )


def _http_error_with_body(exc: urlerror.HTTPError) -> Exception:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""

    if body:
        return RuntimeError(f"HTTP Error {exc.code}: {exc.reason}; body={body[:1200]}")
    return RuntimeError(f"HTTP Error {exc.code}: {exc.reason}")


def call_chat_api(provider: ProviderConfig, messages: List[Dict[str, Any]]) -> str:
    url = provider.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": provider.model,
        "messages": messages,
        "temperature": provider.temperature,
        "max_tokens": provider.max_tokens,
    }
    if not _should_disable_response_format_for_provider(provider, messages):
        payload["response_format"] = {"type": "json_object"}
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
        except urlerror.HTTPError as exc:
            last_err = _http_error_with_body(exc)
            if attempt < provider.max_retries:
                time.sleep(provider.retry_backoff_seconds * attempt)
        except (urlerror.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            if attempt < provider.max_retries:
                time.sleep(provider.retry_backoff_seconds * attempt)

    raise RuntimeError(f"API 调用失败: {last_err}")


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
            out_prefix_dir = (project_root / custom_prefix).resolve()
    else:
        out_prefix_dir = project_root / "results" / "outputs" / "llm_outputs" / "llm_image_output"

    out_dir = out_prefix_dir / model_token
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始执行（PDF多模态），模型: {args.model}")
    print(f"输入文件数量: {len(json_files)}")

    for file_index, json_file in enumerate(json_files, start=1):
        inferred_language, seed = infer_language_seed(json_file)
        explicit_language, cases = load_cases(json_file)
        file_language = explicit_language or inferred_language
        if file_language == "Unknown":
            file_language = "Chinese"

        output: Dict[str, Any] = {
            "source_input": str(json_file).replace("\\", "/"),
            "language": file_language,
            "seed": seed,
            "model": args.model,
            "cases": {},
        }

        selected_cases = list(take_items(cases, args.max_cases))
        print(f"[{file_index}/{len(json_files)}] 处理 {json_file.name}，cases: {len(selected_cases)}")

        for case_idx, (case_key, case_info) in enumerate(selected_cases, start=1):
            cid = str(case_info.get("case_id") or case_key)
            case_result: Dict[str, Any] = {"case_id": cid}
            raw_output = ""

            try:
                if not isinstance(case_info, dict):
                    raise ValueError("case 数据不是对象")

                pdf_path, pdf_source = resolve_pdf_path(case_info, workspace_root=workspace_root, language=file_language)
                case_result["pdf_path"] = str(pdf_path).replace("\\", "/")
                case_result["pdf_source"] = pdf_source

                image_urls = pdf_to_data_urls(
                    pdf_path=pdf_path,
                    max_pages=max(1, int(args.max_pages)),
                    dpi=max(72, int(args.dpi)),
                    max_image_size=max(512, int(args.max_image_size)),
                )
                case_result["image_pages"] = len(image_urls)

                model_json: Optional[Dict[str, Any]] = None
                parse_err: Optional[Exception] = None

                for json_try in range(1, max(1, int(args.json_retries)) + 1):
                    retry_hint = ""
                    if parse_err is not None:
                        retry_hint = f"上一轮JSON解析失败: {parse_err}。请仅输出合法JSON对象。"

                    messages = build_messages(
                        case_id=cid,
                        language=file_language,
                        image_urls=image_urls,
                        retry_hint=retry_hint,
                    )
                    raw_output = call_chat_api(provider, messages)

                    try:
                        model_json = parse_json_from_model_output(raw_output)
                        parse_err = None
                        break
                    except Exception as exc:
                        parse_err = exc
                        if json_try < max(1, int(args.json_retries)):
                            time.sleep(1)

                if model_json is None:
                    raise ValueError(f"JSON 解析失败（重试后仍失败）: {parse_err}")

                case_result["tnm_result"] = normalize_tnm_result(model_json)
                case_result["cdss_result"] = extract_cdss_result(model_json)
                print(f"  - ({case_idx}/{len(selected_cases)}) {cid}: 成功 [{file_language}]")
            except Exception as exc:
                case_result["error"] = str(exc)
                if raw_output.strip():
                    case_result["raw_model_output"] = raw_output
                print(f"  - ({case_idx}/{len(selected_cases)}) {cid}: 失败 -> {exc}")

            output["cases"][cid] = case_result

        out_lang = output["language"] if output["language"] else "Auto"
        out_file = out_dir / f"{model_token}_output_{out_lang}_seed{seed}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"已保存: {out_file}")

    print("全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
