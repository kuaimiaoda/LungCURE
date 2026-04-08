from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CODE_ROOT = PROJECT_ROOT / "code"
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", str(PROJECT_ROOT.parent))).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from code.common.model_api import (
        UnifiedOpenAIClient,
        endpoint_for_model as common_endpoint_for_model,
        extract_message_content as common_extract_message_content,
        extract_reasoning_content as common_extract_reasoning_content,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    if getattr(exc, "name", "") == "openai":
        raise SystemExit(
            "依赖缺失: openai。请先安装后再运行 code/inference/direct_llm_batch_infer.py，例如: pip install openai"
        ) from exc
    raise

from code.common.case_loader import (
    iter_case_files as common_iter_case_files,
    load_case_payload as common_load_case_payload,
    resolve_benchmark_dir as common_resolve_benchmark_dir,
)

# 其他模型默认走 apiyi（kimi 这组）
DEFAULT_BASE_URL = "https://api.apiyi.com/v1"
DEFAULT_API_KEY = "sk-Te6gIPgo3nkjhP3w70B515C4F59b44B385058a6aDd514cE4"

# GLM-4.6V 走 z.ai
GLM46V_BASE_URL = "https://api.z.ai/api/paas/v4/"
GLM46V_API_KEY = "d894e8a6c3f94e81bf0eff2548f6d17e.iR7nm5vuYTRsg3nX"

LLM_UNIFIED_CONFIG: dict[str, Any] = {
    "base_url": DEFAULT_BASE_URL,
    "api_key": DEFAULT_API_KEY,
    "model": "kimi-k2.5",
    "timeout": 480,
    "completion_params": {
        "max_tokens": 8192,
        "temperature": None,
        "top_p": None,
        "presence_penalty": None,
        "frequency_penalty": None,
    },
    "system_prompt": (
        
		r"""角色定义你是一个专业的肿瘤科“TNM分期与临床决策支持”AI助手。你的核心任务是严格按照流程，处理输入的患者病历、影像报告及病理报告、tnm分期等材料，，随后基于分期结果和患者特征生成非小细胞肺癌（NSCLC）的处理意见。核心工作流指令请严格按照以下流程、顺序执行任务。NSCLC治疗方案决策工作流-参考综合分期结果，执行以下步骤：步骤1：关键临床特征结构化提取结合原始病历和TNM结果，提取出决策所需的12个关键字段：1) 病理类型（鳞癌/非鳞癌）2) 体力评分（PS评分：0-1/2/3-4）3) 驱动基因状态（具体突变类型或阴性）4) PD-L1表达情况5) 高危因素（针对早期患者）6) 治疗阶段（根治性术前/根治性术后）7) 既往治疗方案/用药史8) 免疫治疗禁忌症9) 转移类型（寡转移/广泛转移/无）10) 治疗线别（一线治疗/二线及后线）步骤2：生成规范化“处理意见”输出最终的临床决策。必须严格遵循以下医学病历书写格式，禁止输出多余的教学分析或免责声明：病情评估：[用1-2句话概括患者当前阶段、关键分层信息、治疗线别。语言精炼。]处理：1. [总治疗路径]2. [具体推荐方案，写明具体药物或组合]3. [同层级可选的替代方案，分条列出]4. [局部治疗/放疗/会诊安排（如有）]随访：1. [复查/评估安排]2. [疗效监测安排]3. [长期复查安排]执行要求- 严禁脑补缺失的患者信息。对于无法确定的条件，依据“就低原则”或默认阴性处理（例如未提及基因突变视为阴性）。- 使用客观、严肃的临床书面语。输出要求你必须只输出一个 JSON 对象，不要输出 markdown，不要输出额外解释。JSON 必须包含以下字段：{"stage_result": "早期或晚期","cdss_result": "治疗意见全文"}"""
 
    ),
    "system_prompt_en": (
        
        r""" Role definition: You are a professional oncology AI assistant for “TNM staging and clinical decision support.” Your core task is to strictly follow the workflow to process the input patient records, imaging reports, pathology reports, TNM staging, and related materials, and then generate management recommendations for non-small cell lung cancer (NSCLC) based on the staging result and patient characteristics.Core workflow instructions: Please strictly follow the workflow and sequence below.NSCLC treatment decision workflow:Refer to the overall staging result and perform the following steps:Step 1: Structured extraction of key clinical featuresUsing the original medical record together with the TNM result, extract the 10 key fields required for decision-making:1. Pathology type (squamous / non-squamous)2. Performance status (PS score: 0–1 / 2 / 3–4)3. Driver gene status (specific mutation type or negative)4. PD-L1 expression status5. High-risk factors (for early-stage patients)6. Treatment stage (preoperative curative / postoperative curative)7. Prior treatment regimen / medication history8. Contraindications to immunotherapy9. Metastatic pattern (oligometastatic / widespread metastatic / none)10. Line of therapy (first-line / second-line and beyond)Step 2: Generate a standardized “Management Recommendation”Output the final clinical decision. You must strictly follow the medical record writing format below and must not output extra teaching analysis or disclaimers:Disease assessment:[Use 1–2 sentences to summarize the patient’s current stage, key stratification information, and treatment line. Language must be concise.]Management:1. [Overall treatment pathway]2. [Specific recommended regimen, including specific drugs or combinations]3. [Alternative options at the same level, listed separately]4. [Local treatment / radiotherapy / multidisciplinary consultation arrangements, if applicable]Follow-up:1. [Re-examination / assessment plan]2. [Efficacy monitoring plan]3. [Long-term follow-up plan]Execution requirements:* Do not fabricate missing patient information. For conditions that cannot be determined, follow the “lower-bound principle” or treat them as negative by default (for example, if no driver mutation is mentioned, treat it as negative).* Use objective, serious clinical written language.Output requirements:You must output only a single JSON object. Do not output markdown and do not output any extra explanation. The JSON must contain the following fields:{"stage_result": "early or advanced", "cdss_result": "full treatment recommendation text"} """
        
    ),
}

# Optional alternate prompt set for direct LLM decision mode.
# If not explicitly provided, default to the existing prompt templates.
LLM_UNIFIED_CONFIG.setdefault("llmsystem_prompt", LLM_UNIFIED_CONFIG.get("system_prompt", ""))
LLM_UNIFIED_CONFIG.setdefault("llmsystem_prompt_en", LLM_UNIFIED_CONFIG.get("system_prompt_en", ""))

# 不传 --model 时，按此列表顺序自动轮换执行。
MODEL_NAMES = [
    "GLM-4.6V",
    "qwen3.5-397b-a17b",
    "llama-4-maverick",
    "gpt-5.2",
    "claude-sonnet-4-6",
    "grok-4",
    "kimi-k2.5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="直接把病例信息传给 LLM 判断（无工作流），支持多进程并行和多模型自动轮换。"
    )
    parser.add_argument(
        "--benchmark-dir",
        default="benchmark_gt_Chinese_seed2024",
        help="病例目录（支持 *.md/*.txt/*.json/*.pdf）。",
    )
    parser.add_argument(
        "--case-file",
        default=None,
        help="单病例文件路径，设置后忽略 benchmark-dir。",
    )
    parser.add_argument("--limit", type=int, default=0, help="限制病例数，0 表示全部。")
    parser.add_argument("--random-sample", type=int, default=0, help="随机抽取 N 例，0 不启用。")
    parser.add_argument("--random-seed", type=int, default=2026, help="随机种子。")
    parser.add_argument(
        "--cases-per-process",
        type=int,
        default=10,
        help="每进程处理病例数。",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=0,
        help="最大进程数，0 表示不限制。",
    )

    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", str(LLM_UNIFIED_CONFIG["model"])),
        help="模型名。显式传入时只跑该模型；不传则按 MODEL_NAMES 自动轮换。",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", str(LLM_UNIFIED_CONFIG["base_url"])),
        help="OpenAI 兼容 base_url。",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY", str(LLM_UNIFIED_CONFIG["api_key"])),
        help="API Key。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(LLM_UNIFIED_CONFIG["timeout"]),
        help="请求超时秒数。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(LLM_UNIFIED_CONFIG["completion_params"]["max_tokens"]),
        help="max_tokens。",
    )
    parser.add_argument("--temperature", type=float, default=None, help="temperature，不传则不下发。")
    parser.add_argument("--top-p", type=float, default=None, help="top_p，不传则不下发。")
    parser.add_argument("--presence-penalty", type=float, default=None, help="presence_penalty。")
    parser.add_argument("--frequency-penalty", type=float, default=None, help="frequency_penalty。")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="向模型请求思考模式（下发 thinking 参数）。",
    )
    parser.add_argument(
        "--thinking-type",
        default="enabled",
        help="thinking.type 值，默认 enabled。",
    )
    parser.add_argument(
        "--save-thinking",
        action="store_true",
        help="把模型 reasoning 内容保存到输出字段 think。",
    )

    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="System prompt file path.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("zh", "en"),
        default="zh",
        help="Prompt mode: zh=Chinese template, en=English mode.",
    )
    parser.add_argument(
        "--use-english-prompt",
        action="store_true",
        help="Compatibility switch. If set, prompt mode is forced to en.",
    )
    parser.add_argument(
        "--english-system-prompt-file",
        default=None,
        help="English mode system prompt file path (higher priority than --system-prompt-file).",
    )
    parser.add_argument(
        "--use-llm-system-prompt",
        action="store_true",
        help="Use llmsystem_prompt / llmsystem_prompt_en instead of system_prompt / system_prompt_en.",
    )
    parser.add_argument(
        "--output-json",        default="llm_results.json",
        help="输出文件名（默认写到 llm 文件夹内）。",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="兼容参数：当前默认即为追加写入，不会覆盖已有内容。",
    )
    parser.add_argument("--verbose", action="store_true", help="打印处理日志。")
    return parser.parse_args()


def is_flag_explicitly_set(argv: list[str], flag_name: str) -> bool:
    return any(x == flag_name or x.startswith(f"{flag_name}=") for x in argv)


def is_model_explicitly_set(argv: list[str]) -> bool:
    return is_flag_explicitly_set(argv, "--model")


def endpoint_for_model(model: str) -> tuple[str, str]:
    return common_endpoint_for_model(model, route_qwen_to_siliconflow=True, route_kimi_to_siliconflow=False)


def sanitize_model_name(model: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model))
    text = text.strip("._-")
    return text or "model"


def output_path_for_model(base_output: Path, model: str) -> Path:
    suffix = base_output.suffix or ".json"
    stem = base_output.stem
    safe_model = sanitize_model_name(model)
    return base_output.with_name(f"{stem}_{safe_model}{suffix}")


def failed_models_path(base_output: Path) -> Path:
    return base_output.with_name(f"{base_output.stem}_failed_models.txt")


def write_failed_models(path: Path, model_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [x for x in model_names if x]
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def resolve_output_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_absolute():
        return p
    norm = str(p).replace("\\", "/")
    if norm == "code":
        norm = "."
    elif norm.startswith("code/"):
        norm = norm[len("code/") :]
    return (WORKSPACE_ROOT / Path(norm)).resolve()

def resolve_benchmark_dir(path_arg: str) -> Path:
    return common_resolve_benchmark_dir(path_arg, root_dir=WORKSPACE_ROOT)


def iter_case_files(
    case_file: str | None,
    benchmark_dir: str,
    limit: int,
    random_sample: int,
    random_seed: int,
) -> list[Path]:
    # 统一支持目录内 md/txt/json/pdf；也支持直接传单文件。
    return common_iter_case_files(
        case_file=case_file,
        benchmark_dir=benchmark_dir,
        limit=limit,
        random_sample=random_sample,
        random_seed=random_seed,
        read_pdf=True,
        root_dir=WORKSPACE_ROOT,
        default_suffixes=("md", "txt", "json"),
    )


def load_case_payload(case_path: Path) -> dict[str, Any]:
    return common_load_case_payload(case_path)

def extract_message_content(content: Any) -> str:
    return common_extract_message_content(content)


def extract_reasoning_content(message: dict[str, Any]) -> str:
    return common_extract_reasoning_content(message)


def extract_category_from_final_answer(final_answer: str) -> str:
    text = str(final_answer or "").strip()
    if not text:
        return ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.lstrip("#").strip()
        if line:
            return line
    return ""


def normalize_prompt_mode(prompt_mode: str, use_english_prompt: bool = False) -> str:
    if use_english_prompt:
        return "en"
    mode = str(prompt_mode or "zh").strip().lower()
    return "en" if mode == "en" else "zh"


def build_messages(
    case: dict[str, Any],
    system_prompt: str,
    *,
    prompt_mode: str = "zh",
) -> list[dict[str, str]]:
    mode = normalize_prompt_mode(prompt_mode)
    match mode:
        case "en":
            user_text = str(case.get("query", ""))
        case _:
            user_text = (
                f"病例ID: {case.get('case_id', '')}\n"
                f"TNM_ret: {case.get('tnm_ret', '')}\n"
                f"病例内容:\n{case.get('query', '')}\n\n"
                "请给出你的判断与建议。"
            )

    messages: list[dict[str, str]] = []
    if str(system_prompt or "").strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return messages


def load_system_prompt(
    system_prompt_file: str | None,
    *,
    prompt_mode: str = "zh",
    english_system_prompt_file: str | None = None,
    use_llm_system_prompt: bool = False,
) -> str:
    mode = normalize_prompt_mode(prompt_mode)
    prompt_file = system_prompt_file
    if mode == "en" and english_system_prompt_file:
        prompt_file = english_system_prompt_file

    if prompt_file:
        p = Path(prompt_file).resolve()
        return p.read_text(encoding="utf-8")

    match mode:
        case "en":
            return str(LLM_UNIFIED_CONFIG.get("system_prompt_en", ""))
        case _:
            return str(LLM_UNIFIED_CONFIG["system_prompt"])


class OpenAICompatClient(UnifiedOpenAIClient):
    pass

def build_request(model: str, messages: list[dict[str, str]], llm_config: dict[str, Any]) -> dict[str, Any]:
    req: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    for key in ("max_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty"):
        value = llm_config.get(key)
        if value is not None:
            req[key] = value

    if llm_config.get("enable_thinking"):
        req["thinking"] = {"type": str(llm_config.get("thinking_type", "enabled"))}

    return req


def run_case_judge(
    client: OpenAICompatClient,
    case_path: Path,
    *,
    model: str,
    llm_config: dict[str, Any],
    system_prompt: str,
    prompt_mode: str = "zh",
) -> dict[str, Any]:
    case = load_case_payload(case_path)
    save_thinking = bool(llm_config.get("save_thinking"))
    try:
        messages = build_messages(case, system_prompt, prompt_mode=prompt_mode)
        req = build_request(model, messages, llm_config)

        try:
            resp = client.chat_completions(req)
        except Exception as exc:
            err_text = str(exc)
            unsupported_thinking = (
                "Unrecognized request argument supplied: thinking" in err_text
                or "unsupported" in err_text.lower() and "thinking" in err_text.lower()
                or "unknown" in err_text.lower() and "thinking" in err_text.lower()
            )
            if req.get("thinking") is not None and llm_config.get("thinking_fallback", True) and unsupported_thinking:
                req = dict(req)
                req.pop("thinking", None)
                resp = client.chat_completions(req)
            else:
                raise

        choices = resp.get("choices", [])
        if not choices:
            raise RuntimeError(f"响应中无 choices: {resp}")
        message = choices[0].get("message", {})
        final_answer = common_extract_message_content(message.get("content", "")).strip()
        think_text = common_extract_reasoning_content(message).strip()

        item: dict[str, Any] = {
            "case_file": case["case_file"],
            "case_id": case["case_id"],
            "tnm_ret": case["tnm_ret"],
            "关键临床特征是否获取": "否",
            "分类": extract_category_from_final_answer(final_answer),
            "final_answer": final_answer,
            "model": model,
            "error": "",
        }
        if save_thinking:
            item["think"] = think_text
        return item
    except Exception as exc:
        item = {
            "case_file": case["case_file"],
            "case_id": case["case_id"],
            "tnm_ret": case["tnm_ret"],
            "关键临床特征是否获取": "否",
            "分类": "",
            "final_answer": "",
            "model": model,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if save_thinking:
            item["think"] = ""
        return item


def load_output_array(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    raw = output_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def write_output_array(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    try:
        output_path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        fallback = output_path.with_name(f"{output_path.stem}_fallback{output_path.suffix}")
        fallback.write_text(payload, encoding="utf-8")
        print(f"[WARN] ??????????? fallback ??: {fallback} ({type(exc).__name__}: {exc})")


def append_result(output_path: Path, row: dict[str, Any]) -> None:
    arr = load_output_array(output_path)
    arr.append(row)
    write_output_array(output_path, arr)


def initialize_output(output_path: Path, append_output: bool) -> None:
    # 统一改为增写模式：已有文件不清空，只在不存在时初始化为空数组。
    if output_path.exists():
        return
    write_output_array(output_path, [])


def split_chunks(items: list[Path], size: int) -> list[list[Path]]:
    if size <= 0:
        raise ValueError("--cases-per-process 必须大于 0")
    return [items[i : i + size] for i in range(0, len(items), size)]


def process_case_chunk(task: dict[str, Any]) -> list[dict[str, Any]]:
    client = OpenAICompatClient(
        api_key=task["api_key"],
        base_url=task["base_url"],
        timeout=int(task["timeout"]),
    )
    model = str(task["model"])
    llm_config = dict(task["llm_config"])
    system_prompt = str(task["system_prompt"])
    prompt_mode = str(task.get("prompt_mode", "zh"))
    case_paths = [Path(x) for x in task["case_paths"]]

    return [
        run_case_judge(
            client,
            p,
            model=model,
            llm_config=llm_config,
            system_prompt=system_prompt,
            prompt_mode=prompt_mode,
        )
        for p in case_paths
    ]


def run_llm_batch(args: argparse.Namespace) -> dict[str, Any]:
    if not args.api_key:
        raise ValueError("Missing API key: provide --api-key or env LLM_API_KEY")

    case_files = iter_case_files(
        case_file=args.case_file,
        benchmark_dir=args.benchmark_dir,
        limit=args.limit,
        random_sample=args.random_sample,
        random_seed=args.random_seed,
    )
    if not case_files:
        raise FileNotFoundError("No case files found")

    if args.random_sample > 0 and not args.case_file:
        print(
            f"Random sampled {len(case_files)} cases (seed={args.random_seed}): "
            + ", ".join(x.name for x in case_files)
        )

    output_path = resolve_output_path(args.output_json)
    initialize_output(output_path, append_output=args.append_output)

    llm_config = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "enable_thinking": args.enable_thinking,
        "thinking_type": args.thinking_type,
        "save_thinking": args.save_thinking,
        "thinking_fallback": True,
    }
    effective_prompt_mode = normalize_prompt_mode(args.prompt_mode, use_english_prompt=bool(args.use_english_prompt))
    system_prompt = load_system_prompt(
        args.system_prompt_file,
        prompt_mode=effective_prompt_mode,
        english_system_prompt_file=args.english_system_prompt_file,
    )

    total = len(case_files)
    cases_per_process = int(args.cases_per_process)
    if cases_per_process <= 0:
        raise ValueError("--cases-per-process must be > 0")

    process_need = math.ceil(total / cases_per_process)
    max_processes = int(args.max_processes)
    if max_processes > 0:
        process_count = min(process_need, max_processes)
    else:
        process_count = process_need
    process_count = max(1, process_count)

    print(
        f"Total={total}, cases/process={cases_per_process}, "
        f"calculated_processes={process_need}, actual_processes={process_count}"
    )
    print(f"Prompt mode: {effective_prompt_mode}")
    print(
        "System prompt source: "
        + ("llmsystem_prompt" if bool(args.use_llm_system_prompt) else "system_prompt")
    )

    ok_count = 0
    finished = 0

    if process_count == 1:
        client = UnifiedOpenAIClient(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
        for idx, case_path in enumerate(case_files, start=1):
            if args.verbose:
                print(f"[{idx}/{total}] Processing: {case_path.name}")
            row = run_case_judge(
                client,
                case_path,
                model=args.model,
                llm_config=llm_config,
                system_prompt=system_prompt,
                prompt_mode=effective_prompt_mode,
            )
            append_result(output_path, row)
            if not row.get("error"):
                ok_count += 1
        return {
            "output_path": str(output_path),
            "total": total,
            "ok": ok_count,
        }

    chunks = split_chunks(case_files, cases_per_process)
    tasks: list[dict[str, Any]] = []
    for chunk in chunks:
        tasks.append(
            {
                "api_key": args.api_key,
                "base_url": args.base_url,
                "timeout": args.timeout,
                "model": args.model,
                "llm_config": llm_config,
                "system_prompt": system_prompt,
                "prompt_mode": effective_prompt_mode,
                "case_paths": [str(p) for p in chunk],
            }
        )

    with ProcessPoolExecutor(max_workers=process_count) as executor:
        future_to_task = {executor.submit(process_case_chunk, t): t for t in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                chunk_rows = future.result()
            except Exception as exc:
                chunk_rows = []
                for case_path_str in task.get("case_paths", []):
                    p = Path(case_path_str)
                    row = {
                        "case_file": p.name,
                        "case_id": p.stem,
                        "tnm_ret": "",
                        "关键临床特征是否获取": "否",
                        "分类": "",
                        "final_answer": "",
                        "model": args.model,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    if args.save_thinking:
                        row["think"] = ""
                    chunk_rows.append(row)

            for row in chunk_rows:
                finished += 1
                if args.verbose:
                    print(f"[{finished}/{total}] Done: {row.get('case_file', '')}")
                append_result(output_path, row)
                if not row.get("error"):
                    ok_count += 1

    return {
        "output_path": str(output_path),
        "total": total,
        "ok": ok_count,
    }

def run_with_single_model(args: argparse.Namespace) -> dict[str, Any]:
    round_idx = 0
    consecutive_full_zero_success = 0
    last_summary: dict[str, Any] = {"output_path": "", "total": 0, "ok": 0}

    while True:
        round_idx += 1
        print(f"模型 {args.model} 第 {round_idx} 次全量实验")
        summary = run_llm_batch(args)
        last_summary = summary

        ok = int(summary.get("ok", 0))
        total = int(summary.get("total", 0))
        if ok <= 0 and total > 0:
            consecutive_full_zero_success += 1
            print(f"全量实验 0 成功（连续 {consecutive_full_zero_success}/3）: model={args.model}")
        else:
            consecutive_full_zero_success = 0
            print(f"该模型本轮已出现成功病例（{ok}/{total}），切换到下一个模型。")
            break

        if consecutive_full_zero_success >= 3:
            print(f"模型连续三次全量实验无成功，停止该模型并切换下一个: {args.model}")
            break

    return {
        "model": str(args.model),
        "output_json": str(last_summary.get("output_path", "")),
        "success_count": int(last_summary.get("ok", 0)),
        "all_case_count": int(last_summary.get("total", 0)),
        "aborted_no_success": consecutive_full_zero_success >= 3,
        "rounds": round_idx,
    }


def main() -> None:
    args = parse_args()
    argv = sys.argv[1:]
    explicit_model = is_model_explicitly_set(argv)
    explicit_base_url = is_flag_explicitly_set(argv, "--base-url")
    explicit_api_key = is_flag_explicitly_set(argv, "--api-key")

    if explicit_model:
        route_base_url, route_api_key = common_endpoint_for_model(str(args.model), route_qwen_to_siliconflow=True, route_kimi_to_siliconflow=False)
        if not explicit_base_url:
            args.base_url = route_base_url
        if not explicit_api_key:
            args.api_key = route_api_key

        summary = run_with_single_model(args)
        print(f"结果已输出: {summary['output_json']}")
        print(
            f"模型={summary['model']}，轮次={summary['rounds']}，"
            f"最后一轮成功={summary['success_count']}/{summary['all_case_count']}"
        )
        return

    if not MODEL_NAMES:
        raise ValueError("MODEL_NAMES 为空，请先在 code/inference/direct_llm_batch_infer.py 中配置至少一个模型。")

    base_output = resolve_output_path(args.output_json)
    total_models = len(MODEL_NAMES)
    failed_models: list[str] = []
    bad_path = failed_models_path(base_output)

    for idx, model_name in enumerate(MODEL_NAMES, start=1):
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model_name
        model_args.output_json = str(output_path_for_model(base_output, model_name))
        model_args.append_output = False

        route_base_url, route_api_key = common_endpoint_for_model(model_name, route_qwen_to_siliconflow=True, route_kimi_to_siliconflow=False)
        if not explicit_base_url:
            model_args.base_url = route_base_url
        if not explicit_api_key:
            model_args.api_key = route_api_key

        print(f"\n===== 模型 {idx}/{total_models}: {model_name} =====")
        summary = run_with_single_model(model_args)
        if summary.get("aborted_no_success"):
            failed_models.append(model_name)

    write_failed_models(bad_path, failed_models)
    print(f"\n连续三次全量0成功模型记录文件: {bad_path}")
    if failed_models:
        print("失败模型: " + ", ".join(failed_models))
    else:
        print("所有模型都在最多3次全量实验内出现了成功病例。")


if __name__ == "__main__":
    main()




















