
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.common.model_api import (
    UnifiedOpenAIClient,
    call_glm46v_via_zai as common_call_glm46v_via_zai,
    collect_stream_text as common_collect_stream_text,
    endpoint_for_model as common_endpoint_for_model,
    extract_message_content as common_extract_message_content,
    extract_reasoning_content as common_extract_reasoning_content,
    is_glm46v_model as common_is_glm46v_model,
)
from code.common.case_loader import (
    build_case_tnm_map as common_build_case_tnm_map,
    discover_pdf_case_files as common_discover_pdf_case_files,
    extract_pdf_text as common_extract_pdf_text,
    iter_case_files as common_iter_case_files,
    load_case_payload as common_load_case_payload,
)
from code.workflow.workflow_engine import GLOBAL_LLM_CONFIG, NSCLCWorkflowLangGraphReplay
DEFAULT_WORKFLOW_YML = str((PROJECT_ROOT / "非小细胞癌治疗-免提示-en.yml").resolve())

# 非 GLM 模型默认接口（apiyi）
DEFAULT_BASE_URL = "https://api.apiyi.com/v1"
DEFAULT_API_KEY = "sk-Te6gIPgo3nkjhP3w70B515C4F59b44B385058a6aDd514cE4"

# GLM 专用接口（z.ai）
GLM_BASE_URL = "https://api.z.ai/api/paas/v4/"
GLM_API_KEY = "d894e8a6c3f94e81bf0eff2548f6d17e.iR7nm5vuYTRsg3nX"


qwen_BASE_URL = "https://api.siliconflow.cn/v1"
qwen_API_KEY = "sk-aohepgqfelogggybswajmqqagljormvtfxcnaalvmwfuusuu"

MODEL_NAMES = [
    "GLM-4.6V",
    # "qwen3.5-397b-a17b",
    "Qwen/Qwen3.5-397B-A17B",
    "llama-4-maverick",
    "gpt-5.2",
    "claude-sonnet-4-6",
    # "grok-4", 
    "kimi-k2.5"
]
DEFAULT_MODEL = MODEL_NAMES[0]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 benchmark 病例并调用 main.py 工作流（真实 LLM 节点）。"
    )
    parser.add_argument(
        "--workflow-yml",
        default=DEFAULT_WORKFLOW_YML,
        help="显式指定要使用的 yml 文件路径（优先级高于 --workflow）。",
    )
    parser.add_argument(
        "--workflow",
        default=None,
        help="工作流 YAML 路径（兼容旧参数）。默认自动使用当前目录下第一个 .yml。",
    )
    parser.add_argument(
        "--benchmark-dir",
        default="benchmark_gt_Chinese_seed2024",
        help="病例目录（默认读取 *.md JSON）。",
    )
    parser.add_argument(
        "--read-pdf",
        action="store_true",
        help="显式开启后，才会从 benchmark 目录读取 PDF（含 seed_42pdf/seed42pdf）。",
    )
    parser.add_argument(
        "--case-file",
        default=None,
        help="只跑单个病例文件路径。设置后忽略 benchmark-dir。",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=0,
        help="从病例目录随机抽取 N 个样本进行实验，0 表示不启用。",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2026,
        help="随机抽样种子（用于复现实验）。",
    )
    parser.add_argument("--limit", type=int, default=0, help="限制处理病例数，0 表示全部。")
    parser.add_argument(
        "--cases-per-process",
        type=int,
        default=10,
        help="每个进程处理的病例数（程序将自动计算开启进程数）。",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=0,
        help="最大进程数上限，0 表示不限制。",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        help="覆盖工作流所有 llm 节点模型名。",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark_results.json",
        help="结果输出路径（标准 JSON 数组，逐病例增写）。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印节点级执行日志（含各 llm 节点提示词）。",
    )
    parser.add_argument(
        "--include-thinking",
        action="store_true",
        help="保留参数（兼容旧命令）。思考内容写入由 --save-thinking 控制。",
    )
    parser.add_argument(
        "--save-thinking",
        action="store_true",
        help="将思考内容保存到结果新字段 \"<think>\"（并从 final_answer 中剥离）。",
    )
    parser.add_argument(
        "--retry-seconds",
        type=int,
        default=3,
        help="仍有失败病例时，两轮重试之间的等待秒数。",
    )
    return parser.parse_args()


def is_model_explicitly_set(argv: list[str]) -> bool:
    return any(x == "--model" or x.startswith("--model=") for x in argv)


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


def resolve_workflow_path(workflow_arg: str | None) -> Path:
    if workflow_arg:
        path = Path(workflow_arg).resolve()
        if not path.exists():
            raise FileNotFoundError(f"工作流文件不存在: {path}")
        return path

    yml_files = sorted(Path(".").glob("*.yml"))
    if not yml_files:
        raise FileNotFoundError("当前目录未找到 .yml 工作流文件")
    return yml_files[0].resolve()


def iter_case_files(
    case_file: str | None,
    benchmark_dir: str,
    limit: int,
    random_sample: int,
    random_seed: int,
    read_pdf: bool = False,
) -> list[Path]:
    files = common_iter_case_files(
        case_file=case_file,
        benchmark_dir=benchmark_dir,
        limit=limit,
        random_sample=random_sample,
        random_seed=random_seed,
        read_pdf=bool(read_pdf),
        root_dir=PROJECT_ROOT,
        default_suffixes=("md", "txt", "json"),
    )

    if not case_file:
        if read_pdf:
            print(f"输入源: 文本+PDF 混合 ({len(files)} 例)")
        else:
            print(f"输入源: 文本 ({len(files)} 例)")
    return files


def discover_pdf_case_files(base_dir: Path) -> list[Path]:
    return common_discover_pdf_case_files(base_dir)


def build_case_tnm_map(benchmark_dir: str) -> dict[str, str]:
    return common_build_case_tnm_map(benchmark_dir, root_dir=PROJECT_ROOT)


def extract_pdf_text(pdf_path: Path) -> str:
    return common_extract_pdf_text(pdf_path)


def load_case_payload(case_path: Path, tnm_by_case_id: dict[str, str] | None = None) -> dict[str, Any]:
    return common_load_case_payload(case_path, tnm_by_case_id=tnm_by_case_id)

def precheck_case_payload(
    case_path: Path,
    tnm_by_case_id: dict[str, str] | None = None,
) -> tuple[bool, dict[str, Any] | None, str]:
    try:
        payload = load_case_payload(case_path, tnm_by_case_id=tnm_by_case_id)
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"

    query = str(payload.get("query", "")).strip()
    if not query:
        return False, payload, "query 为空"
    return True, payload, ""


def extract_message_content(content: Any) -> str:
    return common_extract_message_content(content)


def extract_category_from_final_answer(final_answer: str) -> str:
    text = str(final_answer or "").strip()
    if not text:
        return ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # 去掉常见标题符号
        line = line.lstrip("#").strip()
        if line:
            return line
    return ""


def extract_reasoning_content(message: dict[str, Any]) -> str:
    return common_extract_reasoning_content(message)


def is_glm46v_model(model_name: str) -> bool:
    name = str(model_name or "").strip().lower()
    return name in {"glm-4.6v", "glm4.6v", "glm-4-6v"}


def is_glm_model(model_name: str) -> bool:
    name = str(model_name or "").strip().lower()
    return name.startswith("glm")


def is_qwen397_model(model_name: str) -> bool:
    name = str(model_name or "").strip().lower()
    return "qwen3.5-397b-a17b" in name


def qwen397_candidate_models(model_name: str) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []

    def add(name: str) -> None:
        text = str(name or "").strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(text)

    add(model_name)
    env_alias = os.getenv("QWEN397_MODEL_NAME", "").strip()
    if env_alias:
        add(env_alias)

    if is_qwen397_model(model_name):
        add("Qwen/Qwen3.5-397B-A17B")
        add("Qwen/Qwen3.5-397B-A17B-Instruct")
        add("qwen3.5-397b-a17b")
        add("qwen3.5-397b-a17b-instruct")

    return candidates


def endpoint_for_model(model_name: str) -> tuple[str, str]:
    return common_endpoint_for_model(model_name, route_qwen_to_siliconflow=True, route_kimi_to_siliconflow=False)


def call_glm46v_via_zai(
    *,
    api_key: str,
    model_name: str,
    messages: list[dict[str, Any]],
) -> tuple[str, str]:
    return common_call_glm46v_via_zai(api_key=api_key, model_name=model_name, messages=messages)


THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def split_think_from_text(text: str) -> tuple[str, str]:
    raw = str(text or "")
    thinks: list[str] = []
    for m in THINK_BLOCK_RE.finditer(raw):
        content = str(m.group(1) or "").strip()
        if content:
            thinks.append(content)
    cleaned = THINK_BLOCK_RE.sub("", raw).strip()
    return cleaned, "\n\n".join(thinks)


def load_output_array(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    raw = output_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 兼容历史 JSONL：每行一个对象
        items: list[dict[str, Any]] = []
        for line in raw.splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    items.append(obj)
            except json.JSONDecodeError:
                continue
        return items
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def append_result_line(output_path: Path, item: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = load_output_array(output_path)
    arr.append(item)
    payload = json.dumps(arr, ensure_ascii=False, indent=2)
    try:
        output_path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        fallback = output_path.with_name(f"{output_path.stem}_fallback{output_path.suffix}")
        fallback.write_text(payload, encoding="utf-8")
        print(f"[WARN] ??????????? fallback ??: {fallback} ({type(exc).__name__}: {exc})")


def build_success_txt_path(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_success.txt"


def load_success_case_ids(success_txt_path: Path) -> set[str]:
    if not success_txt_path.exists():
        return set()
    ids: set[str] = set()
    for line in success_txt_path.read_text(encoding="utf-8").splitlines():
        case_id = line.strip()
        if case_id:
            ids.add(case_id)
    return ids


def save_success_case_ids(success_txt_path: Path, case_ids: set[str]) -> None:
    success_txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = sorted(x for x in case_ids if x)
    text = "\n".join(lines)
    if text:
        text += "\n"
    try:
        success_txt_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"[WARN] ??????ID????????: {success_txt_path} ({type(exc).__name__}: {exc})")


def collect_success_case_ids_from_output(output_path: Path) -> set[str]:
    rows = load_output_array(output_path)
    ids: set[str] = set()
    for row in rows:
        if row.get("error"):
            continue
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            ids.add(case_id)
    return ids


class OpenAICompatClient(UnifiedOpenAIClient):
    pass

def collect_stream_output_for_thinking(
    stream: Any,
    *,
    capture_thinking: bool,
) -> tuple[str, str]:
    return common_collect_stream_text(stream, capture_thinking=capture_thinking)


def build_llm_executor(
    client: OpenAICompatClient,
    save_thinking: bool = False,
):
    def executor(
        *,
        node_id: str,
        node_title: str,
        model_name: str,
        params: dict[str, Any],
        messages: list[dict[str, str]],
        node_data: dict[str, Any],
        state: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        request: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }

        for key in ("max_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty"):
            value = params.get(key)
            if value is not None:
                request[key] = value

        if common_is_glm46v_model(model_name):
            final_text, reasoning_text = common_call_glm46v_via_zai(
                api_key=client.api_key,
                model_name=model_name,
                messages=messages,
            )
            if save_thinking and reasoning_text:
                return f"<think>\n{reasoning_text}\n</think>\n{final_text}".strip()
            return final_text

        if save_thinking:
            stream = client.chat_completions_stream(request)
            final_text, reasoning_text = common_collect_stream_text(
                stream,
                capture_thinking=True,
            )
            if reasoning_text:
                return f"<think>\n{reasoning_text}\n</think>\n{final_text}".strip()
            return final_text

        resp = client.chat_completions(request)
        choices = resp.get("choices", [])
        if not choices:
            raise RuntimeError(f"响应中无 choices: {resp}")
        message = choices[0].get("message", {})
        final_text = common_extract_message_content(message.get("content", "")).strip()
        return final_text

    return executor


def build_case_result_item(
    replay: NSCLCWorkflowLangGraphReplay,
    case_path: Path,
    tnm_by_case_id: dict[str, str] | None = None,
    save_thinking: bool = False,
) -> dict[str, Any]:
    case = load_case_payload(case_path, tnm_by_case_id=tnm_by_case_id)
    try:
        run_result = replay.invoke(
            extracted_json_text=None,
            tnm_ret=case["tnm_ret"],
            document_text=case["query"],
            query=case["query"],
        )
        final_answer_raw = str(run_result.get("final_answer", ""))
        final_answer_clean, think_text = split_think_from_text(final_answer_raw)
        final_answer = final_answer_clean if save_thinking else final_answer_raw

        item: dict[str, Any] = {
            "case_file": case["case_file"],
            "case_id": case["case_id"],
            "tnm_ret": case["tnm_ret"],
            "关键临床特征是否获取": "是",
            "分类": extract_category_from_final_answer(final_answer),
            "final_answer": final_answer,
            "error": "",
        }
        if save_thinking:
            item["<think>"] = think_text
        return item
    except Exception as exc:
        item = {
            "case_file": case["case_file"],
            "case_id": case["case_id"],
            "tnm_ret": case["tnm_ret"],
            "关键临床特征是否获取": "是",
            "分类": "",
            "final_answer": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if save_thinking:
            item["<think>"] = ""
        return item


def split_chunks(items: list[Path], size: int) -> list[list[Path]]:
    if size <= 0:
        raise ValueError("--cases-per-process 必须大于 0")
    return [items[i : i + size] for i in range(0, len(items), size)]


def process_case_chunk(task: dict[str, Any]) -> list[dict[str, Any]]:
    api_key = task["api_key"]
    base_url = task["base_url"]
    workflow_path = task["workflow_path"]
    model = task["model"]
    save_thinking = task.get("save_thinking", False)
    verbose = task["verbose"]
    case_paths = [Path(x) for x in task["case_paths"]]
    tnm_by_case_id = task.get("tnm_by_case_id", {})

    GLOBAL_LLM_CONFIG["model"] = model
    client = UnifiedOpenAIClient(api_key=api_key, base_url=base_url)
    replay = NSCLCWorkflowLangGraphReplay(
        workflow_path=workflow_path,
        llm_executor=build_llm_executor(
            client,
            save_thinking=save_thinking,
        ),
        log_enabled=verbose,
    )
    return [
        build_case_result_item(
            replay,
            p,
            tnm_by_case_id=tnm_by_case_id,
            save_thinking=save_thinking,
        )
        for p in case_paths
    ]


def run_benchmark(
    args: argparse.Namespace,
    case_files_override: list[Path] | None = None,
    tnm_by_case_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    default_base_url, default_api_key = common_endpoint_for_model(str(args.model), route_qwen_to_siliconflow=True, route_kimi_to_siliconflow=False)
    api_key = os.getenv("LLM_API_KEY", default_api_key)
    base_url = os.getenv("LLM_BASE_URL", default_base_url)
    workflow_arg = getattr(args, "workflow_yml", None) or args.workflow
    workflow_path = resolve_workflow_path(workflow_arg)

    if case_files_override is None:
        case_files = iter_case_files(
            case_file=args.case_file,
            benchmark_dir=args.benchmark_dir,
            limit=args.limit,
            random_sample=args.random_sample,
            random_seed=args.random_seed,
            read_pdf=bool(args.read_pdf),
        )
    else:
        case_files = case_files_override
    if not case_files:
        raise FileNotFoundError("没有找到可处理的病例文件")
    if tnm_by_case_id is None:
        tnm_by_case_id = build_case_tnm_map(args.benchmark_dir)

    valid_case_files: list[Path] = []
    skipped_count = 0
    for p in case_files:
        ok, _payload, reason = precheck_case_payload(p, tnm_by_case_id=tnm_by_case_id)
        if ok:
            valid_case_files.append(p)
            continue
        skipped_count += 1
        print(f"[SKIP] {p.name}: {reason}")

    case_files = valid_case_files
    if not case_files:
        print("当前批次无可用病例（均被跳过）。")
        return []
    if skipped_count > 0:
        print(f"本轮跳过空/损坏病例: {skipped_count} 例")

    if case_files_override is None and args.random_sample > 0 and not args.case_file:
        print(
            f"已随机抽样 {len(case_files)} 例（seed={args.random_seed}）: "
            + ", ".join(x.name for x in case_files)
        )

    total = len(case_files)
    output_path = Path(args.output_json).resolve()
    cases_per_process = int(args.cases_per_process)
    if cases_per_process <= 0:
        raise ValueError("--cases-per-process 必须大于 0")
    process_need = math.ceil(total / cases_per_process)
    max_processes = int(args.max_processes)
    if max_processes > 0:
        process_count = min(process_need, max_processes)
    else:
        process_count = process_need
    process_count = max(1, process_count)

    

    print(
        f"总病例数={total}, 每进程处理={cases_per_process}, "
        f"计算进程数={process_need}, 实际开启={process_count}"
    )
    print(f"模型路由: model={args.model}, base_url={base_url}")

    # 使用 api.py 中的模型配置，统一覆盖 main.py 的 llm 节点模型
    GLOBAL_LLM_CONFIG["model"] = args.model
    results: list[dict[str, Any]] = []

    if process_count == 1:
        client = UnifiedOpenAIClient(api_key=api_key, base_url=base_url)
        replay = NSCLCWorkflowLangGraphReplay(
            workflow_path=str(workflow_path),
            llm_executor=build_llm_executor(
                client,
                save_thinking=args.save_thinking,
            ),
            log_enabled=args.verbose,
        )
        for idx, case_path in enumerate(case_files, start=1):
            print(f"[{idx}/{total}] 处理中: {case_path.name}")
            item = build_case_result_item(
                replay,
                case_path,
                tnm_by_case_id=tnm_by_case_id,
                save_thinking=args.save_thinking,
            )
            results.append(item)
            append_result_line(output_path, item)
        return results

    chunks = split_chunks(case_files, cases_per_process)
    tasks: list[dict[str, Any]] = []
    for chunk in chunks:
        tasks.append(
            {
                "api_key": api_key,
                "base_url": base_url,
                "workflow_path": str(workflow_path),
                "model": args.model,
                "save_thinking": args.save_thinking,
                "verbose": args.verbose,
                "tnm_by_case_id": tnm_by_case_id,
                "case_paths": [str(p) for p in chunk],
            }
        )

    finished = 0
    try:
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            future_to_task = {executor.submit(process_case_chunk, t): t for t in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    chunk_results = future.result()
                except Exception as exc:
                    chunk_results = []
                    for case_path_str in task.get("case_paths", []):
                        p = Path(case_path_str)
                        chunk_results.append(
                            {
                                "case_file": p.name,
                                "case_id": p.stem,
                                "tnm_ret": "",
                                "关键临床特征是否获取": "是",
                                "分类": "",
                                "final_answer": "",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        if args.save_thinking:
                            chunk_results[-1]["<think>"] = ""
                for item in chunk_results:
                    finished += 1
                    print(f"[{finished}/{total}] 完成: {item.get('case_file', '')}")
                    results.append(item)
                    append_result_line(output_path, item)
    except Exception as pool_exc:
        print(f"多进程不可用，回退单进程执行: {type(pool_exc).__name__}: {pool_exc}")
        client = UnifiedOpenAIClient(api_key=api_key, base_url=base_url)
        replay = NSCLCWorkflowLangGraphReplay(
            workflow_path=str(workflow_path),
            llm_executor=build_llm_executor(
                client,
                save_thinking=args.save_thinking,
            ),
            log_enabled=args.verbose,
        )
        for idx, case_path in enumerate(case_files, start=1):
            print(f"[{idx}/{total}] 处理中: {case_path.name}")
            item = build_case_result_item(
                replay,
                case_path,
                tnm_by_case_id=tnm_by_case_id,
                save_thinking=args.save_thinking,
            )
            results.append(item)
            append_result_line(output_path, item)

    return results


def run_with_single_model(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output_json).resolve()
    success_txt_path = build_success_txt_path(output_path)
    tnm_by_case_id = build_case_tnm_map(args.benchmark_dir)

    all_case_files = iter_case_files(
        case_file=args.case_file,
        benchmark_dir=args.benchmark_dir,
        limit=args.limit,
        random_sample=args.random_sample,
        random_seed=args.random_seed,
        read_pdf=bool(args.read_pdf),
    )
    if not all_case_files:
        raise FileNotFoundError("没有找到可处理的病例文件")

    if args.random_sample > 0 and not args.case_file:
        print(
            f"已随机抽样 {len(all_case_files)} 例（seed={args.random_seed}）: "
            + ", ".join(x.name for x in all_case_files)
        )

    case_pairs: list[tuple[Path, str]] = []
    skipped_initial = 0
    for case_path in all_case_files:
        ok, payload, reason = precheck_case_payload(case_path, tnm_by_case_id=tnm_by_case_id)
        if not ok:
            skipped_initial += 1
            print(f"[SKIP] {case_path.name}: {reason}")
            continue
        assert payload is not None
        case_pairs.append((case_path, str(payload.get("case_id", "")).strip()))

    if skipped_initial > 0:
        print(f"预检查跳过病例: {skipped_initial} 例")

    if not case_pairs:
        print("无可处理病例（输入为空或读取失败）。")
        return {
            "model": str(args.model),
            "output_json": str(output_path),
            "success_count": 0,
            "all_case_count": 0,
            "aborted_no_success": False,
            "rounds": 0,
        }

    all_case_count = len(case_pairs)
    success_ids = load_success_case_ids(success_txt_path)
    success_ids.update(collect_success_case_ids_from_output(output_path))
    save_success_case_ids(success_txt_path, success_ids)

    pending_files = [p for p, cid in case_pairs if cid and cid not in success_ids]
    pending_files.extend([p for p, cid in case_pairs if not cid])

    round_idx = 0
    all_results: list[dict[str, Any]] = []
    consecutive_full_zero_success = 0
    aborted_no_success = False

    while pending_files:
        round_idx += 1
        current_pending = len(pending_files)
        is_full_round = current_pending == all_case_count
        print(f"第 {round_idx} 轮：待处理 {current_pending}/{all_case_count} 例")
        round_results = run_benchmark(
            args,
            case_files_override=pending_files,
            tnm_by_case_id=tnm_by_case_id,
        )
        all_results.extend(round_results)

        before_count = len(success_ids)
        for item in round_results:
            if item.get("error"):
                continue
            case_id = str(item.get("case_id", "")).strip()
            if case_id:
                success_ids.add(case_id)

        # 再次从输出 JSON 同步，兼容中断重启场景
        success_ids.update(collect_success_case_ids_from_output(output_path))
        save_success_case_ids(success_txt_path, success_ids)
        new_count = len(success_ids) - before_count

        if is_full_round and new_count <= 0:
            consecutive_full_zero_success += 1
            print(
                f"全量实验 0 成功（连续 {consecutive_full_zero_success}/3）: model={args.model}"
            )
        else:
            consecutive_full_zero_success = 0

        pending_files = [p for p, cid in case_pairs if cid and cid not in success_ids]
        pending_files.extend([p for p, cid in case_pairs if not cid])

        if consecutive_full_zero_success >= 3:
            aborted_no_success = True
            print(f"模型连续三次全量实验无成功，停止该模型并切换下一个: {args.model}")
            break

        if pending_files:
            if new_count <= 0:
                print("本轮无新增成功病例，继续重试未成功病例。")
            if args.retry_seconds > 0:
                print(f"等待 {args.retry_seconds} 秒后开始下一轮。")
                time.sleep(args.retry_seconds)

    print(f"结果已写入标准 JSON 数组: {output_path}")
    print(f"成功病例ID文件: {success_txt_path}")
    ok = len(success_ids)
    if aborted_no_success:
        print(f"完成: {ok}/{all_case_count} 成功（因连续三次全量0成功而提前停止）")
    else:
        print(f"完成: {ok}/{all_case_count} 成功（已全部成功）")

    return {
        "model": str(args.model),
        "output_json": str(output_path),
        "success_count": ok,
        "all_case_count": all_case_count,
        "aborted_no_success": aborted_no_success,
        "rounds": round_idx,
    }


def main() -> None:
    args = parse_args()
    explicit_model = is_model_explicitly_set(sys.argv[1:])

    if explicit_model:
        summary = run_with_single_model(args)
        if summary.get("aborted_no_success"):
            base_output = Path(args.output_json).resolve()
            bad_path = failed_models_path(base_output)
            write_failed_models(bad_path, [str(args.model)])
            print(f"已记录失败模型: {bad_path}")
        return

    base_output = Path(args.output_json).resolve()
    total_models = len(MODEL_NAMES)
    bad_path = failed_models_path(base_output)
    failed_models: list[str] = []
    for idx, model_name in enumerate(MODEL_NAMES, start=1):
        model_args = argparse.Namespace(**vars(args))
        model_args.model = model_name
        model_args.output_json = str(output_path_for_model(base_output, model_name))
        print(f"\n===== 模型 {idx}/{total_models}: {model_name} =====")
        summary = run_with_single_model(model_args)
        if summary.get("aborted_no_success"):
            failed_models.append(model_name)

    write_failed_models(bad_path, failed_models)
    print(f"连续三次全量0成功模型记录文件: {bad_path}")


if __name__ == "__main__":
    main()



