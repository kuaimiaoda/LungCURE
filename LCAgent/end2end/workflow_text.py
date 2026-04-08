import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

import cdss as cdss
import tnm as tnm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path("lg/workflow_results/outputs")
DEFAULT_PROVIDER = "zai-org/GLM-4.6V"
CDSS_TREATMENT_STAGE_NODE = "1765802928890"
CDSS_FIRST_STRUCTURED_NODE = "1765803586857"
ENGLISH_ROUTE_NAMES = {
    "可手术NCCN": "Resectable NCCN",
    "潜在可切除": "Potentially Resectable",
    "局晚期": "Locally Advanced",
    "晚期驱动基因阴性一线": "Advanced Driver-Negative First Line",
    "晚期驱动基因阳性一线": "Advanced Driver-Positive First Line",
    "寡转移": "Oligometastatic",
    "晚期驱动基因阳性后线": "Advanced Driver-Positive Later Line",
    "晚期驱动基因阳性二线及后线": "Advanced Driver-Positive Second Line and Beyond",
}
ENGLISH_STRUCTURED_KEY_MAP = {
    "病理类型": "Histology",
    "体力评分": "Performance Status",
    "驱动基因": "Driver Gene",
    "PDL1表达": "PD-L1 Expression",
    "高危因素": "High-Risk Factors",
    "治疗阶段": "Treatment Stage",
    "既往治疗方案": "Previous Treatment",
    "免疫禁忌": "Immunotherapy Contraindication",
    "转移类型": "Metastatic Pattern",
    "是否一线治疗": "First-line Treatment",
    "TNM分期": "TNM Stage",
    "综合分期": "Overall Stage",
}


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def sanitize_fs_name(name: str) -> str:
    value = re.sub(r'[\\/:*?"<>|]+', "_", str(name or "").strip())
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value or "model"


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} 必须是对象映射，当前类型: {type(data).__name__}")
    return data


def load_api_model_name(config_path: Path, provider: str = "") -> str:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误: {config_path}")

    selected: dict[str, Any] = data
    active_provider = str(provider or "").strip() or str(data.get("active_provider", "")).strip()
    providers = data.get("providers")
    if active_provider and isinstance(providers, dict):
        provider_data = providers.get(active_provider)
        if isinstance(provider_data, dict):
            selected = provider_data

    model = str(selected.get("model", "gpt-5.2")).strip() or "gpt-5.2"
    return model


def configure_provider_override(config_path: Path, provider: str) -> Path:
    """Use a single source of truth for API keys and select provider via env override."""
    provider_name = str(provider or "").strip()
    if not provider_name:
        os.environ.pop("ACTIVE_PROVIDER_OVERRIDE", None)
        return config_path

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误: {config_path}")

    providers = data.get("providers")
    if not isinstance(providers, dict):
        raise ValueError(f"配置文件缺少 providers 映射: {config_path}")
    provider_data = providers.get(provider_name)
    if not isinstance(provider_data, dict):
        raise ValueError(f"配置文件中不存在 provider: {provider_name}")

    os.environ["ACTIVE_PROVIDER_OVERRIDE"] = provider_name
    return config_path


def resolve_input_files(input_arg: str, run_all: bool, max_files: int | None, data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"找不到数据目录: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"数据目录不是文件夹: {data_dir}")

    if input_arg:
        path = Path(input_arg)
        if path.exists():
            files = [path]
        else:
            fallback = data_dir / input_arg
            if not fallback.exists():
                raise FileNotFoundError(f"找不到输入文件: {input_arg}（已尝试: {fallback}）")
            files = [fallback]
    else:
        files = sorted(data_dir.glob("benchmark_*.json"))
        if not files:
            raise FileNotFoundError(f"{data_dir} 目录下未找到 benchmark_*.json")
        if not run_all:
            files = files[:1]

    if max_files is not None:
        files = files[:max_files]
    return files


def infer_language(input_file: Path) -> str:
    name = input_file.name.lower()
    if "chinese" in name:
        return "Chinese"
    if "english" in name:
        return "English"
    raise ValueError(f"无法从文件名推断语言: {input_file.name}")


def infer_seed(input_file: Path) -> str:
    match = re.search(r"seed(\d+)", input_file.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"无法从文件名推断 seed: {input_file.name}")
    return match.group(1)


def pick_case_path(case_obj: dict[str, Any], language: str) -> str:
    key = "Chinese_file_name" if language == "Chinese" else "English_file_name"
    if key not in case_obj:
        raise KeyError(f"病例缺少字段: {key}")
    return str(case_obj[key])


def parse_tnm_stage_text(final_answer: str) -> str:
    text = str(final_answer or "")
    patterns = [
        r"综合分期结论\s*#+\s*\*\*(.+?)\*\*",
        r"综合分期结论\s*#+\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip()
            if value:
                return value
    return "未知"


def classify_stage_bucket(stage_text: str) -> str:
    text = str(stage_text).upper()
    if "IV" in text or "Ⅳ" in text:
        return "晚期"
    return "早期"


def build_tnm_for_cdss(tnm_stage: str, stage_text: str) -> str:
    compact_tnm = re.sub(r"\s+", "", tnm_stage or "")
    return f"TNM分期: {compact_tnm}\n综合分期: {stage_text or '未知'}"


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def build_case_query(language: str, user_query: str) -> str:
    base = user_query.strip()
    if language == "English":
        rule = (
            "Answer entirely in English. Do not output any Chinese characters. "
            "Use English for all headings, explanations, JSON values, and final answers. "
            "If you output structured JSON for CDSS extraction, keep the JSON keys exactly as the original Chinese schema and only make the values English. "
            "Do not translate after generation. Respond directly in English."
        )
    else:
        rule = "请全程使用中文回答。"
    return f"{base}\n{rule}".strip() if base else rule


def normalize_stage_text_for_export(stage_text: str, language: str) -> str:
    text = str(stage_text or "").strip()
    if language != "English":
        return text or "未知"
    if not text:
        return "Unknown"
    if text == "需人工核定":
        return "Manual Review Required"
    normalized = text.replace("期", "")
    if normalized == "未知":
        return "Unknown"
    return normalized


def normalize_stage_bucket_for_export(stage_bucket: str, language: str) -> str:
    if language != "English":
        return stage_bucket
    return {"早期": "Early", "晚期": "Advanced"}.get(stage_bucket, stage_bucket or "Unknown")


def build_export_tnm_for_cdss(tnm_stage: str, stage_text: str, language: str) -> str:
    compact_tnm = re.sub(r"\s+", "", tnm_stage or "")
    if language == "English":
        return f"TNM Stage: {compact_tnm}\nOverall Stage: {normalize_stage_text_for_export(stage_text, language)}"
    return f"TNM分期: {compact_tnm}\n综合分期: {stage_text or '未知'}"


def localize_route_name(route_name: str, language: str) -> str:
    if language != "English":
        return route_name
    return ENGLISH_ROUTE_NAMES.get(route_name, route_name or "")


def localize_structured_info(data: dict[str, Any], language: str) -> dict[str, Any]:
    if language != "English":
        return data

    value_map = {
        "鳞癌": "Squamous",
        "非鳞癌": "Non-squamous",
        "未知": "Unknown",
        "无": "None",
        "有": "Present",
        "0-1分": "0-1",
        "2分": "2",
        "3-4分": "3-4",
        "需人工核定": "Manual Review Required",
    }
    treatment_stage_map = {
        "0": "Curative Preoperative",
        "1": "Curative Postoperative",
        "未知": "Unknown",
    }
    prev_treatment_map = {
        "1": "Untreated",
        "未知": "Unknown",
    }
    metastasis_map = {
        "0": "Oligometastatic",
        "1": "Extensive",
        "未知": "Unknown",
    }
    first_line_map = {
        "1": "Yes",
        "0": "No",
        "未知": "Unknown",
    }

    result: dict[str, Any] = {}
    for key, value in data.items():
        out_key = ENGLISH_STRUCTURED_KEY_MAP.get(key, key)
        out_value: Any = value
        if isinstance(value, str):
            out_value = value_map.get(value, value)
            if key == "治疗阶段":
                out_value = treatment_stage_map.get(value, out_value)
            elif key == "既往治疗方案":
                out_value = prev_treatment_map.get(value, out_value)
            elif key == "转移类型":
                out_value = metastasis_map.get(value, out_value)
            elif key == "是否一线治疗":
                out_value = first_line_map.get(value, out_value)
            elif key == "综合分期":
                out_value = normalize_stage_text_for_export(value, language)
        result[out_key] = out_value
    return result


def localize_case_output(case_data: dict[str, Any], language: str) -> dict[str, Any]:
    if language != "English":
        return case_data

    tnm_result = dict(case_data.get("tnm_result", {}))
    parsed = dict(tnm_result.get("parsed", {}))
    stage_text = str(tnm_result.get("stage_text", ""))
    compact_tnm = str(parsed.get("Final_TNM", "")).replace(" ", "")

    tnm_result["tnm_for_cdss"] = build_export_tnm_for_cdss(compact_tnm, stage_text, language)
    tnm_result["stage_result"] = normalize_stage_bucket_for_export(str(tnm_result.get("stage_result", "")), language)
    tnm_result["stage_text"] = normalize_stage_text_for_export(stage_text, language)

    cdss_route = dict(case_data.get("cdss_route", {}))
    cdss_route["route_name"] = localize_route_name(str(cdss_route.get("route_name", "")), language)

    return {
        **case_data,
        "tnm_result": tnm_result,
        "cdss_result": str(case_data.get("cdss_result", "")),
        "cdss_route": cdss_route,
        "cdss_structured_info": localize_structured_info(
            dict(case_data.get("cdss_structured_info", {})),
            language,
        ),
    }


def parse_json_string(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_cdss_node_titles(workflow_path: Path) -> dict[str, str]:
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    nodes = data.get("workflow", {}).get("graph", {}).get("nodes", [])
    if not isinstance(nodes, list):
        return {}

    titles: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip()
        node_data = node.get("data", {})
        if node_id and isinstance(node_data, dict):
            titles[node_id] = str(node_data.get("title", node_id))
    return titles


def normalize_route_name(title: str) -> str:
    if not title:
        return ""
    name = re.sub(r"\(.*?\)", "", title).strip()
    return name


def extract_cdss_route(cdss_state: dict[str, Any], node_titles: dict[str, str]) -> dict[str, Any]:
    node_outputs = cdss_state.get("node_outputs", {})
    if not isinstance(node_outputs, dict):
        node_outputs = {}

    stage_flag_raw = node_outputs.get(CDSS_TREATMENT_STAGE_NODE, {}).get("result")
    try:
        stage_flag = int(stage_flag_raw)
    except Exception:
        stage_flag = None

    selected_treatment_node = ""
    for node_id, output in node_outputs.items():
        if node_id == "llm":
            continue
        if not isinstance(output, dict):
            continue
        if "text" not in output:
            continue
        title = node_titles.get(node_id, "")
        if not title:
            continue
        if any(flag in title for flag in ("直接回复", "开始", "文档提取器", "病人信息提取")):
            continue
        selected_treatment_node = node_id

    route_name = normalize_route_name(node_titles.get(selected_treatment_node, ""))
    return {
        "stage_flag": stage_flag,
        "selected_treatment_node": selected_treatment_node,
        "route_name": route_name,
    }


def run_single_case(
    case_key: str,
    case_obj: dict[str, Any],
    language: str,
    query: str,
    cdss_app: Any,
    cdss_node_titles: dict[str, str],
) -> dict[str, Any]:
    pdf_path = pick_case_path(case_obj, language)
    case_query = build_case_query(language, query)

    tnm_state = {
        "case_id": case_obj.get("case_id", case_key),
        "benchmark_key": case_key,
        "language": language.lower(),
        "query": case_query,
        "input_file_name": pdf_path,
        "input_file_exists": Path(pdf_path).exists(),
    }
    tnm_output = tnm.app1.invoke(tnm_state)

    final_tnm = str(tnm_output.get("tnm_stage", "")).strip()
    stage_text = parse_tnm_stage_text(str(tnm_output.get("final_answer", "")))
    tnm_for_cdss = build_tnm_for_cdss(final_tnm, stage_text)
    stage_bucket = classify_stage_bucket(stage_text)

    cdss_output = cdss.run_once(cdss_app, query=case_query, tnm_ret=tnm_for_cdss, files=[pdf_path])
    cdss_structured_info = parse_json_string(
        str(cdss_output.get("node_outputs", {}).get(CDSS_FIRST_STRUCTURED_NODE, {}).get("result", ""))
    )
    cdss_route = extract_cdss_route(cdss_output, cdss_node_titles)

    return localize_case_output({
        "case_id": case_obj.get("case_id", case_key),
        "pdf_path": pdf_path,
        "tnm_result": {
            "tnm_for_cdss": build_export_tnm_for_cdss(final_tnm, stage_text, language),
            "tnm_text": str(tnm_output.get("final_answer", "")),
            "stage_result": normalize_stage_bucket_for_export(stage_bucket, language),
            "stage_text": normalize_stage_text_for_export(stage_text, language),
            "parsed": {
                "T": str(tnm_output.get("T", "")),
                "N": str(tnm_output.get("N", "")),
                "M": str(tnm_output.get("M", "")),
                "Final_TNM": " ".join(
                    part for part in [
                        str(tnm_output.get("T", "")).strip(),
                        str(tnm_output.get("N", "")).strip(),
                        str(tnm_output.get("M", "")).strip(),
                    ] if part
                ),
            },
        },
        "cdss_result": str(cdss_output.get("final_answer", "")),
        "cdss_route": {
            **cdss_route,
            "route_name": localize_route_name(str(cdss_route.get("route_name", "")), language),
        },
        "cdss_structured_info": cdss_structured_info,
    }, language)


def process_input_file(
    input_file: Path,
    case_limit: int | None,
    query: str,
    output_dir: Path,
    model_name: str,
    cdss_app: Any,
    cdss_node_titles: dict[str, str],
) -> Path:
    benchmark = load_json_object(input_file, "benchmark 文件")
    language = infer_language(input_file)
    seed = infer_seed(input_file)
    case_items = list(benchmark.items())
    if case_limit is not None:
        case_items = case_items[:case_limit]

    log(f"开始处理 {input_file.name}，语言={language}，病例数={len(case_items)}")

    cases: dict[str, Any] = {}
    for idx, (case_key, case_obj) in enumerate(case_items, start=1):
        log(f"[{input_file.name}] [{idx}/{len(case_items)}] 处理病例 {case_key}")
        if not isinstance(case_obj, dict):
            cases[case_key] = {
                "case_id": case_key,
                "error": f"病例结构错误，实际类型: {type(case_obj).__name__}",
            }
            continue

        try:
            cases[case_key] = run_single_case(
                case_key=case_key,
                case_obj=case_obj,
                language=language,
                query=query,
                cdss_app=cdss_app,
                cdss_node_titles=cdss_node_titles,
            )
        except Exception as exc:
            pdf_path = ""
            try:
                pdf_path = pick_case_path(case_obj, language)
            except Exception:
                pass
            cases[case_key] = {
                "case_id": case_obj.get("case_id", case_key),
                "pdf_path": pdf_path,
                "tnm_result": {},
                "cdss_result": "",
                "cdss_route": {
                    "stage_flag": None,
                    "selected_treatment_node": "",
                    "route_name": "",
                },
                "cdss_structured_info": {},
                "error": str(exc),
            }
            log(f"[{input_file.name}] 病例 {case_key} 执行失败: {exc}")

    payload = {
        "source_input": str(input_file.resolve()),
        "language": language,
        "cases": cases,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = sanitize_fs_name(model_name)
    output_path = output_dir / f"{safe_model_name}_output_{language}_seed{seed}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"输出已写入: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="串联执行 TNM 与 CDSS 工作流")
    parser.add_argument(
        "--input",
        default="",
        help="单个输入文件；为空时默认读取 --data-dir 指定目录",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data_text"),
        help="JSON 数据文件目录",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="执行 data 目录下全部 benchmark 文件",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="最多执行多少个输入 JSON 文件",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="每个输入 JSON 最多执行多少个病例",
    )
    parser.add_argument(
        "--query",
        default="",
        help="传给工作流的 query（对应 sys.query）",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "api_config.yaml"),
        help="API 配置文件路径",
    )
    parser.add_argument(
        "--provider",
        default="gpt-5.2",
        help="指定 providers 下的 provider 名称；为空时使用配置文件 active_provider",
    )
    parser.add_argument(
        "--tnm-workflow",
        default=str(PROJECT_ROOT /"LCAgent/end2end/files/stage1.yml"),
        help="TNM 工作流 YAML 路径",
    )
    parser.add_argument(
        "--cdss-workflow",
        default=str(PROJECT_ROOT /"LCAgent/end2end/files/stage2.yml"),
        help="CDSS 工作流 YAML 路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "results_end2end/outputs/agent_outputs/agent_text_outputs"),
        help="输出目录",
    )
    args = parser.parse_args()

    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files 必须大于 0")
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases 必须大于 0")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    config_path = configure_provider_override(config_path, args.provider)

    tnm_workflow_path = Path(args.tnm_workflow)
    cdss_workflow_path = Path(args.cdss_workflow)
    if not tnm_workflow_path.exists():
        raise FileNotFoundError(f"找不到 TNM 工作流文件: {tnm_workflow_path}")
    if not cdss_workflow_path.exists():
        raise FileNotFoundError(f"找不到 CDSS 工作流文件: {cdss_workflow_path}")

    tnm.set_api_config_path(str(config_path))
    tnm.DIFY_WORKFLOW_PATH = tnm_workflow_path
    tnm._load_dify_workflow.cache_clear()
    tnm._dify_nodes_index.cache_clear()

    cdss.set_api_config_path(str(config_path))
    cdss.set_workflow_path(str(cdss_workflow_path))
    cdss_app = cdss.build_cdss_graph()
    cdss_node_titles = load_cdss_node_titles(cdss_workflow_path)

    data_dir = Path(args.data_dir)
    model_name = load_api_model_name(config_path, args.provider)
    input_files = resolve_input_files(args.input, args.all, args.max_files, data_dir)

    for input_file in input_files:
        process_input_file(
            input_file=input_file,
            case_limit=args.max_cases,
            query=args.query,
            output_dir=Path(args.output_dir),
            model_name=model_name,
            cdss_app=cdss_app,
            cdss_node_titles=cdss_node_titles,
        )


if __name__ == "__main__":
    main()
