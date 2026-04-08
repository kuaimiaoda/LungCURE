import argparse
from pathlib import Path
from typing import Any

import cdss as cdss
import tnm as tnm
import workflow_text as base_workflow
from multimodal_pdf import MultimodalPdfExtractor, detect_language_from_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path("lg/workflow_results/outputs")
DEFAULT_PROVIDER = "gpt-5.2"


def _patch_pdf_readers(extractor: MultimodalPdfExtractor) -> None:
    original_tnm_read_pdf_text = tnm._read_pdf_text
    original_cdss_read_file_text = cdss._read_file_text

    def patched_tnm_read_pdf_text(file_name: str, max_chars: int = 20000) -> str:
        language = detect_language_from_path(file_name)
        text = extractor.extract_key_info(file_name, language=language)
        text = text.strip()
        if not text:
            raise RuntimeError("视觉模型关键事实提取结果为空")
        return text[:max_chars]

    def patched_cdss_read_file_text(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix != ".pdf":
            return original_cdss_read_file_text(path)

        language = detect_language_from_path(str(path))
        text = extractor.extract_key_info(str(path), language=language)
        return text.strip()

    tnm._read_pdf_text = patched_tnm_read_pdf_text
    cdss._read_file_text = patched_cdss_read_file_text

    # Keep references to avoid garbage collection and allow introspection.
    setattr(tnm, "_original_read_pdf_text", original_tnm_read_pdf_text)
    setattr(cdss, "_original_read_file_text", original_cdss_read_file_text)


def _patch_case_initial_inputs(extractor: MultimodalPdfExtractor) -> None:
    original_run_single_case = base_workflow.run_single_case

    def patched_run_single_case(
        case_key: str,
        case_obj: dict[str, Any],
        language: str,
        query: str,
        cdss_app: Any,
        cdss_node_titles: dict[str, str],
    ) -> dict[str, Any]:
        enriched_query = query
        try:
            pdf_path = base_workflow.pick_case_path(case_obj, language)
            key_info = extractor.extract_key_info(pdf_path, language=language)
            key_info = str(key_info or "").strip()
            if key_info:
                if str(language).lower() == "english":
                    block = f"[Vision-Extracted Key Case Facts]\n{key_info}"
                else:
                    block = f"[视觉模型提取关键病例信息]\n{key_info}"
                enriched_query = f"{query.strip()}\n\n{block}".strip()
        except Exception:
            enriched_query = query

        return original_run_single_case(
            case_key=case_key,
            case_obj=case_obj,
            language=language,
            query=enriched_query,
            cdss_app=cdss_app,
            cdss_node_titles=cdss_node_titles,
        )

    base_workflow.run_single_case = patched_run_single_case
    setattr(base_workflow, "_original_run_single_case", original_run_single_case)


def process_with_multimodal(
    input_file: Path,
    case_limit: int | None,
    query: str,
    output_dir: Path,
    model_name: str,
    cdss_app: Any,
    cdss_node_titles: dict[str, str],
) -> Path:
    return base_workflow.process_input_file(
        input_file=input_file,
        case_limit=case_limit,
        query=query,
        output_dir=output_dir,
        model_name=model_name,
        cdss_app=cdss_app,
        cdss_node_titles=cdss_node_titles,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="串联执行 TNM 与 CDSS 工作流（视觉关键事实输入）")
    parser.add_argument("--input", default="", help="单个输入文件；为空时默认读取 --data-dir 指定目录")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data_image"), help="JSON 数据文件目录（默认: data）")
    parser.add_argument("--all", action="store_true", help="执行 data 目录下全部 benchmark 文件")
    parser.add_argument("--max-files", type=int, default=None, help="最多执行多少个输入 JSON 文件")
    parser.add_argument("--max-cases", type=int, default=None, help="每个输入 JSON 最多执行多少个病例")
    parser.add_argument("--query", default="", help="传给工作流的 query（对应 sys.query）")
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
        default=str(PROJECT_ROOT / "LCAgent/end2end/files/stage1.yml"),
        help="TNM 工作流 YAML 路径",
    )
    parser.add_argument(
        "--cdss-workflow",
        default=str(PROJECT_ROOT / "LCAgent/end2end/files/stage2.yml"),
        help="CDSS 工作流 YAML 路径",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" /"results_end2end/outputs/agent_outputs/agent_image_outputs"),
        help="输出目录",
    )

    parser.add_argument("--mm-max-pages", type=int, default=12, help="视觉提取最多读取 PDF 页数")
    parser.add_argument("--mm-max-chars", type=int, default=12000, help="单病例关键事实最大字符数")
    parser.add_argument("--mm-dpi", type=int, default=200, help="PDF 渲染 DPI")

    args = parser.parse_args()

    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files 必须大于 0")
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases 必须大于 0")
    if args.mm_max_pages <= 0:
        raise ValueError("--mm-max-pages 必须大于 0")
    if args.mm_max_chars <= 0:
        raise ValueError("--mm-max-chars 必须大于 0")
    if args.mm_dpi <= 0:
        raise ValueError("--mm-dpi 必须大于 0")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    config_path = base_workflow.configure_provider_override(config_path, args.provider)

    tnm_workflow_path = Path(args.tnm_workflow)
    cdss_workflow_path = Path(args.cdss_workflow)
    if not tnm_workflow_path.exists():
        raise FileNotFoundError(f"找不到 TNM 工作流文件: {tnm_workflow_path}")
    if not cdss_workflow_path.exists():
        raise FileNotFoundError(f"找不到 CDSS 工作流文件: {cdss_workflow_path}")

    extractor = MultimodalPdfExtractor(
        config_path=config_path,
        provider=args.provider,
        max_pages=args.mm_max_pages,
        max_chars=args.mm_max_chars,
        render_dpi=args.mm_dpi,
    )

    # 让工作流消费视觉抽取的关键事实，而不是全文OCR文本。
    _patch_case_initial_inputs(extractor)
    _patch_pdf_readers(extractor)

    tnm.set_api_config_path(str(config_path))
    tnm.DIFY_WORKFLOW_PATH = tnm_workflow_path
    tnm._load_dify_workflow.cache_clear()
    tnm._dify_nodes_index.cache_clear()

    cdss.set_api_config_path(str(config_path))
    cdss.set_workflow_path(str(cdss_workflow_path))
    cdss_app = cdss.build_cdss_graph()
    cdss_node_titles = base_workflow.load_cdss_node_titles(cdss_workflow_path)

    data_dir = Path(args.data_dir)
    model_name = base_workflow.load_api_model_name(config_path, args.provider)
    input_files = base_workflow.resolve_input_files(args.input, args.all, args.max_files, data_dir)

    for input_file in input_files:
        process_with_multimodal(
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
