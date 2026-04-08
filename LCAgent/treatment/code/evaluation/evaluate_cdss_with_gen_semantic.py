"""
CDSS 治疗方案推荐质量评测脚本

任务说明：
    给定肺癌患者的TNM分期信息及相关临床特征（病理类型、体力评分、驱动基因、
    PDL1表达等），临床决策支持系统（CDSS）需为患者生成个性化治疗方案推荐。
    本脚本对比 AI 系统输出的 cdss_result 与 GT 的 cdss_result，
    从精确匹配、词级重叠、语义相似度、生成质量四个维度综合评分。

用法：
    python evaluate/cdss_evaluate.py --pred-dir ... --gt-dir ...
"""

import argparse
import ast
import json
import os
import re
import traceback
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


DEFAULT_PRED_DIR = "final_agent_results/gpt-5.2/outputs"
DEFAULT_GT_DIR = "gt/cdss_gt"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_API_CONFIG = str((PROJECT_ROOT / "api_config.yaml").resolve())
DEFAULT_OUTPUT_PATH = str((PROJECT_ROOT / "outputs" / "eval" / "cdss_eval_results.json").resolve())
DEFAULT_MAIN_CONFIG_PATH = str((PROJECT_ROOT / "code" / "workflow" / "workflow_engine.py").resolve())
DEFAULT_API_PY_PATH = str((PROJECT_ROOT / "code" / "inference" / "workflow_batch_infer.py").resolve())
DEFAULT_PRED_FIELD = "cdss_result"
DEFAULT_GT_FIELD = "cdss_result"
DEFAULT_TASK_QUESTION = (
    "给定肺癌患者的临床信息与TNM分期（包括病理类型、体力评分、驱动基因状态、"
    "PDL1表达、既往治疗方案等），临床决策支持系统（CDSS）需为患者生成个性化的"
    "治疗方案推荐，涵盖治疗路径选择、用药建议及相关临床决策依据。"
    "请评估AI生成的治疗建议与标准临床指南推荐的一致性与质量。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CDSS 评测脚本")
    parser.add_argument("--pred-dir", type=str, default=DEFAULT_PRED_DIR, help="AI 预测结果目录，目录下为 json 文件")
    parser.add_argument("--gt-dir", type=str, default=DEFAULT_GT_DIR, help="GT 目录，目录下为 json 文件")
    parser.add_argument(
        "--api-config",
        type=str,
        default=DEFAULT_API_CONFIG,
        help="可选：API 配置文件路径（仅作兜底，不传也可运行）",
    )
    parser.add_argument(
        "--main-config-path",
        type=str,
        default=DEFAULT_MAIN_CONFIG_PATH,
        help="main.py 路径，用于读取 GLOBAL_LLM_CONFIG（模型与采样参数）",
    )
    parser.add_argument(
        "--api-py-path",
        type=str,
        default=DEFAULT_API_PY_PATH,
        help="api.py 路径，用于复用其接口地址路由配置",
    )
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH, help="评测结果输出 json 路径")
    parser.add_argument("--pred-field", type=str, default=DEFAULT_PRED_FIELD, help="预测记录中用于评测的字段名")
    parser.add_argument("--gt-field", type=str, default=DEFAULT_GT_FIELD, help="GT 记录中用于评测的字段名")
    parser.add_argument("--task-question", type=str, default=DEFAULT_TASK_QUESTION, help="传给生成质量评分的任务描述")
    parser.add_argument(
        "--provider",
        type=str,
        default="",
        help="在 api_config.yaml 中选择 provider 名称；为空时按 evaluation.judge_provider / active_provider 自动选择",
    )
    parser.add_argument("--max-files", type=int, default=0, help="最多处理多少个预测文件，0 表示全部")
    parser.add_argument("--max-cases", type=int, default=0, help="每个文件最多处理多少个 case，0 表示全部")
    parser.add_argument("--max-workers", type=int, default=8, help="并发线程数")
    return parser.parse_args()


def _resolve_api_provider(cfg: dict, provider_override: str = "") -> tuple[str, dict]:
    """
    兼容两种格式：
    1) 新格式：providers + evaluation.judge_provider
    2) 旧格式：顶层直接包含 api_key/base_url/model
    """
    providers = cfg.get("providers") if isinstance(cfg, dict) else None
    if isinstance(providers, dict):
        provider_name = provider_override.strip()

        if not provider_name:
            evaluation_value = cfg.get("evaluation")
            eval_cfg = evaluation_value if isinstance(evaluation_value, dict) else {}
            provider_name = str(eval_cfg.get("judge_provider", "")).strip()

        if not provider_name:
            provider_name = str(cfg.get("active_provider", "")).strip()

        if not provider_name and providers:
            provider_name = next(iter(providers.keys()))

        provider = providers.get(provider_name)
        if not isinstance(provider, dict):
            raise KeyError(
                f"Provider not found in api_config.yaml: {provider_name}, available={list(providers.keys())}"
            )
        return provider_name, provider

    if isinstance(cfg, dict):
        return "root", cfg

    raise ValueError("Invalid api_config format")


def _try_load_api_config(api_config_path: str) -> dict:
    path = Path(str(api_config_path or "").strip())
    if not path.exists() or not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_main_global_llm_config(main_config_path: str) -> dict:
    path = Path(main_config_path)
    if not path.exists():
        return {}
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return {}

    for node in tree.body:
        value_node = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GLOBAL_LLM_CONFIG":
                    value_node = node.value
                    break
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "GLOBAL_LLM_CONFIG":
                value_node = node.value

        if value_node is None:
            continue

        try:
            value = ast.literal_eval(value_node)
            if isinstance(value, dict):
                return value
        except Exception:
            return {}
    return {}


def _normalize_completion_params(cfg: dict) -> dict:
    allowed_keys = {
        "max_tokens",
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
    }
    result: dict = {}
    raw = cfg.get("completion_params")
    if not isinstance(raw, dict):
        return result
    for key, value in raw.items():
        if key in allowed_keys and value is not None:
            result[key] = value
    return result


def _load_api_py_endpoints(api_py_path: str) -> dict:
    path = Path(api_py_path)
    if not path.exists():
        return {}
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return {}

    target_names = {"DEFAULT_BASE_URL", "DEFAULT_API_KEY", "GLM_BASE_URL", "GLM_API_KEY"}
    values: dict = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in target_names:
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        if isinstance(value, str) and value.strip():
            values[name] = value.strip()

    return values


def _is_glm_model(model_name: str) -> bool:
    return "glm" in str(model_name).lower()


def _resolve_api_from_api_py(model_name: str, api_py_cfg: dict) -> tuple[str, str, str] | None:
    if not isinstance(api_py_cfg, dict) or not api_py_cfg:
        return None

    if _is_glm_model(model_name):
        base_url = str(api_py_cfg.get("GLM_BASE_URL", "")).strip()
        api_key = str(api_py_cfg.get("GLM_API_KEY", "")).strip()
        if base_url and api_key:
            return base_url, api_key, "api.py:GLM"

    base_url = str(api_py_cfg.get("DEFAULT_BASE_URL", "")).strip()
    api_key = str(api_py_cfg.get("DEFAULT_API_KEY", "")).strip()
    if base_url and api_key:
        return base_url, api_key, "api.py:DEFAULT"
    return None


def _init_eval_runtime(api_config_path: str, provider_override: str, main_config_path: str, api_py_path: str):
    # 优先使用 main.py 的模型配置，再结合 api.py 的接口路由；api_config 仅作兜底。
    api_cfg = _try_load_api_config(api_config_path)
    main_llm_cfg = _load_main_global_llm_config(main_config_path)
    main_provider = str(main_llm_cfg.get("provider", "")).strip() if isinstance(main_llm_cfg, dict) else ""
    effective_provider_override = provider_override.strip() or main_provider

    provider_name = "main+api.py"
    api_key = ""
    base_url = ""
    provider_model = "gpt-4o-mini"
    if api_cfg:
        try:
            provider_name, provider_cfg = _resolve_api_provider(api_cfg, effective_provider_override)
            api_key = str(provider_cfg.get("api_key", "")).strip()
            base_url = str(provider_cfg.get("base_url", "https://api.openai.com/v1")).strip()
            provider_model = str(provider_cfg.get("model", "gpt-4o-mini")).strip()
        except Exception:
            # api_config 非必需，解析失败时仅忽略
            pass

    main_model = str(main_llm_cfg.get("model", "")).strip() if isinstance(main_llm_cfg, dict) else ""
    model = main_model or provider_model
    completion_params = _normalize_completion_params(main_llm_cfg if isinstance(main_llm_cfg, dict) else {})

    # 接口地址路由优先参照 api.py（GLM 与非 GLM 分流）
    endpoint_source = "none"
    api_py_cfg = _load_api_py_endpoints(api_py_path)
    route = _resolve_api_from_api_py(model, api_py_cfg)
    if route is not None:
        base_url, api_key, endpoint_source = route

    # 其次使用环境变量兜底
    if not base_url:
        env_base = str(os.getenv("LLM_BASE_URL", "")).strip()
        if env_base:
            base_url = env_base
            endpoint_source = "env:LLM_BASE_URL"
    if not api_key:
        env_key = str(os.getenv("LLM_API_KEY", "")).strip()
        if env_key:
            api_key = env_key
            endpoint_source = "env:LLM_API_KEY"

    if not api_key or not base_url:
        raise KeyError(
            "无法解析评测接口。请在 api.py 配置 DEFAULT/GLM 接口，或设置 LLM_BASE_URL/LLM_API_KEY，"
            "或传入 --api-config。"
        )

    os.environ["OPENAI_API_KEY"] = api_key
    cfg_path = str(api_config_path or "").strip()
    if cfg_path and Path(cfg_path).exists():
        os.environ["API_CONFIG_PATH"] = cfg_path
    elif "API_CONFIG_PATH" in os.environ:
        del os.environ["API_CONFIG_PATH"]

    # 延迟导入，确保 eval_g 能读取到上面的环境变量。
    from eval import cal_em, cal_f1
    from eval_r import cal_rsim
    import eval_g
    from code.common.model_api import UnifiedOpenAIClient

    evalg_client = UnifiedOpenAIClient(api_key=api_key, base_url=base_url)
    setattr(eval_g, "client", evalg_client)
    setattr(eval_g, "_client", evalg_client)
    setattr(eval_g, "_model", model)
    setattr(eval_g, "_completion_params", completion_params)

    return cal_em, cal_f1, cal_rsim, eval_g, provider_name, model, completion_params, endpoint_source, base_url


def _extract_language_seed(pred_file: Path) -> tuple[str, str]:
    """从预测文件名中提取语言和 seed。"""
    tokens = [tok.strip() for tok in pred_file.stem.split("_") if tok.strip()]
    language = ""
    seed = ""

    for tok in tokens:
        lower = tok.lower()
        if lower in {"chinese", "cn"}:
            language = "Chinese"
        elif lower in {"english", "en"}:
            language = "English"

        match_obj = re.fullmatch(r"seed(\d+)", lower)
        if match_obj:
            seed = match_obj.group(1)
            continue

        # 兼容命名：gt_42_xxx.json / gt-42-xxx.json
        match_obj = re.fullmatch(r"gt[-_]?(\d+)", lower)
        if match_obj:
            seed = match_obj.group(1)
            continue

        # 兼容 token 中包含 seed/gt 信息
        match_obj = re.search(r"(?:seed|gt)[-_]?(\d+)", lower)
        if match_obj:
            seed = match_obj.group(1)

    if not seed:
        raise ValueError(f"无法从预测文件名解析 seed: {pred_file.name}")
    return language, seed


def _resolve_gt_file(pred_file: Path, gt_dir: Path) -> Path:
    """根据预测文件名解析出的 language/seed 找到对应 GT 文件。"""
    language, seed = _extract_language_seed(pred_file)
    if language:
        gt_name = f"benchmark_gt_cdss_{language}_seed{seed}.json"
        gt_file = gt_dir / gt_name
        if gt_file.exists():
            return gt_file

    # language 缺失时，按 seed 回退匹配（优先 Chinese）
    candidates = sorted(gt_dir.glob(f"*seed{seed}.json"))
    if not candidates:
        raise FileNotFoundError(f"未找到对应 seed 的 GT 文件: seed={seed}, gt_dir={gt_dir}")
    for prefer in ("Chinese", "English"):
        preferred = gt_dir / f"benchmark_gt_cdss_{prefer}_seed{seed}.json"
        if preferred in candidates:
            return preferred
    return candidates[0]


def _first_non_empty(record: dict, fields: tuple[str, ...]) -> str:
    for key in fields:
        value = str(record.get(key, "")).strip()
        if value:
            return value
    return ""


def _load_cases_from_file(file_path: Path) -> dict:
    """读取单个 json 文件并返回 {case_id: record}。"""
    with open(file_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    def _normalize_case(case_id: str, raw: dict) -> dict:
        # 不改原文件，仅在内存归一化字段，兼容 treatment/final_answer 评测
        item = dict(raw) if isinstance(raw, dict) else {}
        if case_id and not item.get("case_id"):
            item["case_id"] = case_id
        if not str(item.get("cdss_result", "")).strip():
            item["cdss_result"] = _first_non_empty(item, ("treatment", "final_answer", "cdss_result"))
        return item

    cases_map = {}
    if isinstance(data, dict):
        cases = data.get("cases")
        if isinstance(cases, dict):
            for case_id, value in cases.items():
                if isinstance(value, dict):
                    cid = str(case_id)
                    cases_map[cid] = _normalize_case(cid, value)
        else:
            for key, value in data.items():
                if isinstance(value, dict):
                    cid = str(key)
                    cases_map[cid] = _normalize_case(cid, value)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "case_id" in item:
                cid = str(item["case_id"])
                cases_map[cid] = _normalize_case(cid, item)

    return cases_map


def evaluate_one(
    case_id: str,
    pred_record: dict,
    gt_record: dict,
    pred_field: str,
    gt_field: str,
    task_question: str,
    cal_em,
    cal_f1,
    cal_rsim,
    eval_g,
) -> dict:
    prediction = str(pred_record.get(pred_field, "")).strip()
    reference = str(gt_record.get(gt_field, "")).strip()

    # GT 或预测为空时跳过 G-E，其余指标仍计算
    if not prediction:
        return {
            "case_id": case_id,
            "prediction": prediction,
            "reference": reference,
            "em": 0.0,
            "f1": 0.0,
            "rsim": 0.0,
            "gen": None,
            "skipped": "prediction is empty",
        }

    em_score = cal_em([[reference]], [prediction])
    f1_score = cal_f1([[reference]], [prediction])

    # R-Sim：把 reference 视为“知识”，prediction 视为“检索内容”
    rsim_score = cal_rsim([prediction], [reference]) if reference else 0.0

    gen_result = None
    if reference:
        gen_result = eval_g.cal_gen(
            question=task_question,
            answers=[reference],
            generation=prediction,
            f1_score=f1_score,
        )

    return {
        "case_id": case_id,
        "prediction": prediction,
        "reference": reference,
        "em": round(float(em_score), 4),
        "f1": round(float(f1_score), 4),
        "rsim": round(float(rsim_score), 4),
        "gen": round(float(gen_result["score"]), 4) if gen_result else None,
        "gen_exp": gen_result["explanation"] if gen_result else None,
    }


def main() -> None:
    args = parse_args()

    cal_em, cal_f1, cal_rsim, eval_g, provider_name, model_name, completion_params, endpoint_source, base_url = _init_eval_runtime(
        api_config_path=args.api_config,
        provider_override=args.provider,
        main_config_path=args.main_config_path,
        api_py_path=args.api_py_path,
    )

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    print(f"Using provider: {provider_name}, model: {model_name}")
    print(f"Using endpoint: {endpoint_source}, base_url={base_url}")
    if completion_params:
        print(f"Using completion params from GLOBAL_LLM_CONFIG: {completion_params}")
    print("Loading prediction files ...")

    pred_files = sorted(pred_dir.glob("*.json"))
    if args.max_files > 0:
        pred_files = pred_files[: args.max_files]

    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in: {pred_dir}")

    print(f"Found {len(pred_files)} prediction files")

    results = []
    failed = []
    file_summaries = []

    for pred_file in pred_files:
        gt_file = _resolve_gt_file(pred_file, gt_dir)
        preds = _load_cases_from_file(pred_file)
        gts = _load_cases_from_file(gt_file)

        common_ids = sorted(set(preds) & set(gts))
        if args.max_cases > 0:
            common_ids = common_ids[: args.max_cases]

        print(
            f"Matched file pair: {pred_file.name} <-> {gt_file.name}, "
            f"cases={len(common_ids)}"
        )

        file_results = []
        file_failed = []

        def _eval_wrapper(case_id):
            try:
                record = evaluate_one(
                    case_id=case_id,
                    pred_record=preds[case_id],
                    gt_record=gts[case_id],
                    pred_field=args.pred_field,
                    gt_field=args.gt_field,
                    task_question=args.task_question,
                    cal_em=cal_em,
                    cal_f1=cal_f1,
                    cal_rsim=cal_rsim,
                    eval_g=eval_g,
                )
                record["pred_file"] = pred_file.name
                record["gt_file"] = gt_file.name
                return record
            except Exception:
                traceback.print_exc()
                return {
                    "case_id": case_id,
                    "pred_file": pred_file.name,
                    "gt_file": gt_file.name,
                    "error": traceback.format_exc(),
                }

        # 并发评测（G-E 内部已并发7个维度，这里外层适度并发）
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for res in tqdm(
                executor.map(_eval_wrapper, common_ids),
                total=len(common_ids),
                desc=f"Evaluating {pred_file.name}",
            ):
                if "error" in res:
                    file_failed.append(res)
                else:
                    file_results.append(res)

        results.extend(file_results)
        failed.extend(file_failed)
        file_summaries.append(
            {
                "pred_file": pred_file.name,
                "gt_file": gt_file.name,
                "matched_cases": len(common_ids),
                "evaluated": len(file_results),
                "failed": len(file_failed),
            }
        )

    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    total_cases = sum(item["matched_cases"] for item in file_summaries)

    summary = {
        "provider": provider_name,
        "model": model_name,
        "total_files": len(file_summaries),
        "total_cases": total_cases,
        "evaluated": len(results),
        "failed": len(failed),
        "skipped_no_ref": sum(1 for r in results if not r.get("reference")),
        "overall_em": avg("em"),
        "overall_f1": avg("f1"),
        "overall_rsim": avg("rsim"),
        "overall_gen": avg("gen"),
        "files": file_summaries,
    }

    print("\n=== Evaluation Summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": summary,
        "details": results,
        "failed": failed,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()




