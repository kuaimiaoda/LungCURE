from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
JUDGE_METRICS_PATH = SCRIPT_DIR / "evaluate_cdss_metrics.py"


def _load_judge_metrics_module():
    if not JUDGE_METRICS_PATH.exists():
        raise FileNotFoundError(f"evaluate_cdss_metrics.py not found: {JUDGE_METRICS_PATH}")

    spec = importlib.util.spec_from_file_location("judge_cdss_metrics", JUDGE_METRICS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec: {JUDGE_METRICS_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def f1score(case_file: str | Path, gt_file: str | Path, *, case_id: str | None = None) -> dict[str, Any]:
    """
    Pure wrapper that calls functions from evaluate_cdss_metrics.py.
    """
    mod = _load_judge_metrics_module()

    if hasattr(mod, "f1score"):
        return mod.f1score(case_file=case_file, gt_file=gt_file, case_id=case_id)

    if hasattr(mod, "compute_case_final_score_from_case_and_gt_files"):
        return mod.compute_case_final_score_from_case_and_gt_files(
            case_file=case_file,
            gt_file=gt_file,
            case_id=case_id,
        )

    raise AttributeError(
        "evaluate_cdss_metrics.py missing required function: "
        "f1score or compute_case_final_score_from_case_and_gt_files"
    )


def _safe_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def _infer_lang_for_seed(mod: Any, pred_file: Path, pred_payload: Any) -> str:
    # 1) payload language if provided
    payload_lang = ""
    if hasattr(mod, "_extract_payload_language"):
        payload_lang = str(mod._extract_payload_language(pred_payload) or "")
    if payload_lang and hasattr(mod, "_normalize_language_to_gt"):
        norm = mod._normalize_language_to_gt(payload_lang)
        if norm in {"Chinese", "English"}:
            return norm

    # 2) filename hint
    low_name = pred_file.name.lower()
    if "english" in low_name:
        return "English"
    if "chinese" in low_name:
        return "Chinese"

    # 3) heuristic from predicted cdss text
    pred_cases = mod._extract_pred_cases(pred_payload) if hasattr(mod, "_extract_pred_cases") else {}
    if isinstance(pred_cases, dict):
        sampled_texts: list[str] = []
        for _, item in list(pred_cases.items())[:8]:
            if isinstance(item, dict):
                text = (
                    str(item.get("cdss_result") or "")
                    or str(item.get("final_answer") or "")
                    or str(item.get("treatment") or "")
                )
                if text:
                    sampled_texts.append(text)
        if sampled_texts:
            cjk_hits = sum(1 for t in sampled_texts if _contains_cjk(t))
            return "Chinese" if cjk_hits >= max(1, len(sampled_texts) // 2) else "English"

    # default
    return "Chinese"


def _iter_seed_json_files(input_dir: Path, benchseed: str) -> list[Path]:
    files = [p for p in sorted(input_dir.glob("*.json")) if p.is_file()]
    seed_text = str(benchseed).strip()
    if not seed_text:
        raise ValueError("benchseed 不能为空")
    pattern = re.compile(
        rf"(?:^|[_-])gt[_-]?{re.escape(seed_text)}(?:[_-]|\.|$)|seed[_-]?{re.escape(seed_text)}(?:[_-]|\.|$)",
        re.IGNORECASE,
    )
    return [p for p in files if pattern.search(p.name)]


def _extract_seed_and_model_from_pred_file(pred_file: Path) -> tuple[str, str]:
    stem = pred_file.stem
    lower = stem.lower()

    benchseed = ""
    seed_patterns = [
        re.compile(r"seed[_-]?(\d+)", re.IGNORECASE),
        re.compile(r"(?:^|[_-])gt[_-]?(\d+)(?:[_-]|$)", re.IGNORECASE),
    ]
    for pat in seed_patterns:
        match = pat.search(stem)
        if match:
            benchseed = str(match.group(1))
            break

    model_text = stem
    for suffix in ("_dedup_with_think_treatment", "_with_think_treatment", "_dedup"):
        if lower.endswith(suffix):
            model_text = model_text[: -len(suffix)]
            break

    if benchseed:
        prefix_patterns = [
            re.compile(rf"^gt[_-]?{re.escape(benchseed)}[_-]?(?:en|english|cn|chinese)?[_-]?", re.IGNORECASE),
            re.compile(rf"^seed[_-]?{re.escape(benchseed)}[_-]?", re.IGNORECASE),
        ]
        for pat in prefix_patterns:
            model_text = pat.sub("", model_text, count=1)
    else:
        model_text = re.sub(r"^gt[_-]?\d+[_-]?(?:en|english|cn|chinese)?[_-]?", "", model_text, count=1)

    model_text = model_text.strip("_- ")
    if not model_text:
        model_text = stem
    if not benchseed:
        benchseed = "unknown"
    return benchseed, model_text


def _group_label(group_name: str) -> str:
    name = str(group_name).lower()
    if name.startswith("agent"):
        return "agent"
    if name.startswith("llm"):
        return "llm"
    return str(group_name)


def _write_summary_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "benchseed", "f1score", "group"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": str(row.get("model", "")),
                    "benchseed": str(row.get("benchseed", "")),
                    "f1score": float(row.get("f1score", 0.0)),
                    "group": str(row.get("group", "")),
                }
            )


def _evaluate_one_pred_file(mod: Any, pred_file: Path, gt_file: Path) -> dict[str, Any]:
    pred_payload = _safe_read_json(pred_file)
    gt_payload = _safe_read_json(gt_file)

    pred_cases = mod._extract_pred_cases(pred_payload) if hasattr(mod, "_extract_pred_cases") else {}
    gt_cases = mod._extract_cases(gt_payload) if hasattr(mod, "_extract_cases") else {}
    if not isinstance(pred_cases, dict):
        pred_cases = {}
    if not isinstance(gt_cases, dict):
        gt_cases = {}

    shared_case_ids = [cid for cid in pred_cases.keys() if cid in gt_cases]
    if not shared_case_ids:
        shared_case_ids = list(pred_cases.keys())

    case_results: dict[str, Any] = {}
    f1_values: list[float] = []
    errors = 0

    for cid in shared_case_ids:
        try:
            res = mod.compute_case_final_score_from_case_and_gt_files(
                case_file=str(pred_file),
                gt_file=str(gt_file),
                case_id=cid,
            )
            f1 = float(res.get("f1_score", res.get("final_score", 0.0)) or 0.0)
            f1_values.append(f1)
            case_results[cid] = {
                "f1_score": f1,
                "final_score": float(res.get("final_score", 0.0) or 0.0),
                "base_score": float((res.get("score_info") or {}).get("base_score", 0.0) or 0.0),
                "bonus": float((res.get("score_info") or {}).get("bonus", 0.0) or 0.0),
                "penalty": float((res.get("score_info") or {}).get("penalty", 0.0) or 0.0),
            }
        except Exception as exc:  # pragma: no cover
            errors += 1
            case_results[cid] = {"error": f"{type(exc).__name__}: {exc}"}

    avg_f1 = (sum(f1_values) / len(f1_values)) if f1_values else 0.0
    return {
        "pred_file": str(pred_file),
        "gt_file": str(gt_file),
        "case_total": len(shared_case_ids),
        "case_success": len(f1_values),
        "case_error": errors,
        "avg_f1_score": avg_f1,
        "cases": case_results,
    }


def run_seed_batch(
    *,
    benchseed: str,
    agent_dir: str | Path,
    llm_dir: str | Path,
    final_gt_dir: str | Path,
    output_root: str | Path,
    force_gt_file: str | Path | None = None,
) -> dict[str, Any]:
    mod = _load_judge_metrics_module()

    agent_path = Path(agent_dir).resolve()
    llm_path = Path(llm_dir).resolve()
    gt_root = Path(final_gt_dir).resolve()
    out_root = Path(output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not agent_path.exists():
        raise FileNotFoundError(f"agent目录不存在: {agent_path}")
    if not llm_path.exists():
        raise FileNotFoundError(f"llm目录不存在: {llm_path}")
    if not gt_root.exists():
        raise FileNotFoundError(f"final_gt目录不存在: {gt_root}")

    seed_text = str(benchseed).strip()
    if not seed_text:
        raise ValueError("benchseed 不能为空")

    fixed_gt_path: Path | None = None
    if force_gt_file:
        fixed_gt_path = Path(force_gt_file).resolve()
        if not fixed_gt_path.exists() or not fixed_gt_path.is_file():
            raise FileNotFoundError(f"--force-gt-file 不存在或不是文件: {fixed_gt_path}")

    groups = [
        (f"agent_processed_gt{seed_text}", agent_path),
        (f"llm_llm_gt{seed_text}", llm_path),
    ]
    overall_records: dict[str, Any] = {}
    all_summary_csv_rows: list[dict[str, Any]] = []

    for group_name, src_dir in groups:
        group_out_dir = out_root / group_name
        group_out_dir.mkdir(parents=True, exist_ok=True)

        pred_files = _iter_seed_json_files(src_dir, seed_text)
        group_file_records: list[dict[str, Any]] = []
        group_avg_pool: list[float] = []
        group_summary_csv_rows: list[dict[str, Any]] = []

        for pred_file in pred_files:
            pred_payload = _safe_read_json(pred_file)
            if fixed_gt_path is not None:
                gt_file = fixed_gt_path
            else:
                lang = _infer_lang_for_seed(mod, pred_file, pred_payload)
                gt_file = gt_root / f"benchmark_gt_cdss_{lang}_seed{seed_text}.json"
                if not gt_file.exists():
                    # fallback to Chinese seed if inferred file missing
                    gt_file = gt_root / f"benchmark_gt_cdss_Chinese_seed{seed_text}.json"

            result = _evaluate_one_pred_file(mod, pred_file, gt_file)
            out_file = group_out_dir / f"{pred_file.stem}_f1score.json"
            out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            benchseed, model_name = _extract_seed_and_model_from_pred_file(pred_file)
            summary_csv_row = {
                "model": model_name,
                "benchseed": benchseed,
                "f1score": float(result["avg_f1_score"]),
                "group": _group_label(group_name),
            }
            group_summary_csv_rows.append(summary_csv_row)
            all_summary_csv_rows.append(summary_csv_row)

            group_file_records.append(
                {
                    "pred_file": pred_file.name,
                    "gt_file": Path(result["gt_file"]).name,
                    "case_total": int(result["case_total"]),
                    "case_success": int(result["case_success"]),
                    "case_error": int(result["case_error"]),
                    "avg_f1_score": float(result["avg_f1_score"]),
                    "output_file": str(out_file),
                }
            )
            group_avg_pool.append(float(result["avg_f1_score"]))

        group_summary_csv_path = group_out_dir / "summary.csv"
        _write_summary_csv(group_summary_csv_path, group_summary_csv_rows)

        group_summary = {
            "group_name": group_name,
            "source_dir": str(src_dir),
            "output_dir": str(group_out_dir),
            "file_count": len(group_file_records),
            "avg_f1_score_over_files": (sum(group_avg_pool) / len(group_avg_pool)) if group_avg_pool else 0.0,
            "summary_csv": str(group_summary_csv_path),
            "files": group_file_records,
        }
        overall_records[group_name] = group_summary

    overall_summary_csv_path = out_root / f"summary_gt{seed_text}.csv"
    _write_summary_csv(overall_summary_csv_path, all_summary_csv_rows)

    overall_summary = {
        "benchseed": seed_text,
        "output_root": str(out_root),
        "summary_csv": str(overall_summary_csv_path),
        "groups": overall_records,
    }
    return overall_summary


def run_gt42_batch(
    *,
    agent_dir: str | Path,
    llm_dir: str | Path,
    final_gt_dir: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    # Backward-compatible wrapper.
    return run_seed_batch(
        benchseed="42",
        agent_dir=agent_dir,
        llm_dir=llm_dir,
        final_gt_dir=final_gt_dir,
        output_root=output_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute final_score/F1 by calling judge_cdss_metrics functions.")
    parser.add_argument("--case-file", default=None, help="Case JSON file path.")
    parser.add_argument("--gt-file", default=None, help="GT JSON file path.")
    parser.add_argument("--case-id", default=None, help="Optional case_id. If omitted, auto-select one.")
    parser.add_argument("--output-json", default=None, help="Optional output JSON file path.")
    parser.add_argument("--batch-gt42", action="store_true", help="Run batch evaluation for seed42 files (legacy flag).")
    parser.add_argument("--batch-gt2024", action="store_true", help="Run batch evaluation for seed2024 files.")
    parser.add_argument("--batch-gt3407", action="store_true", help="Run batch evaluation for seed3407 files.")
    parser.add_argument(
        "--batch-seed",
        default="",
        help="Run batch evaluation for a specific bench seed (e.g. 42, 2024, 3407).",
    )
    parser.add_argument("--agent-dir", default=str(PROJECT_ROOT / "agent" / "processed"), help="agent processed directory.")
    parser.add_argument("--llm-dir", default=str(PROJECT_ROOT / "llm" / "llm"), help="llm result directory.")
    parser.add_argument("--final-gt-dir", default=str(PROJECT_ROOT / "final_gt"), help="final GT directory.")
    parser.add_argument(
        "--force-gt-file",
        default="",
        help="Optional fixed GT JSON file for batch mode. If set, use this file for all predictions.",
    )
    parser.add_argument("--batch-output-root", default=str(SCRIPT_DIR / "f1score"), help="Batch output root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_seed = str(args.batch_seed or "").strip()
    if not batch_seed:
        if bool(args.batch_gt2024):
            batch_seed = "2024"
        elif bool(args.batch_gt3407):
            batch_seed = "3407"
        elif bool(args.batch_gt42):
            batch_seed = "42"

    if batch_seed:
        summary = run_seed_batch(
            benchseed=batch_seed,
            agent_dir=args.agent_dir,
            llm_dir=args.llm_dir,
            final_gt_dir=args.final_gt_dir,
            output_root=args.batch_output_root,
            force_gt_file=(args.force_gt_file or None),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if not args.case_file or not args.gt_file:
        raise ValueError(
            "单文件模式需要同时提供 --case-file 和 --gt-file，或使用 "
            "--batch-seed/--batch-gt42/--batch-gt2024/--batch-gt3407。"
        )

    result = f1score(case_file=args.case_file, gt_file=args.gt_file, case_id=args.case_id)

    if args.output_json:
        out = Path(args.output_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(out))
        return

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
