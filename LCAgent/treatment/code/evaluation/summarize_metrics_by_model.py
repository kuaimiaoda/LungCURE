from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


# 兼容两种输入目录：
# 1) agent_seed42 / llm_seed42
# 2) agent / llm
# 3) agent-en / llm-en
GROUP_DIR_RE = re.compile(r"^(agent|llm)(-en)?(?:[_-]seed(\d+))?$", re.IGNORECASE)
FILE_RE_STRICT = re.compile(
    r"^gt_(\d+)_(?:(?:en|zh|cn)_)?(.+?)_(?:dedup_)?with_think_treatment_metric\.json$",
    re.IGNORECASE,
)
FILE_RE_LOOSE = re.compile(
    r"^gt_(\d+)_(?:(?:en|zh|cn)_)?(.+?)(?:_.*)?_metric\.json$",
    re.IGNORECASE,
)

EXPERIMENT_GROUPS = ("agent", "llm", "agent-en", "llm-en")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = str((PROJECT_ROOT / "outputs" / "eval" / "result").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "把测评结果按 模型/实验组/seed 归档："
            "result/<model>/<agent|llm>/gt-<seed>/..."
        )
    )
    parser.add_argument(
        "--result-root",
        default=DEFAULT_RESULT_ROOT,
        help="当前测评结果根目录（含 agent_seed42、llm_seed42 等目录）。",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="默认拷贝（推荐）；不传时也拷贝，此参数仅用于显式表达。",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="移动文件到新结构（与 --copy 二选一）。",
    )
    return parser.parse_args()


def parse_file_seed_model(file_name: str) -> tuple[str, str] | None:
    m = FILE_RE_STRICT.match(file_name)
    if m:
        return m.group(1), m.group(2)
    m = FILE_RE_LOOSE.match(file_name)
    if m:
        return m.group(1), m.group(2)
    return None


def parse_group_dir_name(dir_name: str) -> tuple[str, str]:
    """
    Parse source group directory name.
    Returns: (group_name, seed_from_dir)
    group_name in {"agent","llm","agent-en","llm-en"}.
    """
    gm = GROUP_DIR_RE.match(dir_name or "")
    if not gm:
        raise ValueError(f"Unrecognized group directory: {dir_name}")

    base = gm.group(1).lower()
    en_suffix = gm.group(2) or ""
    seed = gm.group(3) or ""
    group = f"{base}{en_suffix.lower()}"
    return group, seed


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _var_population(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _avg(values)
    return sum((x - m) * (x - m) for x in values) / len(values)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _compute_file_metrics(metric_file: Path) -> dict[str, float]:
    payload = _read_json(metric_file)
    if not isinstance(payload, dict):
        return {
            "avg_cdss_accuracy": 0.0,
            "avg_cdss_quality": 0.0,
            "avg_f1": 0.0,
        }

    cases = payload.get("cases")
    if not isinstance(cases, dict):
        return {
            "avg_cdss_accuracy": 0.0,
            "avg_cdss_quality": 0.0,
            "avg_f1": 0.0,
        }

    acc_values: list[float] = []
    quality_values: list[float] = []
    f1_values: list[float] = []

    for case_obj in cases.values():
        if not isinstance(case_obj, dict):
            continue
        acc = _safe_float((case_obj.get("cdss_accuracy") or {}).get("score"))
        if acc is not None:
            acc_values.append(acc)

        quality = _safe_float((case_obj.get("cdss_quality") or {}).get("quality_score"))
        if quality is not None:
            quality_values.append(quality)

        f1 = _safe_float(case_obj.get("treatment_micro_f1"))
        if f1 is not None:
            f1_values.append(f1)

    return {
        "avg_cdss_accuracy": _avg(acc_values),
        "avg_cdss_quality": _avg(quality_values),
        "avg_f1": _avg(f1_values),
    }


def _collect_model_dirs(result_root: Path) -> list[Path]:
    model_dirs: list[Path] = []
    for p in result_root.iterdir():
        if not p.is_dir():
            continue
        if GROUP_DIR_RE.match(p.name):
            continue
        model_dirs.append(p)
    return sorted(model_dirs)


def _summarize_model_group(model_dir: Path, group: str) -> dict[str, Any]:
    group_dir = model_dir / group
    if not group_dir.exists() or not group_dir.is_dir():
        return {
            "bench": {},
            "across_bench": {
                "bench_count": 0,
                "avg_cdss_accuracy_mean": 0.0,
                "avg_cdss_accuracy_var": 0.0,
                "avg_cdss_quality_mean": 0.0,
                "avg_cdss_quality_var": 0.0,
                "avg_f1_mean": 0.0,
                "avg_f1_var": 0.0,
            },
        }

    bench_stats: dict[str, Any] = {}
    for gt_dir in sorted([x for x in group_dir.iterdir() if x.is_dir() and x.name.startswith("gt-")]):
        seed = gt_dir.name.replace("gt-", "")
        metric_files = sorted(gt_dir.glob("*_metric.json"))
        if not metric_files:
            continue

        per_file = [_compute_file_metrics(f) for f in metric_files]
        bench_stats[seed] = {
            "file_count": len(metric_files),
            "avg_cdss_accuracy": _avg([x["avg_cdss_accuracy"] for x in per_file]),
            "avg_cdss_quality": _avg([x["avg_cdss_quality"] for x in per_file]),
            "avg_f1": _avg([x["avg_f1"] for x in per_file]),
            "files": [f.name for f in metric_files],
        }

    acc_bench = [v["avg_cdss_accuracy"] for v in bench_stats.values()]
    quality_bench = [v["avg_cdss_quality"] for v in bench_stats.values()]
    f1_bench = [v["avg_f1"] for v in bench_stats.values()]

    return {
        "bench": bench_stats,
        "across_bench": {
            "bench_count": len(bench_stats),
            "avg_cdss_accuracy_mean": _avg(acc_bench),
            "avg_cdss_accuracy_var": _var_population(acc_bench),
            "avg_cdss_quality_mean": _avg(quality_bench),
            "avg_cdss_quality_var": _var_population(quality_bench),
            "avg_f1_mean": _avg(f1_bench),
            "avg_f1_var": _var_population(f1_bench),
        },
    }


def build_summary(result_root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "result_root": str(result_root),
        "models": {},
    }
    for model_dir in _collect_model_dirs(result_root):
        model_name = model_dir.name
        model_groups: dict[str, Any] = {}
        for group in EXPERIMENT_GROUPS:
            model_groups[group] = _summarize_model_group(model_dir, group)
        summary["models"][model_name] = model_groups
    return summary


def write_summary_files(result_root: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    json_path = result_root / "model_bench_stats_summary.json"
    csv_path = result_root / "model_bench_stats_summary.csv"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "model,group,bench_count,avg_cdss_accuracy_mean,avg_cdss_accuracy_var,avg_cdss_quality_mean,avg_cdss_quality_var,avg_f1_mean,avg_f1_var"
    ]
    models = summary.get("models", {})
    if isinstance(models, dict):
        for model_name, model_info in models.items():
            if not isinstance(model_info, dict):
                continue
            for group in EXPERIMENT_GROUPS:
                group_info = model_info.get(group, {})
                if not isinstance(group_info, dict):
                    continue
                across = group_info.get("across_bench", {})
                if not isinstance(across, dict):
                    continue
                lines.append(
                    ",".join(
                        [
                            str(model_name),
                            group,
                            str(across.get("bench_count", 0)),
                            str(across.get("avg_cdss_accuracy_mean", 0.0)),
                            str(across.get("avg_cdss_accuracy_var", 0.0)),
                            str(across.get("avg_cdss_quality_mean", 0.0)),
                            str(across.get("avg_cdss_quality_var", 0.0)),
                            str(across.get("avg_f1_mean", 0.0)),
                            str(across.get("avg_f1_var", 0.0)),
                        ]
                    )
                )

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).resolve()
    if not result_root.exists() or not result_root.is_dir():
        raise FileNotFoundError(f"结果目录不存在: {result_root}")
    if args.copy and args.move:
        raise ValueError("--copy 与 --move 不能同时使用")

    use_move = bool(args.move)
    action_name = "MOVE" if use_move else "COPY"

    total = 0
    matched = 0
    for sub in sorted([p for p in result_root.iterdir() if p.is_dir()]):
        if not GROUP_DIR_RE.match(sub.name):
            continue
        group, seed_from_dir = parse_group_dir_name(sub.name)

        for src_file in sorted(sub.glob("*_metric.json")):
            total += 1
            parsed = parse_file_seed_model(src_file.name)
            if parsed is None:
                print(f"[SKIP] 无法解析模型名: {src_file.name}")
                continue

            seed_from_file, model = parsed
            seed = seed_from_file or seed_from_dir
            model_dir = result_root / model / group / f"gt-{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            dst_file = model_dir / src_file.name

            if use_move:
                shutil.move(str(src_file), str(dst_file))
            else:
                shutil.copy2(src_file, dst_file)

            matched += 1
            print(f"[{action_name}] {src_file} -> {dst_file}")

    print("=" * 72)
    print(f"扫描 metric 文件: {total}")
    print(f"成功归档: {matched}")
    summary = build_summary(result_root)
    json_path, csv_path = write_summary_files(result_root, summary)
    print(f"统计汇总(JSON): {json_path}")
    print(f"统计汇总(CSV): {csv_path}")
    print(f"结果根目录: {result_root}")


if __name__ == "__main__":
    main()
