from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def _has_flag(args: list[str], name: str) -> bool:
    return any(x == name or x.startswith(f"{name}=") for x in args)


def _get_flag_value(args: list[str], name: str) -> str:
    prefix = f"{name}="
    for i, token in enumerate(args):
        if token == name and i + 1 < len(args):
            return args[i + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return ""


def _normalize_value(raw: str) -> str:
    text = str(raw or "")
    if not text:
        return text
    if Path(text).is_absolute():
        return text
    norm = text.replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    if norm == "code":
        return "."
    if norm.startswith("code/"):
        return norm[len("code/") :]
    return norm


def _safe_output_stem(path_value: str, default_stem: str) -> str:
    stem = Path(str(path_value or "")).stem.strip()
    if not stem:
        return default_stem
    stem = re.sub(r'[<>:"/\\|?*]+', "_", stem)
    stem = stem.strip(" .")
    return stem or default_stem


def _rewrite_path_for_project(value: str, *, project_root: Path, treatment_root: Path) -> str:
    if not value:
        return value
    if Path(value).is_absolute():
        return value

    norm = _normalize_value(value)
    candidates = [norm]

    for cand in candidates:
        if cand and (project_root / cand).exists():
            return cand

    for cand in candidates:
        if cand and (treatment_root / cand).exists():
            return str(Path("LCAgent") / "treatment" / cand).replace("\\", "/")

    return norm


def _normalize_path_flags(args: list[str], *, project_root: Path, treatment_root: Path, flags: set[str]) -> list[str]:
    out = list(args)
    for i, token in enumerate(out):
        if token in flags and i + 1 < len(out):
            out[i + 1] = _rewrite_path_for_project(out[i + 1], project_root=project_root, treatment_root=treatment_root)
            continue
        for flag in flags:
            prefix = f"{flag}="
            if token.startswith(prefix):
                v = token[len(prefix) :]
                out[i] = f"{prefix}{_rewrite_path_for_project(v, project_root=project_root, treatment_root=treatment_root)}"
                break
    return out


def main() -> None:
    treatment_root = Path(__file__).resolve().parent
    project_root = treatment_root.parents[1]
    target = (treatment_root / "code" / "inference" / "workflow_batch_infer.py").resolve()
    if not target.exists():
        raise FileNotFoundError(f"目标脚本不存在: {target}")

    args = list(sys.argv[1:])
    if args and args[0] == "--":
        args = args[1:]

    args = _normalize_path_flags(
        args,
        project_root=project_root,
        treatment_root=treatment_root,
        flags={"--workflow-yml", "--workflow", "--benchmark-dir", "--case-file", "--output-json"},
    )

    if not _has_flag(args, "--benchmark-dir") and not _has_flag(args, "--case-file"):
        args.extend(["--benchmark-dir", "LC_patient_text"])

    if not _has_flag(args, "--output-json"):
        case_file = _get_flag_value(args, "--case-file")
        if case_file:
            out_name = f"{_safe_output_stem(case_file, 'case')}_agent_results.json"
        else:
            out_name = "agent_results.json"
        out_rel = Path("LCAgent") / "outputs" / out_name
        (project_root / out_rel).parent.mkdir(parents=True, exist_ok=True)
        args.extend(["--output-json", str(out_rel).replace("\\", "/")])

    if not _has_flag(args, "--read-pdf"):
        case_file = _get_flag_value(args, "--case-file")
        bench_dir = _get_flag_value(args, "--benchmark-dir")
        if case_file.lower().endswith(".pdf") or "LC_patient_image" in bench_dir:
            args.append("--read-pdf")

    env = os.environ.copy()
    env["API_CONFIG_PATH"] = str((project_root / "api_config.yaml").resolve())
    env["WORKSPACE_ROOT"] = str(project_root.resolve())

    cmd = [sys.executable, str(target), *args]
    raise SystemExit(subprocess.run(cmd, cwd=str(project_root), env=env, check=False).returncode)


if __name__ == "__main__":
    main()
