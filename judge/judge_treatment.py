from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _has_flag(args: list[str], name: str) -> bool:
    return any(x == name or x.startswith(f"{name}=") for x in args)


def _normalize_code_prefixed_path(value: str) -> str:
    raw = str(value)
    if not raw:
        return raw
    if Path(raw).is_absolute():
        return raw

    norm = raw.replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]

    if norm == "code":
        return "."
    if norm.startswith("code/"):
        return norm[len("code/") :]
    return raw


def _normalize_path_flags(args: list[str], flags: set[str]) -> list[str]:
    out = list(args)
    for i, token in enumerate(out):
        if token in flags and i + 1 < len(out):
            out[i + 1] = _normalize_code_prefixed_path(out[i + 1])
            continue
        for flag in flags:
            prefix = f"{flag}="
            if token.startswith(prefix):
                out[i] = f"{prefix}{_normalize_code_prefixed_path(token[len(prefix):])}"
                break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge 统一入口")
    _, extra = parser.parse_known_args()

    root = Path(__file__).resolve().parents[1]
    target = (root / "code" / "evaluation" / "evaluate_cdss_metrics.py").resolve()
    if not target.exists():
        raise FileNotFoundError(f"目标脚本不存在: {target}")

    args = list(extra)
    if args and args[0] == "--":
        args = args[1:]

    args = _normalize_path_flags(
        args,
        {
            "--api-config",
            "--metrics-dir",
            "--result-root",
            "--pred-dir",
            "--gt-dir",
        },
    )

    if not _has_flag(args, "--api-config"):
        args.extend(["--api-config", "api_config.yaml"])
    if not _has_flag(args, "--metrics-dir") and not _has_flag(args, "--result-root"):
        out_rel = Path("LCAgent") / "outputs" / "judge" / "single"
        (root / out_rel).mkdir(parents=True, exist_ok=True)
        args.extend(["--mode", "single", "--metrics-dir", str(out_rel)])

    cmd = [sys.executable, str(target), *args]
    raise SystemExit(subprocess.run(cmd, cwd=str(root), check=False).returncode)


if __name__ == "__main__":
    main()
