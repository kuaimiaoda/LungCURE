from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable


def resolve_benchmark_dir(path_arg: str, *, root_dir: Path | None = None) -> Path:
    p = Path(str(path_arg or "").strip())
    if p.is_absolute() and p.exists():
        return p.resolve()
    if p.exists():
        return p.resolve()
    if root_dir is not None:
        candidate = (Path(root_dir) / p).resolve()
        if candidate.exists():
            return candidate
    return p.resolve()


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        rp = p.resolve()
        key = str(rp).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(rp)
    return uniq


def discover_pdf_case_files(base_dir: Path) -> list[Path]:
    files: list[Path] = []

    files.extend(sorted(base_dir.glob("*.pdf")))

    preferred_dirs = [
        "seed_42pdf",
        "seed42pdf",
        "seed_2024pdf",
        "seed2024pdf",
        "seed_3407pdf",
        "seed3407pdf",
    ]
    for name in preferred_dirs:
        folder = base_dir / name
        if folder.exists() and folder.is_dir():
            files.extend(sorted(folder.glob("*.pdf")))

    for folder in sorted(base_dir.glob("seed*pdf")):
        if folder.is_dir():
            files.extend(sorted(folder.glob("*.pdf")))

    return _dedupe_paths(files)


def discover_text_case_files(base_dir: Path, *, suffixes: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    normalized = tuple(x.lower().lstrip(".") for x in suffixes)
    for suf in normalized:
        files.extend(sorted(base_dir.glob(f"*.{suf}")))
    return _dedupe_paths(files)


def iter_case_files(
    case_file: str | None,
    benchmark_dir: str,
    limit: int,
    random_sample: int,
    random_seed: int,
    *,
    read_pdf: bool,
    root_dir: Path | None = None,
    default_suffixes: tuple[str, ...] = ("md", "txt", "json"),
) -> list[Path]:
    if case_file:
        files = [Path(case_file).resolve()]
    else:
        base = resolve_benchmark_dir(benchmark_dir, root_dir=root_dir)
        text_files = discover_text_case_files(base, suffixes=default_suffixes)
        pdf_files = discover_pdf_case_files(base) if read_pdf else []

        # read_pdf=True 时支持同目录混合输入；否则只读文本类输入。
        files = _dedupe_paths([*text_files, *pdf_files])

        if random_sample > 0:
            k = min(random_sample, len(files))
            rng = random.Random(random_seed)
            files = rng.sample(files, k)

    if limit > 0:
        files = files[:limit]
    return files


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺少 pdfminer.six，请先安装：pip install pdfminer.six") from exc

    text = pdf_extract_text(str(pdf_path))
    text = str(text or "").replace("\x0c", "\n").strip()
    if not text:
        raise ValueError(f"PDF 提取结果为空: {pdf_path}")
    return text


def _read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def load_case_payload(case_path: Path, tnm_by_case_id: dict[str, str] | None = None) -> dict[str, Any]:
    suffix = case_path.suffix.lower()
    if suffix == ".pdf":
        case_id = case_path.stem
        tnm_ret = (tnm_by_case_id or {}).get(case_id, "")
        query = extract_pdf_text(case_path)
        return {
            "case_file": case_path.name,
            "case_id": case_id,
            "tnm_ret": tnm_ret,
            "query": query,
        }

    raw = _read_text_file(case_path)
    case_id = case_path.stem
    tnm_ret = ""
    query = raw

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = None

    if isinstance(obj, dict):
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
        inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}

        case_id = str(
            payload.get("user")
            or payload.get("case_id")
            or payload.get("id")
            or case_path.stem
        ).strip()
        tnm_ret = str(
            inputs.get("TNM_ret")
            or payload.get("tnm_ret")
            or payload.get("TNM_ret")
            or payload.get("TNM")
            or ""
        ).strip()
        query = str(
            payload.get("query")
            or payload.get("document_text")
            or payload.get("text")
            or raw
        )

    if not tnm_ret and tnm_by_case_id:
        tnm_ret = str(tnm_by_case_id.get(case_id, "")).strip()

    return {
        "case_file": case_path.name,
        "case_id": case_id,
        "tnm_ret": tnm_ret,
        "query": query,
    }


def build_case_tnm_map(benchmark_dir: str, *, root_dir: Path | None = None) -> dict[str, str]:
    base = resolve_benchmark_dir(benchmark_dir, root_dir=root_dir)
    roots = [base]
    if base.parent != base:
        roots.append(base.parent)

    tnm_by_case_id: dict[str, str] = {}
    visited: set[str] = set()
    for root in roots:
        key = str(root.resolve()).lower()
        if key in visited:
            continue
        visited.add(key)

        for path in discover_text_case_files(root, suffixes=("md", "json", "txt")):
            try:
                raw = _read_text_file(path)
                obj = json.loads(raw)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
            inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
            case_id = str(payload.get("user") or payload.get("case_id") or payload.get("id") or "").strip()
            tnm_ret = str(
                inputs.get("TNM_ret")
                or payload.get("tnm_ret")
                or payload.get("TNM_ret")
                or payload.get("TNM")
                or ""
            ).strip()
            if case_id and tnm_ret and case_id not in tnm_by_case_id:
                tnm_by_case_id[case_id] = tnm_ret

    return tnm_by_case_id
