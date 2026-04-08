from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
try:
    yaml = importlib.import_module("yaml")
except Exception as exc:
    raise RuntimeError("缺少 pyyaml 依赖，请先安装 pyyaml") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.common.model_api import chat_content_with_retry


@dataclass
class ProviderConfig:
    name: str
    model: str
    base_url: str
    api_key: str
    timeout: int = 300
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    temperature: float = 0.1
    max_tokens: int = 4096


class LLMClient:
    def __init__(self, provider: ProviderConfig):
        self.provider = provider
        self.chat_url = self._build_chat_url(provider.base_url)

    @staticmethod
    def _build_chat_url(base_url: str) -> str:
        trimmed = _safe_str(base_url).rstrip("/")
        if not trimmed:
            raise ValueError("provider.base_url 不能为空")
        if trimmed.endswith("/chat/completions"):
            return trimmed
        return f"{trimmed}/chat/completions"

    @staticmethod
    def _extract_content(payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if not isinstance(message, dict):
            return ""
        return _safe_str(message.get("content"))

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return chat_content_with_retry(
            api_key=self.provider.api_key,
            base_url=self.provider.base_url,
            model=self.provider.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(self.provider.temperature),
            max_tokens=int(self.provider.max_tokens),
            timeout=int(self.provider.timeout),
            max_retries=int(self.provider.max_retries),
            retry_backoff_seconds=float(self.provider.retry_backoff_seconds),
            alias_retry=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 AI CDSS 输出与 GT 的一致性与质量")
    parser.add_argument(
        "--mode",
        choices=["single", "paired"],
        default="paired",
        help="single: 单实验目录评测；paired: 同时评测两组实验（默认）。",
    )
    parser.add_argument(
        "--pred-dir",
        default=str(PROJECT_ROOT / "agent" / "processed"),
        help="single 模式下的预测结果目录。",
    )
    parser.add_argument(
        "--gt-dir",
        default=str(PROJECT_ROOT / "final_gt"),
        help="医生 GT 目录（benchmark_gt_cdss_*.json）。",
    )
    parser.add_argument(
        "--api-config",
        default=str(PROJECT_ROOT /"api_config.yaml"),
        help="裁判模型配置文件路径",
    )
    parser.add_argument(
        "--metrics-dir",
        default=str((Path(__file__).resolve().parent / "result" / "single")),
        help="single 模式输出目录（兼容旧参数）。",
    )
    parser.add_argument(
        "--result-root",
        default=str(Path(__file__).resolve().parent / "result"),
        help="paired 模式输出根目录（会写到 result/<实验组名>）。",
    )
    parser.add_argument(
        "--exp1-name",
        default="agent",
        help="paired 模式实验组1名称。",
    )
    parser.add_argument(
        "--exp1-pred-dir",
        default=str(PROJECT_ROOT / "agent" / "processed"),
        help="paired 模式实验组1预测目录。",
    )
    parser.add_argument(
        "--exp2-name",
        default="llm",
        help="paired 模式实验组2名称。",
    )
    parser.add_argument(
        "--exp2-pred-dir",
        default=str(PROJECT_ROOT / "llm" / "processed"),
        help="paired 模式实验组2预测目录。",
    )
    parser.add_argument(
        "--default-language",
        default="Chinese",
        help="当预测文件未提供 language 时使用的默认语言（Chinese/English）。",
    )
    parser.add_argument(
        "--include",
        default="*dedup_with_think_treatment*.json",
        help="输入文件匹配规则。",
    )
    parser.add_argument("--max-files", type=int, default=0, help="最多处理多少个文件，0 表示全部")
    parser.add_argument("--max-cases", type=int, default=0, help="每个文件最多处理多少病例，0 表示全部")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "并行进程数。单文件评测时用于病例并行；多文件评测时用于文件并行。"
            "0 表示自动，1 表示串行。"
        ),
    )
    return parser.parse_args()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_json_object(text: str) -> dict[str, Any]:
    text = _safe_str(text)
    if not text:
        return {}

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1)

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _contains_cjk_chars(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", _safe_str(text)))


_ASCII_SURFACE_PATTERN_CACHE: dict[str, re.Pattern] = {}


def _compile_ascii_surface_pattern(surface: str) -> re.Pattern:
    cached = _ASCII_SURFACE_PATTERN_CACHE.get(surface)
    if cached is not None:
        return cached

    escaped = re.escape(surface)
    # Allow flexible space/hyphen usage for English phrases.
    escaped = escaped.replace(r"\ ", r"\s+").replace(r"\-", r"[-\s]?")
    pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    _ASCII_SURFACE_PATTERN_CACHE[surface] = pattern
    return pattern


def _surface_in_text(text: str, surface: str) -> bool:
    text = _safe_str(text)
    surface = _safe_str(surface)
    if not text or not surface:
        return False
    if _contains_cjk_chars(surface):
        return surface in text
    return bool(_compile_ascii_surface_pattern(surface).search(text))


def _extract_embedded_answer_fields(raw_text: Any) -> tuple[str, str]:
    parsed = _extract_json_object(_safe_str(raw_text))
    if not parsed:
        return "", ""
    cdss_result = _safe_str(parsed.get("cdss_result") or parsed.get("treatment") or parsed.get("final_answer"))
    stage_result = _safe_str(parsed.get("stage_result") or parsed.get("stage"))
    return cdss_result, stage_result


def _to_score_0_5(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(5, score))


def _to_binary_0_1(value: Any) -> int:
    try:
        val = int(round(float(value)))
    except (TypeError, ValueError):
        val = 0
    return 1 if val >= 1 else 0


def _mean_no_round(values: list[float]) -> float:
    clean = [float(v) for v in values]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg_or_zero(sum_value: float, count_value: int) -> float:
    if count_value <= 0:
        return 0.0
    return sum_value / count_value


def _default_treatment_tags() -> dict[str, dict[str, int]]:
    return {
        "surgery": {"ai": 0, "expert": 0},
        "chemo": {"ai": 0, "expert": 0},
        "radiotherapy": {"ai": 0, "expert": 0},
        "targeted": {"ai": 0, "expert": 0},
        "immuno": {"ai": 0, "expert": 0},
        "observation": {"ai": 0, "expert": 0},
    }


def _normalize_treatment_tags(raw_obj: dict[str, Any]) -> dict[str, dict[str, int]]:
    tags = _default_treatment_tags()
    alias = {
        "s": "surgery",
        "surgery": "surgery",
        "surgical": "surgery",
        "c": "chemo",
        "chemo": "chemo",
        "chemotherapy": "chemo",
        "r": "radiotherapy",
        "rt": "radiotherapy",
        "radiotherapy": "radiotherapy",
        "t": "targeted",
        "targeted": "targeted",
        "targeted_therapy": "targeted",
        "i": "immuno",
        "immuno": "immuno",
        "immunotherapy": "immuno",
        "o": "observation",
        "observation": "observation",
        "follow_up": "observation",
        "followup": "observation",
    }

    for key, value in raw_obj.items():
        norm_key = alias.get(_safe_str(key).strip().lower())
        if not norm_key or not isinstance(value, dict):
            continue
        tags[norm_key]["ai"] = _to_binary_0_1(value.get("ai"))
        tags[norm_key]["expert"] = _to_binary_0_1(value.get("expert"))
    return tags


def _calc_micro_f1_from_tags(tags: dict[str, dict[str, int]]) -> dict[str, Any]:
    tp = 0
    fp = 0
    fn = 0
    for item in tags.values():
        ai = _to_binary_0_1(item.get("ai"))
        expert = _to_binary_0_1(item.get("expert"))
        if ai == 1 and expert == 1:
            tp += 1
        elif ai == 1 and expert == 0:
            fp += 1
        elif ai == 0 and expert == 1:
            fn += 1

    if tp == 0 and fp == 0 and fn == 0:
        micro_f1 = 1.0
    else:
        micro_f1 = (2.0 * tp) / (2.0 * tp + fp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_f1": micro_f1,
    }


def _compute_cases_metric_stats(cases: dict[str, Any]) -> dict[str, Any]:
    sum_cdss_accuracy = 0.0
    cnt_cdss_accuracy = 0
    sum_cdss_quality = 0.0
    cnt_cdss_quality = 0
    sum_f1 = 0.0
    cnt_f1 = 0

    for case_obj in cases.values():
        if not isinstance(case_obj, dict):
            continue

        acc = _safe_float(((case_obj.get("cdss_accuracy") or {}).get("score")))
        if acc is not None:
            sum_cdss_accuracy += acc
            cnt_cdss_accuracy += 1

        quality = _safe_float(((case_obj.get("cdss_quality") or {}).get("quality_score")))
        if quality is not None:
            sum_cdss_quality += quality
            cnt_cdss_quality += 1

        f1 = _safe_float(case_obj.get("treatment_micro_f1"))
        if f1 is not None:
            sum_f1 += f1
            cnt_f1 += 1

    return {
        "sum_cdss_accuracy": sum_cdss_accuracy,
        "cnt_cdss_accuracy": cnt_cdss_accuracy,
        "sum_cdss_quality": sum_cdss_quality,
        "cnt_cdss_quality": cnt_cdss_quality,
        "sum_f1": sum_f1,
        "cnt_f1": cnt_f1,
        "avg_cdss_accuracy": _avg_or_zero(sum_cdss_accuracy, cnt_cdss_accuracy),
        "avg_cdss_quality": _avg_or_zero(sum_cdss_quality, cnt_cdss_quality),
        "avg_f1": _avg_or_zero(sum_f1, cnt_f1),
    }


def _load_judge_provider(api_config_path: Path) -> ProviderConfig:
    data = yaml.safe_load(api_config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("api_config.yaml 格式错误：根节点必须是对象")

    providers = data.get("providers")
    if not isinstance(providers, dict):
        raise ValueError("api_config.yaml 缺少 providers")

    judge_name = _safe_str((data.get("evaluation") or {}).get("judge_provider")) or "judge_model"
    conf = providers.get(judge_name)
    if not isinstance(conf, dict):
        raise ValueError(f"api_config.yaml 缺少裁判 provider: {judge_name}")

    provider = ProviderConfig(
        name=judge_name,
        model=_safe_str(conf.get("model")),
        base_url=_safe_str(conf.get("base_url")),
        api_key=_safe_str(conf.get("api_key")),
        timeout=int(conf.get("timeout", 300)),
        max_retries=int(conf.get("max_retries", 3)),
        retry_backoff_seconds=float(conf.get("retry_backoff_seconds", 2)),
        temperature=float(conf.get("temperature", 0.1)),
        max_tokens=int(conf.get("max_tokens", 4096)),
    )
    if not provider.api_key:
        raise ValueError("裁判模型 API Key 为空")
    if not provider.model:
        raise ValueError("裁判模型 model 为空")
    return provider


def _extract_cases(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    cases = payload.get("cases")
    if isinstance(cases, dict):
        return cases
    return payload


def _extract_payload_language(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    return _safe_str(payload.get("language"))


def _extract_payload_source_input(payload: Any, fallback_file_name: str = "") -> str:
    if isinstance(payload, dict):
        source_input = _safe_str(payload.get("source_input"))
        if source_input:
            return source_input
    return _safe_str(fallback_file_name)


def _parse_tnm_from_text(raw_text: Any) -> dict[str, str]:
    text = _safe_str(raw_text).replace("\u3000", " ")
    if not text:
        return {"T": "", "N": "", "M": "", "Final_TNM": ""}

    # 兼容 "T1b N0 M0"、"T1bN0M0"、"T1b/N0/M0" 等格式
    match_map: dict[str, str] = {}
    for stage, value in re.findall(r"\b([TNMtnm])\s*([0-9A-Za-zxX]+)\b", text):
        key = stage.upper()
        match_map[key] = f"{key}{value}"

    t_val = match_map.get("T", "")
    n_val = match_map.get("N", "")
    m_val = match_map.get("M", "")
    final = " ".join(x for x in [t_val, n_val, m_val] if x).strip()
    return {"T": t_val, "N": n_val, "M": m_val, "Final_TNM": final}


def _normalize_feature_capture_flag(raw_value: Any) -> str:
    text = _safe_str(raw_value).strip().lower()
    if not text:
        return ""
    if text in {"是", "yes", "y", "true", "1"}:
        return "是"
    if text in {"否", "no", "n", "false", "0"}:
        return "否"
    return ""


def _normalize_pred_case(case_key: str, case_item: Any) -> tuple[str, dict[str, Any]]:
    item = case_item if isinstance(case_item, dict) else {}
    case_id = (
        _safe_str(item.get("case_id"))
        or _safe_str(item.get("id"))
        or _safe_str(item.get("病例ID"))
        or _safe_str(case_key)
    )

    raw_treatment = _safe_str(item.get("treatment"))
    raw_final_answer = _safe_str(item.get("final_answer"))

    cdss_result = _safe_str(item.get("cdss_result"))
    if not cdss_result:
        cdss_result = raw_treatment
    if not cdss_result:
        cdss_result = raw_final_answer

    for raw_text in (raw_treatment, raw_final_answer):
        parsed_cdss, _ = _extract_embedded_answer_fields(raw_text)
        if parsed_cdss:
            cdss_result = parsed_cdss
            break

    cdss_think = _safe_str(item.get("cdss_think"))
    if not cdss_think:
        cdss_think = _safe_str(item.get("think") or item.get("<think>"))

    tnm_result = item.get("tnm_result")
    if isinstance(tnm_result, dict):
        parsed = tnm_result.get("parsed")
        if not isinstance(parsed, dict):
            parsed = _parse_tnm_from_text(
                item.get("tnm_ret") or item.get("TNM_ret") or item.get("TNM")
            )
    else:
        parsed = _parse_tnm_from_text(
            item.get("tnm_ret") or item.get("TNM_ret") or item.get("TNM")
        )
        tnm_result = {"parsed": parsed}

    feature_capture = _normalize_feature_capture_flag(
        item.get("关键临床特征是否获取") or item.get("key_clinical_feature_captured")
    )

    stage_result = _safe_str(item.get("stage_result") or item.get("stage"))
    if not stage_result:
        for raw_text in (raw_treatment, raw_final_answer):
            _, parsed_stage = _extract_embedded_answer_fields(raw_text)
            if parsed_stage:
                stage_result = parsed_stage
                break

    classification = _safe_str(item.get("分类") or item.get("classification"))

    normalized = {
        "case_id": case_id,
        "tnm_result": tnm_result,
        "cdss_result": cdss_result,
        "cdss_think": cdss_think,
        "关键临床特征是否获取": feature_capture,
        "classification": classification,
        "分类": classification,
        "stage_result": stage_result,
        "stage": stage_result,
        "tnm_ret": _safe_str(item.get("tnm_ret") or item.get("TNM_ret")),
    }
    return case_id, normalized


def _extract_pred_cases(payload: Any) -> dict[str, Any]:
    # 1) 兼容 {"cases": {...}}
    if isinstance(payload, dict):
        cases = payload.get("cases")
        if isinstance(cases, dict):
            result: dict[str, Any] = {}
            for key, item in cases.items():
                cid, normalized = _normalize_pred_case(_safe_str(key), item)
                if cid:
                    result[cid] = normalized
            return result

        # 2) 兼容直接 dict 按 case_id 键控
        result: dict[str, Any] = {}
        for key, item in payload.items():
            if not isinstance(item, dict):
                continue
            cid, normalized = _normalize_pred_case(_safe_str(key), item)
            if cid:
                result[cid] = normalized
        return result

    # 3) 兼容 list[case]
    if isinstance(payload, list):
        result: dict[str, Any] = {}
        for idx, item in enumerate(payload, start=1):
            key = f"row_{idx}"
            cid, normalized = _normalize_pred_case(key, item)
            if cid:
                result[cid] = normalized
        return result

    return {}


def _metric_file_name(input_name: str) -> str:
    if "output" in input_name:
        return input_name.replace("output", "metric", 1)
    p = Path(input_name)
    if p.suffix:
        return f"{p.stem}_metric{p.suffix}"
    return f"{input_name}_metric"


def _infer_language(pred_file_name: str, payload_language: str) -> str:
    if _safe_str(payload_language):
        return _safe_str(payload_language)
    if "Chinese" in pred_file_name or "chinese" in pred_file_name:
        return "Chinese"
    if "English" in pred_file_name or "english" in pred_file_name:
        return "English"
    return ""


def _is_english_language(language: str) -> bool:
    return _safe_str(language).lower().startswith("english")


def _normalize_language_to_gt(language_text: str) -> str | None:
    text = _safe_str(language_text).strip().lower()
    if not text:
        return None

    # 仅按“完整语言词”匹配，避免 qwen / treatment 等字符串误触发 "en"
    if "中文" in text:
        return "Chinese"
    if "英文" in text:
        return "English"

    word_tokens = re.findall(r"[a-z]+", text)
    token_set = set(word_tokens)

    if token_set & {"chinese", "zh", "cn"}:
        return "Chinese"
    if token_set & {"english", "en"}:
        return "English"

    return None


def _extract_seed_from_ai_payload(pred_file: Path, pred_payload: Any) -> str:
    source_input = _extract_payload_source_input(pred_payload, pred_file.name)
    candidates = [source_input, pred_file.name]

    patterns = [
        re.compile(r"seed[_-]?(\d+)", re.IGNORECASE),
        re.compile(r"(?:^|[_-])(\d{2,5})(?:[_\.-]|$)", re.IGNORECASE),
        re.compile(r"gt[_-](\d+)", re.IGNORECASE),
    ]

    for text in candidates:
        for pat in patterns:
            match = pat.search(text)
            if not match:
                continue
            seed = _safe_str(match.group(1))
            # 避免匹配到版本号/无意义数字，当前项目常见 seed 为 42/2024/3407
            if seed in {"42", "2024", "3407"}:
                return seed

    raise ValueError(f"无法从 AI 文件中提取 seed: {pred_file.name}")


def _resolve_gt_file(
    pred_file: Path,
    pred_payload: Any,
    gt_dir: Path,
    default_language: str = "Chinese",
) -> Path:
    payload_language = _extract_payload_language(pred_payload)
    inferred_language = _infer_language(pred_file.name, payload_language)
    source_input = _extract_payload_source_input(pred_payload, pred_file.name)

    lang = _normalize_language_to_gt(payload_language)
    if not lang:
        lang = _normalize_language_to_gt(inferred_language)
    if not lang:
        lang = _normalize_language_to_gt(source_input)
    if not lang:
        lang = _normalize_language_to_gt(pred_file.name)
    if not lang:
        lang = _normalize_language_to_gt(default_language) or "Chinese"

    seed = _extract_seed_from_ai_payload(pred_file, pred_payload)
    candidate = gt_dir / f"benchmark_gt_cdss_{lang}_seed{seed}.json"
    if candidate.exists():
        return candidate

    # fallback：同 seed 文件中优先 Chinese，再 English，再首个
    same_seed_candidates = sorted(gt_dir.glob(f"*seed{seed}.json"))
    if not same_seed_candidates:
        raise FileNotFoundError(
            "未找到同 seed 的 GT 文件。"
            f" file={pred_file.name}, seed={seed}, gt_dir={gt_dir}"
        )

    for pref in ["Chinese", "English"]:
        pref_candidate = gt_dir / f"benchmark_gt_cdss_{pref}_seed{seed}.json"
        if pref_candidate in same_seed_candidates:
            return pref_candidate

    return same_seed_candidates[0]


def _extract_pred_tnm(case_item: dict[str, Any]) -> dict[str, str]:
    tnm_result = case_item.get("tnm_result")
    if not isinstance(tnm_result, dict):
        tnm_result = {}
    parsed = tnm_result.get("parsed")
    if not isinstance(parsed, dict):
        parsed = {}
    return {
        "T": _safe_str(parsed.get("T")),
        "N": _safe_str(parsed.get("N")),
        "M": _safe_str(parsed.get("M")),
        "Final_TNM": _safe_str(parsed.get("Final_TNM")),
    }


def _extract_gt_tnm(gt_case: dict[str, Any]) -> dict[str, str]:
    tnm_gt = gt_case.get("TNM_GT")
    if not isinstance(tnm_gt, dict):
        tnm_gt = gt_case.get("tnm_gt")
    if not isinstance(tnm_gt, dict):
        tnm_gt = {}

    t_val = _safe_str(tnm_gt.get("T_stage") or tnm_gt.get("T"))
    n_val = _safe_str(tnm_gt.get("N_stage") or tnm_gt.get("N"))
    m_val = _safe_str(tnm_gt.get("M_stage") or tnm_gt.get("M"))
    return {
        "T": t_val,
        "N": n_val,
        "M": m_val,
        "Final_TNM": _safe_str(tnm_gt.get("Final_TNM")),
    }


def _extract_pred_cdss(pred_case: dict[str, Any]) -> dict[str, str]:
    cdss_result = _safe_str(pred_case.get("cdss_result"))
    if not cdss_result:
        cdss_result = _safe_str(pred_case.get("treatment"))
    if not cdss_result:
        cdss_result = _safe_str(pred_case.get("final_answer"))

    cdss_think = _safe_str(pred_case.get("cdss_think"))
    if not cdss_think:
        cdss_think = _safe_str(pred_case.get("think") or pred_case.get("<think>"))

    return {
        "cdss_result": cdss_result,
        "cdss_think": cdss_think,
    }


def _apply_feature_capture_to_cdss_quality(
    cdss_quality: dict[str, Any],
    feature_capture_flag: str,
    case_language: str,
) -> dict[str, Any]:
    # 在不改提示词的前提下，把结构化字段纳入质量分后处理。
    if feature_capture_flag != "否":
        return cdss_quality

    evidence = max(0, _to_score_0_5(cdss_quality.get("evidence")) - 1)
    reasoning = max(0, _to_score_0_5(cdss_quality.get("reasoning")) - 1)
    safety = _to_score_0_5(cdss_quality.get("safety"))
    consistency = max(0, _to_score_0_5(cdss_quality.get("consistency")) - 1)
    quality_score = _mean_no_round([evidence, reasoning, safety, consistency])

    reason = _safe_str(cdss_quality.get("reason"))
    use_english = _is_english_language(case_language)
    note = (
        "Penalty applied: key clinical features were marked as not captured."
        if use_english
        else "已按“关键临床特征是否获取=否”施加质量分惩罚。"
    )
    reason = (reason + " " + note).strip()

    adjusted = dict(cdss_quality)
    adjusted["evidence"] = evidence
    adjusted["reasoning"] = reasoning
    adjusted["safety"] = safety
    adjusted["consistency"] = consistency
    adjusted["quality_score"] = quality_score
    adjusted["reason"] = reason
    return adjusted


def _judge_cdss_accuracy(
    judge_client: LLMClient,
    case_id: str,
    predicted_cdss_result: str,
    gt_cdss_result: str,
    case_language: str,
) -> dict[str, Any]:
    use_english = _is_english_language(case_language)

    if not predicted_cdss_result or not gt_cdss_result:
        return {
            "score": 0,
            "reason": (
                "Missing predicted or GT cdss_result; unable to evaluate similarity."
                if use_english
                else "缺少预测或GT的cdss_result，无法进行有效比对。"
            ),
        }

    if use_english:
        system_prompt = "You are an oncology clinical reviewer. You must output strict JSON only, with no extra text."
        user_prompt = f"""
Please evaluate CDSS result similarity. Case ID: {case_id}

[Base model CDSS result]
{predicted_cdss_result}

[GT CDSS result]
{gt_cdss_result}

Score only from the perspective of result similarity, and do not evaluate writing style.
- score: integer from 0 to 5
- reason: brief rationale (within 120 words)

Output JSON only:
{{
  "score": 0,
  "reason": ""
}}
""".strip()
    else:
        system_prompt = "你是肿瘤临床评审专家。必须严格输出 JSON，不要输出其他内容。"
        user_prompt = f"""
请评估 CDSS 结果相似度。病例ID: {case_id}

【基座模型 CDSS 结果】
{predicted_cdss_result}

【GT CDSS 结果】
{gt_cdss_result}

只从“结果相似度”角度打分，不评估风格。
- score: 0-5 整数
- reason: 简短理由（不超过120字）

只输出 JSON：
{{
  "score": 0,
  "reason": ""
}}
""".strip()

    raw = judge_client.chat(system_prompt=system_prompt, user_prompt=user_prompt)
    parsed = _extract_json_object(raw)
    return {
        "score": _to_score_0_5(parsed.get("score")),
        "reason": _safe_str(parsed.get("reason")),
    }


def _judge_cdss_quality(
    judge_client: LLMClient,
    case_id: str,
    predicted_cdss_result: str,
    gt_cdss_result: str,
    gt_cdss_think: str,
    case_language: str,
) -> dict[str, Any]:
    use_english = _is_english_language(case_language)

    if not predicted_cdss_result or not gt_cdss_result:
        return {
            "evidence": 0,
            "reasoning": 0,
            "safety": 0,
            "consistency": 0,
            "quality_score": 0.0,
            "reason": (
                "Missing predicted or GT cdss_result; unable to evaluate quality."
                if use_english
                else "缺少预测或GT的cdss_result，无法进行有效评估。"
            ),
            "raw_response": "",
        }

    if use_english:
        system_prompt = "You are an oncology clinical reviewer. You must output strict JSON only, with no extra text."
        user_prompt = f"""
Please evaluate CDSS quality from the evidence basis behind the model's conclusion. Case ID: {case_id}

[Base model CDSS result]
{predicted_cdss_result}

[GT CDSS result]
{gt_cdss_result}

[GT CDSS think]
{gt_cdss_think}

GT CDSS think contains the physician's reasoning process and evidence basis, and is usually more detailed than the final result.
Focus on comparing key information points between predicted_cdss_result and gt_cdss_think, not only surface similarity with gt_cdss_result.
Must be expressed concisely and directly, not overly complicated, avoiding excessive verbiage that merely guesses the answer.:
Score the following four dimensions (0-5 integers):
- evidence: whether key evidence points from physician opinion are covered and aligned
- reasoning: whether the reasoning chain is complete and explainable
- safety: whether there are potential medical risks or misleading content
- consistency: consistency and information sufficiency against physician opinion

Output rules:
- quality_score is the average of the four scores (without rounding)
- provide a brief reason

Output JSON only:
{{
  "evidence": 0,
  "reasoning": 0,
  "safety": 0,
  "consistency": 0,
  "quality_score": 0,
  "reason": ""
}}
""".strip()
    else:
        system_prompt = "你是肿瘤临床评审专家。必须严格输出 JSON，不要输出其他内容。"
        user_prompt = f"""
请从模型得出结论的依据上评估 CDSS 质量。病例ID: {case_id}
【基座模型 CDSS 结果】
{predicted_cdss_result}
【GT CDSS 结果】
{gt_cdss_result}
【GT CDSS think】
{gt_cdss_think}
你是一名肿瘤临床决策评估专家。
【评估前提（必须遵守）】
- 考虑模型输出是在“关键临床特征是否获取”的前提下进行决策：两种情况，一，关键临床特征已获取，模型进行医疗决策；二，关键临床特征未获取，模型自己推断分期然后再做医疗决策。	
- 不评估这些输入是否正确，只评估模型   如何基于这些条件进行决策
- 重点评估：模型是否基于这些已知条件，做出正确、规范、直接的治疗决策
- 不鼓励额外假设（如自行补充PS评分、PD-L1、病理类型等）
【评估核心】
- 优先对比模型输出 与 gt_cdss_think（医生真实思考过程）
- 判断模型是否“用对了证据 + 走对了决策路径”，而不是只看结果是否相似
【评分维度（0-5整数）】
1. evidence  
- 是否覆盖并正确使用 gt_cdss_think 中的关键证据点  
- 是否遗漏关键条件（如分期、转移、治疗线）
2. reasoning  
- 是否形成清晰决策链（分期 → 分层 → 治疗路径 → 方案）  
- 是否严格基于已给定信息推理（无自行假设）
3. safety  
- 是否存在潜在医疗风险（错误路径、错误分层、误导治疗）  
- 是否引入未经验证的临床假设
4. consistency  
- 是否与医生思路一致（尤其是决策路径）  
- 是否信息充分但不冗余（避免空话或泛指南复述）
【输出要求】
仅输出JSON：
{{
  "evidence": 0,
  "reasoning": 0,
  "safety": 0,
  "consistency": 0,
  "quality_score": 0,
  "reason": ""
}}
- quality_score 为四项平均分（不要四舍五入）  
- reason 必须简短直接（1句话），不能冗长
""".strip()

    raw = judge_client.chat(system_prompt=system_prompt, user_prompt=user_prompt)
    parsed = _extract_json_object(raw)

    evidence = _to_score_0_5(parsed.get("evidence"))
    reasoning = _to_score_0_5(parsed.get("reasoning"))
    safety = _to_score_0_5(parsed.get("safety"))
    consistency = _to_score_0_5(parsed.get("consistency"))
    quality_score = _mean_no_round([evidence, reasoning, safety, consistency])

    return {
        "evidence": evidence,
        "reasoning": reasoning,
        "safety": safety,
        "consistency": consistency,
        "quality_score": quality_score,
        "reason": _safe_str(parsed.get("reason")),
        "raw_response": raw,
    }


def _judge_treatment_tags(
    judge_client: LLMClient,
    case_id: str,
    predicted_cdss_result: str,
    gt_cdss_result: str,
    case_language: str,
) -> dict[str, Any]:
    use_english = _is_english_language(case_language)

    if not predicted_cdss_result and not gt_cdss_result:
        tags = _default_treatment_tags()
        f1_info = _calc_micro_f1_from_tags(tags)
        return {
            "labels": tags,
            "tp": f1_info["tp"],
            "fp": f1_info["fp"],
            "fn": f1_info["fn"],
            "micro_f1": f1_info["micro_f1"],
            "raw_response": "",
        }

    if use_english:
        system_prompt = "You are an oncology clinical reviewer. You must output strict JSON only, with no extra text."
        user_prompt = f"""
Please compare the following AI treatment recommendation and expert recommendation, then determine whether each treatment category is included by AI and by expert.
Case ID: {case_id}

[AI recommendation]
{predicted_cdss_result}

[Expert recommendation] 
{gt_cdss_result}

Categories:
- S (Surgery): surgery (thoracoscopy, thoracotomy, wedge resection, etc.)
- C (Chemotherapy): chemotherapy (including neoadjuvant and adjuvant)
- R (Radiotherapy): radiotherapy
- T (Targeted): targeted therapy
- I (Immuno): immunotherapy
- O (Observation): observation / follow-up

Output JSON only using this exact schema. Use 1 for yes and 0 for no:
{{
  "surgery": {{"ai": 0, "expert": 0}},
  "chemo": {{"ai": 0, "expert": 0}},
  "radiotherapy": {{"ai": 0, "expert": 0}},
  "targeted": {{"ai": 0, "expert": 0}},
  "immuno": {{"ai": 0, "expert": 0}},
  "observation": {{"ai": 0, "expert": 0}}
}}
""".strip()
    else:
        system_prompt = "你是肿瘤临床评审专家。必须严格输出 JSON，不要输出其他内容。"
        user_prompt = f"""
请对比以下 AI 治疗建议与专家建议，判断 AI 与专家是否包含以下治疗类别。
病例ID: {case_id}

【AI 治疗建议】
{predicted_cdss_result}

【专家建议】
{gt_cdss_result}

类别定义：
- S (Surgery): 手术（胸腔镜、开胸、楔形切除等）
- C (Chemotherapy): 化疗（含新辅助、辅助）
- R (Radiotherapy): 放疗
- T (Targeted): 靶向治疗
- I (Immuno): 免疫治疗
- O (Observation): 观察/随访

只输出 JSON，必须使用以下结构，包含为1，不包含为0：
{{
  "surgery": {{"ai": 0, "expert": 0}},
  "chemo": {{"ai": 0, "expert": 0}},
  "radiotherapy": {{"ai": 0, "expert": 0}},
  "targeted": {{"ai": 0, "expert": 0}},
  "immuno": {{"ai": 0, "expert": 0}},
  "observation": {{"ai": 0, "expert": 0}}
}}
""".strip()

    raw = judge_client.chat(system_prompt=system_prompt, user_prompt=user_prompt)
    parsed = _extract_json_object(raw)
    labels = _normalize_treatment_tags(parsed)
    f1_info = _calc_micro_f1_from_tags(labels)
    return {
        "labels": labels,
        "tp": f1_info["tp"],
        "fp": f1_info["fp"],
        "fn": f1_info["fn"],
        "micro_f1": f1_info["micro_f1"],
        "raw_response": raw,
    }


def _build_metrics_case(
    case_id: str,
    pred_case: dict[str, Any],
    gt_case: dict[str, Any],
    judge_client: LLMClient,
    case_language: str,
) -> dict[str, Any]:
    pred_tnm = _extract_pred_tnm(pred_case)
    gt_tnm = _extract_gt_tnm(gt_case)

    pred_cdss = _extract_pred_cdss(pred_case)
    predicted_cdss_result = pred_cdss["cdss_result"]
    predicted_cdss_think = pred_cdss["cdss_think"]
    feature_capture_flag = _normalize_feature_capture_flag(
        pred_case.get("关键临床特征是否获取") or pred_case.get("key_clinical_feature_captured")
    )
    gt_cdss_result = _safe_str(gt_case.get("cdss_result"))
    gt_cdss_structured_info = gt_case.get("cdss_structured_info")
    if isinstance(gt_cdss_structured_info, (dict, list)):
        gt_cdss_think = json.dumps(gt_cdss_structured_info, ensure_ascii=False)
    else:
        gt_cdss_think = _safe_str(gt_case.get("cdss_think") or gt_cdss_structured_info)

    cdss_accuracy = _judge_cdss_accuracy(
        judge_client=judge_client,
        case_id=case_id,
        predicted_cdss_result=predicted_cdss_result,
        gt_cdss_result=gt_cdss_result,
        case_language=case_language,
    )
    cdss_quality = _judge_cdss_quality(
        judge_client=judge_client,
        case_id=case_id,
        predicted_cdss_result=predicted_cdss_result,
        gt_cdss_result=gt_cdss_result,
        gt_cdss_think=gt_cdss_think,
        case_language=case_language,
    )
    cdss_quality = _apply_feature_capture_to_cdss_quality(
        cdss_quality=cdss_quality,
        feature_capture_flag=feature_capture_flag,
        case_language=case_language,
    )
    treatment_tag_eval = _judge_treatment_tags(
        judge_client=judge_client,
        case_id=case_id,
        predicted_cdss_result=predicted_cdss_result,
        gt_cdss_result=gt_cdss_result,
        case_language=case_language,
    )

    return {
        "case_id": case_id,
        "base_model_tnm": pred_tnm,
        "TNM_gt": gt_tnm,
        "base_model_cdss": {
            "cdss_result": predicted_cdss_result,
            "cdss_think": predicted_cdss_think,
        },
        "关键临床特征是否获取": feature_capture_flag,
        "cdss_accuracy": cdss_accuracy,
        "cdss_quality": cdss_quality,
        "treatment_label_eval": treatment_tag_eval,
        "treatment_micro_f1": treatment_tag_eval.get("micro_f1", 0.0),
    }


def _evaluate_one_case_task(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    case_id = _safe_str(task.get("case_id"))
    pred_case = task.get("pred_case")
    gt_case = task.get("gt_case")
    case_language = _safe_str(task.get("case_language"))
    judge_provider = task.get("judge_provider")

    try:
        if not isinstance(pred_case, dict):
            pred_case = {}
        if not isinstance(gt_case, dict):
            gt_case = {}
        if not isinstance(judge_provider, ProviderConfig):
            raise ValueError("judge_provider 非法")

        judge_client = LLMClient(judge_provider)
        metrics_case = _build_metrics_case(
            case_id=case_id,
            pred_case=pred_case,
            gt_case=gt_case,
            judge_client=judge_client,
            case_language=case_language,
        )
        return case_id, metrics_case
    except Exception as exc:
        return case_id, {
            "case_id": case_id,
            "status": "error",
            "reason": str(exc),
        }


def _resolve_worker_count(requested_workers: int, file_count: int) -> int:
    if file_count <= 1:
        return 1
    if requested_workers <= 0:
        cpu = os.cpu_count() or 1
        return max(1, min(cpu, file_count))
    return max(1, min(requested_workers, file_count))


def _evaluate_single_pred_file(
    *,
    experiment_name: str,
    pred_file: Path,
    gt_dir: Path,
    output_dir: Path,
    default_language: str,
    max_cases: int,
    judge_provider: ProviderConfig,
    case_workers: int = 1,
) -> dict[str, Any]:
    pred_payload = _safe_read_json(pred_file)
    gt_file = _resolve_gt_file(pred_file, pred_payload, gt_dir, default_language=default_language)

    gt_payload = _safe_read_json(gt_file)
    pred_cases = _extract_pred_cases(pred_payload)
    gt_cases = _extract_cases(gt_payload)
    if not isinstance(pred_cases, dict):
        raise ValueError(f"[{experiment_name}] 预测文件 cases 结构非法: {pred_file}")
    if not isinstance(gt_cases, dict):
        raise ValueError(f"[{experiment_name}] GT 文件 cases 结构非法: {gt_file}")

    payload_language = _extract_payload_language(pred_payload)
    language = _infer_language(pred_file.name, payload_language)
    if not _safe_str(language):
        language = _normalize_language_to_gt(default_language) or default_language

    source_input = _extract_payload_source_input(pred_payload, pred_file.name)
    metrics_path = output_dir / _metric_file_name(pred_file.name)

    existing_cases: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            existing_payload = _safe_read_json(metrics_path)
            if isinstance(existing_payload, dict):
                raw_cases = existing_payload.get("cases")
                if isinstance(raw_cases, dict):
                    existing_cases = raw_cases
        except Exception:
            existing_cases = {}

    metrics_payload: dict[str, Any] = {
        "experiment": experiment_name,
        "source_input": source_input,
        "language": language,
        "matched_gt_file": gt_file.name,
        "cases": dict(existing_cases),
    }

    case_items = list(pred_cases.items())
    if max_cases > 0:
        case_items = case_items[:max_cases]

    total = len(case_items)
    pending_case_items: list[tuple[str, Any]] = []
    for case_key, case_item in case_items:
        case_item = case_item if isinstance(case_item, dict) else {}
        case_id = _safe_str(case_item.get("case_id")) or _safe_str(case_key)
        if case_id and case_id in metrics_payload["cases"]:
            existed = metrics_payload["cases"].get(case_id)
            existed_status = (
                _safe_str(existed.get("status")).lower()
                if isinstance(existed, dict)
                else ""
            )
            # 历史异常结果允许重试，避免永久跳过。
            if existed_status != "error":
                continue
        pending_case_items.append((case_key, case_item))

    print(
        f"\n[INFO] Processing: {pred_file.name} | total={total}, existing={len(existing_cases)}, pending={len(pending_case_items)}",
        flush=True,
    )
    print(f"[INFO] Matched GT: {gt_file.name}", flush=True)
    if not pending_case_items:
        file_stats = _compute_cases_metric_stats(metrics_payload["cases"])
        return {
            "status": "ok",
            "file_name": pred_file.name,
            "language": language,
            "source_input": source_input,
            "matched_gt_file": gt_file.name,
            "case_total": total,
            "file_stats": file_stats,
            "metrics_path": str(metrics_path),
            "skipped_all": True,
        }

    case_tasks: list[dict[str, Any]] = []
    for case_key, case_item in pending_case_items:
        case_item = case_item if isinstance(case_item, dict) else {}
        case_id = _safe_str(case_item.get("case_id")) or _safe_str(case_key)
        gt_case = gt_cases.get(case_id)
        gt_case = gt_case if isinstance(gt_case, dict) else {}
        case_tasks.append(
            {
                "case_id": case_id,
                "pred_case": case_item,
                "gt_case": gt_case,
                "case_language": language,
                "judge_provider": judge_provider,
            }
        )

    resolved_case_workers = _resolve_worker_count(case_workers, len(case_tasks))
    print(f"[INFO] case_workers={resolved_case_workers}", flush=True)

    if resolved_case_workers <= 1:
        for idx, task in enumerate(case_tasks, start=1):
            case_id = _safe_str(task.get("case_id"))
            print(f"[INFO] [{idx}/{len(case_tasks)}] Evaluating case: {case_id}", flush=True)
            cid, metrics_case = _evaluate_one_case_task(task)
            metrics_payload["cases"][cid] = metrics_case
            _write_json(metrics_path, metrics_payload)
            print(f"[INFO] [{idx}/{len(case_tasks)}] Saved metrics: {metrics_path.name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=resolved_case_workers) as pool:
            future_map = {
                pool.submit(_evaluate_one_case_task, task): _safe_str(task.get("case_id"))
                for task in case_tasks
            }
            finished = 0
            for fut in as_completed(future_map):
                finished += 1
                default_case_id = future_map[fut]
                try:
                    cid, metrics_case = fut.result()
                except Exception as exc:
                    cid = default_case_id
                    metrics_case = {
                        "case_id": cid,
                        "status": "error",
                        "reason": str(exc),
                    }
                metrics_payload["cases"][cid] = metrics_case
                _write_json(metrics_path, metrics_payload)
                print(f"[INFO] [{finished}/{len(case_tasks)}] Saved metrics: {metrics_path.name}", flush=True)

    file_stats = _compute_cases_metric_stats(metrics_payload["cases"])
    return {
        "status": "ok",
        "file_name": pred_file.name,
        "language": language,
        "source_input": source_input,
        "matched_gt_file": gt_file.name,
        "case_total": total,
        "file_stats": file_stats,
        "metrics_path": str(metrics_path),
    }


def _evaluate_experiment(
    *,
    experiment_name: str,
    pred_dir: Path,
    gt_dir: Path,
    output_dir: Path,
    include: str,
    max_files: int,
    max_cases: int,
    default_language: str,
    judge_provider: ProviderConfig,
    workers: int,
) -> dict[str, Any]:
    if not pred_dir.exists():
        raise FileNotFoundError(f"[{experiment_name}] 预测结果目录不存在: {pred_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    summary_payload: dict[str, Any] = {
        "experiment": experiment_name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": [],
    }
    lang_agg: dict[str, dict[str, Any]] = {}

    pred_files = sorted([p for p in pred_dir.glob(include) if p.is_file() and p.suffix.lower() == ".json"])
    if max_files > 0:
        pred_files = pred_files[:max_files]
    if not pred_files:
        raise FileNotFoundError(f"[{experiment_name}] 在 {pred_dir} 中未找到匹配 {include} 的 json 文件")

    print(f"\n[INFO] ===== 实验组: {experiment_name} =====", flush=True)
    print(f"[INFO] pred_dir={pred_dir}", flush=True)
    print(f"[INFO] output_dir={output_dir}", flush=True)
    resolved_workers = _resolve_worker_count(workers, len(pred_files))
    print(f"[INFO] workers={resolved_workers}", flush=True)

    def _merge_file_result(result: dict[str, Any]) -> None:
        if result.get("status") != "ok":
            summary_payload["records"].append(
                {
                    "type": "file_error",
                    "experiment": experiment_name,
                    "file_name": _safe_str(result.get("file_name")),
                    "error": _safe_str(result.get("error")),
                }
            )
            _write_json(summary_path, summary_payload)
            print(f"[WARN] File failed: {result.get('file_name')} -> {result.get('error')}", flush=True)
            return

        file_stats = result.get("file_stats") if isinstance(result.get("file_stats"), dict) else {}
        file_record = {
            "type": "file",
            "experiment": experiment_name,
            "file_name": _safe_str(result.get("file_name")),
            "language": _safe_str(result.get("language")),
            "source_input": _safe_str(result.get("source_input")),
            "matched_gt_file": _safe_str(result.get("matched_gt_file")),
            "skipped_all": bool(result.get("skipped_all", False)),
            "case_total": int(result.get("case_total", 0)),
            "cdss_accuracy_case_count": int(file_stats.get("cnt_cdss_accuracy", 0)),
            "cdss_quality_case_count": int(file_stats.get("cnt_cdss_quality", 0)),
            "f1_case_count": int(file_stats.get("cnt_f1", 0)),
            "avg_cdss_accuracy": _safe_float(file_stats.get("avg_cdss_accuracy")) or 0.0,
            "avg_cdss_quality": _safe_float(file_stats.get("avg_cdss_quality")) or 0.0,
            "avg_f1": _safe_float(file_stats.get("avg_f1")) or 0.0,
        }
        summary_payload["records"].append(file_record)
        _write_json(summary_path, summary_payload)
        print(f"[INFO] Saved summary checkpoint: {summary_path.name}", flush=True)

        lang_key = _safe_str(result.get("language")) or "Unknown"
        if lang_key not in lang_agg:
            lang_agg[lang_key] = {
                "file_count": 0,
                "case_total": 0,
                "sum_cdss_accuracy": 0.0,
                "cnt_cdss_accuracy": 0,
                "sum_cdss_quality": 0.0,
                "cnt_cdss_quality": 0,
                "sum_f1": 0.0,
                "cnt_f1": 0,
            }

        lang_item = lang_agg[lang_key]
        lang_item["file_count"] += 1
        lang_item["case_total"] += int(result.get("case_total", 0))
        lang_item["sum_cdss_accuracy"] += _safe_float(file_stats.get("sum_cdss_accuracy")) or 0.0
        lang_item["cnt_cdss_accuracy"] += int(file_stats.get("cnt_cdss_accuracy", 0))
        lang_item["sum_cdss_quality"] += _safe_float(file_stats.get("sum_cdss_quality")) or 0.0
        lang_item["cnt_cdss_quality"] += int(file_stats.get("cnt_cdss_quality", 0))
        lang_item["sum_f1"] += _safe_float(file_stats.get("sum_f1")) or 0.0
        lang_item["cnt_f1"] += int(file_stats.get("cnt_f1", 0))

    if resolved_workers <= 1:
        for pred_file in pred_files:
            try:
                result = _evaluate_single_pred_file(
                    experiment_name=experiment_name,
                    pred_file=pred_file,
                    gt_dir=gt_dir,
                    output_dir=output_dir,
                    default_language=default_language,
                    max_cases=max_cases,
                    judge_provider=judge_provider,
                    case_workers=int(workers),
                )
            except Exception as exc:
                result = {
                    "status": "error",
                    "file_name": pred_file.name,
                    "error": str(exc),
                }
            _merge_file_result(result)
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as pool:
            future_map = {
                pool.submit(
                    _evaluate_single_pred_file,
                    experiment_name=experiment_name,
                    pred_file=pred_file,
                    gt_dir=gt_dir,
                    output_dir=output_dir,
                    default_language=default_language,
                    max_cases=max_cases,
                    judge_provider=judge_provider,
                    case_workers=1,
                ): pred_file
                for pred_file in pred_files
            }

            for fut in as_completed(future_map):
                src_file = future_map[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {
                        "status": "error",
                        "file_name": src_file.name,
                        "error": str(exc),
                    }
                _merge_file_result(result)

    for lang_key in sorted(lang_agg.keys()):
        lang_item = lang_agg[lang_key]
        lang_record = {
            "type": "language",
            "experiment": experiment_name,
            "language": lang_key,
            "file_count": lang_item["file_count"],
            "case_total": lang_item["case_total"],
            "cdss_accuracy_case_count": lang_item["cnt_cdss_accuracy"],
            "cdss_quality_case_count": lang_item["cnt_cdss_quality"],
            "f1_case_count": lang_item["cnt_f1"],
            "avg_cdss_accuracy": _avg_or_zero(lang_item["sum_cdss_accuracy"], lang_item["cnt_cdss_accuracy"]),
            "avg_cdss_quality": _avg_or_zero(lang_item["sum_cdss_quality"], lang_item["cnt_cdss_quality"]),
            "avg_f1": _avg_or_zero(lang_item["sum_f1"], lang_item["cnt_f1"]),
        }
        summary_payload["records"].append(lang_record)

    summary_payload["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(summary_path, summary_payload)
    print(f"[INFO] Saved final summary: {summary_path.name}", flush=True)
    return summary_payload



    ####

def normalize_case_context(case_json: Dict[str, Any]) -> Dict[str, Any]:
    def getv(*keys, default="unknown"):
        for k in keys:
            if k in case_json and case_json[k] not in [None, ""]:
                return case_json[k]
        return default

    def first_non_empty(*values, default="unknown"):
        for value in values:
            text = _safe_str(value)
            if text and text.lower() != "unknown":
                return text
        return default

    driver = getv("driver_gene", "driver_status", "driver", "驱动基因")
    treatment_stage = getv("treatment_stage", "治疗阶段")
    prior_treatment = getv("prior_treatment", "既往治疗方案", "previous_treatment", "prior_therapy")
    metastasis_type = getv("metastasis_type", "转移类型", "metastasis")
    therapy_line = getv("line_of_therapy", "therapy_line", "treatment_line", "是否一线治疗")
    ps = getv("ps_score", "PS", "体力评分", "performance_status")
    pdl1 = getv("pdl1", "PDL1", "PD-L1", "pdl1_expression")
    immuno_contra = getv(
        "immunotherapy_contraindication",
        "免疫禁忌",
        "immunotherapy_contra",
        "contraindication_for_immunotherapy",
    )
    pathology = getv("pathology_type", "病理类型", "histology", "pathology")
    stage_text = _safe_str(getv("stage", "分期", "stage_result", "classification", "分类", default=""))
    tnm_text = _safe_str(getv("tnm", "TNM", "tnm_ret", "TNM_ret", "tnm_for_cdss", default=""))

    tnm_result = case_json.get("tnm_result")
    if isinstance(tnm_result, dict):
        parsed = tnm_result.get("parsed")
        if isinstance(parsed, dict):
            parsed_triplet = " ".join(
                part for part in [_safe_str(parsed.get("T")), _safe_str(parsed.get("N")), _safe_str(parsed.get("M"))] if part
            )
            tnm_text = first_non_empty(tnm_text, parsed.get("Final_TNM"), parsed_triplet, default="")

    tnm_gt = case_json.get("TNM_GT")
    if not isinstance(tnm_gt, dict):
        tnm_gt = case_json.get("tnm_gt")
    if isinstance(tnm_gt, dict):
        gt_triplet = " ".join(
            part
            for part in [
                _safe_str(tnm_gt.get("T_stage") or tnm_gt.get("T")),
                _safe_str(tnm_gt.get("N_stage") or tnm_gt.get("N")),
                _safe_str(tnm_gt.get("M_stage") or tnm_gt.get("M")),
            ]
            if part
        )
        tnm_text = first_non_empty(tnm_text, tnm_gt.get("Final_TNM"), gt_triplet, default="")

    if not stage_text:
        stage_text = "unknown"
    if not tnm_text:
        tnm_text = "unknown"

    # driver
    driver_s = _safe_str(driver)
    driver_l = driver_s.lower()
    if driver_s == "0" or driver_l in {"negative", "driver_negative", "wildtype", "wild-type"}:
        driver_status = "driver_negative"
    elif "egfr" in driver_l:
        driver_status = "driver_positive_EGFR"
    elif "alk" in driver_l:
        driver_status = "driver_positive_ALK"
    elif "ros1" in driver_l:
        driver_status = "driver_positive_ROS1"
    elif driver_s.upper() in {"EGFR", "ALK", "ROS1", "BRAF", "MET", "RET", "NTRK"}:
        driver_status = f"driver_positive_{driver_s.upper()}"
    else:
        driver_status = "unknown"

    # treatment stage
    treatment_stage_s = _safe_str(treatment_stage).lower()
    if treatment_stage_s in {"0", "preop", "pre-operative", "preoperative", "neoadjuvant"}:
        treatment_stage_norm = "radical_preoperative"
    elif treatment_stage_s in {"1", "postop", "post-operative", "postoperative", "adjuvant"}:
        treatment_stage_norm = "radical_postoperative"
    else:
        treatment_stage_norm = "unknown"

    # prior treatment
    prior_s = _safe_str(prior_treatment).lower()
    if prior_s in {"1", "treatment_naive", "naive", "untreated"}:
        prior_treatment_norm = "treatment_naive"
    elif prior_s and prior_s != "unknown":
        prior_treatment_norm = "previously_treated"
    else:
        prior_treatment_norm = "unknown"

    # metastasis
    meta_s = _safe_str(metastasis_type).lower()
    if meta_s in {"0", "oligometastatic", "oligometastasis", "limited_metastatic"}:
        metastasis_norm = "oligometastatic"
    elif meta_s in {"1", "extensive_metastatic", "widespread_metastatic", "multiple_metastatic"}:
        metastasis_norm = "extensive_metastatic"
    else:
        metastasis_norm = "unknown"

    # line
    line_s = _safe_str(therapy_line).lower()
    if line_s in {"1", "first", "first_line", "first-line", "1st", "一线"} or "first line" in line_s:
        therapy_line_norm = "first_line"
    elif line_s in {"0", "second", "second_line", "second-line", "2nd", "二线"} or "second line" in line_s:
        therapy_line_norm = "second_line"
    else:
        therapy_line_norm = "unknown"

    # ps
    ps_s = _safe_str(ps).lower().replace(" ", "")
    if ps_s in {"0", "1", "0-1", "0~1", "ps0-1", "ps0~1"}:
        ps_group = "PS_0_1"
    elif ps_s in {"2", "ps2"}:
        ps_group = "PS_2"
    elif ps_s in {"3", "4", "3-4", "ps3", "ps4", "ps3-4"}:
        ps_group = "PS_3_4"
    else:
        ps_group = "unknown"

    # pdl1
    pdl1s = _safe_str(pdl1).lower()
    if pdl1s in {"unknown", "", "未检测", "未知", "not tested", "not available", "na", "n/a"}:
        pdl1_group = "PDL1_UNKNOWN"
    elif re.search(r"(>=|≥)\s*50|50\s*%\s*(or\s+more|or\s+greater|and\s+above)", pdl1s):
        pdl1_group = "PDL1_GE_50"
    elif re.search(r"1\s*[-~]\s*49|1\s*%\s*[-~]\s*49\s*%", pdl1s):
        pdl1_group = "PDL1_1_49"
    elif re.search(r"<\s*1|less\s+than\s*1", pdl1s):
        pdl1_group = "PDL1_LT_1"
    else:
        pdl1_group = "PDL1_UNKNOWN"

    # immunotherapy eligibility
    immuno_s = _safe_str(immuno_contra).lower()
    if immuno_s in {"none", "无", "无免疫禁忌", "false", "0", "no", "no contraindication"}:
        immunotherapy_eligibility = "immunotherapy_eligible"
    elif immuno_s in {"present", "有", "有免疫禁忌", "true", "1", "contraindicated"}:
        immunotherapy_eligibility = "immunotherapy_ineligible"
    else:
        immunotherapy_eligibility = "unknown"

    # pathology
    pathology_s = _safe_str(pathology).lower()
    if pathology_s in {"nonsquamous", "non-squamous", "non squamous", "非鳞癌", "腺癌", "adenocarcinoma"}:
        pathology_type = "nonsquamous"
    elif pathology_s in {"squamous", "鳞癌", "squamous cell", "squamous carcinoma"}:
        pathology_type = "squamous"
    else:
        pathology_type = "pathology_unknown"

    # disease stage
    stage_hint = f"{stage_text} {tnm_text}".lower()
    whole_stage = stage_hint.upper()
    if (
        any(x in whole_stage for x in ["IV", "IVA", "IVB", "M1A", "M1B", "M1C"])
        or "advanced" in stage_hint
        or "metastatic" in stage_hint
    ):
        disease_stage = "advanced"
    elif any(x in whole_stage for x in ["III", "IIIA", "IIIB", "IIIC"]) or "locally advanced" in stage_hint:
        disease_stage = "locally_advanced"
    elif any(x in whole_stage for x in ["I", "IA", "IB", "II", "IIA", "IIB"]) or "early" in stage_hint:
        disease_stage = "early"
    else:
        disease_stage = "unknown"

    # resectability heuristic
    if disease_stage == "advanced" or metastasis_norm == "extensive_metastatic":
        resectability = "unresectable"
    elif metastasis_norm == "oligometastatic":
        resectability = "potentially_resectable"
    elif disease_stage == "early":
        resectability = "resectable"
    else:
        resectability = "unknown"

    return {
        "driver_status": driver_status,
        "treatment_stage": treatment_stage_norm,
        "prior_treatment": prior_treatment_norm,
        "metastasis_type": metastasis_norm,
        "therapy_line": therapy_line_norm,
        "ps_group": ps_group,
        "pdl1_group": pdl1_group,
        "immunotherapy_eligibility": immunotherapy_eligibility,
        "pathology_type": pathology_type,
        "disease_stage": disease_stage,
        "resectability": resectability,
        "raw_stage_text": stage_text,
        "raw_tnm_text": tnm_text,
    }
NORMALIZATION_LEXICON = {
    "驱动基因阴性": "driver_negative",
    "无驱动基因突变": "driver_negative",
    "驱动阴性": "driver_negative",
    "EGFR阳性": "driver_positive_EGFR",
    "ALK阳性": "driver_positive_ALK",
    "ROS1阳性": "driver_positive_ROS1",
    "一线治疗": "first_line",
    "初始治疗": "first_line",
    "初治": "first_line",
    "二线治疗": "second_line",
    "广泛转移": "extensive_metastatic",
    "晚期广泛转移": "extensive_metastatic",
    "全身多发转移": "extensive_metastatic",
    "寡转移": "oligometastatic",
    "可切除": "resectable",
    "潜在可切除": "potentially_resectable",
    "不可切除": "unresectable",
    "手术优先": "surgery_priority",
    "根治性手术": "radical_surgery",
    "新辅助治疗": "neoadjuvant",
    "术后辅助治疗": "adjuvant",
    "系统治疗": "systemic_treatment",
    "全身药物治疗": "systemic_treatment",
    "最佳支持治疗": "best_supportive_care",
    "免疫治疗": "immunotherapy",
    "化疗": "chemotherapy",
    "免疫联合化疗": "immunochemotherapy",
    "免疫治疗联合化疗": "immunochemotherapy",
    "免疫联合含铂双药化疗": "immunochemotherapy_platinum_doublet",
    "双免治疗": "dual_immunotherapy",
    "双免联合化疗": "dual_immunotherapy_plus_chemo",
    "化疗联合抗血管": "chemo_plus_antiangiogenic",
    "免疫联合抗血管联合化疗": "immuno_antiangiogenic_chemo",
    "含铂双药": "platinum_doublet",
    "顺铂": "cisplatin",
    "卡铂": "carboplatin",
    "鳞癌": "squamous",
    "非鳞癌": "nonsquamous",
    "病理未知": "pathology_unknown",
    "待明确病理": "pathology_unknown",
    "培美曲塞": "pemetrexed",
    "吉西他滨": "gemcitabine",
    "紫杉醇": "paclitaxel",
    "白蛋白紫杉醇": "nab_paclitaxel",
    "长春瑞滨": "vinorelbine",
    "贝伐珠单抗": "bevacizumab",
    "帕博利珠单抗": "pembrolizumab",
    "阿替利珠单抗": "atezolizumab",
    "西米普利单抗": "cemiplimab",
    "纳武利尤单抗": "nivolumab",
    "伊匹木单抗": "ipilimumab",
    "度伐利尤单抗": "durvalumab",
    "非鳞癌优选培美曲塞": "preferred_nonsquamous_pemetrexed",
    "鳞癌优选吉西他滨": "preferred_squamous_gemcitabine",
    # English aliases
    "driver gene negative": "driver_negative",
    "driver-gene negative": "driver_negative",
    "driver negative": "driver_negative",
    "driver-negative": "driver_negative",
    "no driver mutation": "driver_negative",
    "no actionable driver mutation": "driver_negative",
    "egfr positive": "driver_positive_EGFR",
    "egfr mutation positive": "driver_positive_EGFR",
    "alk positive": "driver_positive_ALK",
    "alk rearrangement positive": "driver_positive_ALK",
    "ros1 positive": "driver_positive_ROS1",
    "ros1 rearrangement positive": "driver_positive_ROS1",
    "first-line": "first_line",
    "first line": "first_line",
    "initial treatment": "first_line",
    "treatment-naive": "first_line",
    "second-line": "second_line",
    "second line": "second_line",
    "later-line": "second_line",
    "extensive metastases": "extensive_metastatic",
    "widespread metastases": "extensive_metastatic",
    "multiple metastases": "extensive_metastatic",
    "systemic spread": "extensive_metastatic",
    "oligometastatic": "oligometastatic",
    "oligometastasis": "oligometastatic",
    "potentially resectable": "potentially_resectable",
    "unresectable": "unresectable",
    "inoperable": "unresectable",
    "resectable": "resectable",
    "surgery first": "surgery_priority",
    "surgery priority": "surgery_priority",
    "radical surgery": "radical_surgery",
    "curative surgery": "radical_surgery",
    "neoadjuvant therapy": "neoadjuvant",
    "adjuvant therapy": "adjuvant",
    "systemic therapy": "systemic_treatment",
    "systemic treatment": "systemic_treatment",
    "best supportive care": "best_supportive_care",
    "palliative care": "best_supportive_care",
    "targeted therapy": "targeted_therapy",
    "immunotherapy": "immunotherapy",
    "chemotherapy": "chemotherapy",
    "chemoimmunotherapy": "immunochemotherapy",
    "immunochemotherapy": "immunochemotherapy",
    "immunotherapy plus chemotherapy": "immunochemotherapy",
    "immunotherapy combined with chemotherapy": "immunochemotherapy",
    "immunotherapy plus platinum doublet chemotherapy": "immunochemotherapy_platinum_doublet",
    "dual immunotherapy": "dual_immunotherapy",
    "dual immunotherapy plus chemotherapy": "dual_immunotherapy_plus_chemo",
    "chemotherapy plus anti-angiogenic": "chemo_plus_antiangiogenic",
    "immunotherapy plus anti-angiogenic plus chemotherapy": "immuno_antiangiogenic_chemo",
    "platinum doublet": "platinum_doublet",
    "platinum-based chemotherapy": "platinum_doublet",
    "cisplatin": "cisplatin",
    "carboplatin": "carboplatin",
    "non-squamous": "nonsquamous",
    "nonsquamous": "nonsquamous",
    "adenocarcinoma": "nonsquamous",
    "squamous cell": "squamous",
    "squamous carcinoma": "squamous",
    "histology unknown": "pathology_unknown",
    "pathology unknown": "pathology_unknown",
    "pathology pending": "pathology_unknown",
    "pemetrexed": "pemetrexed",
    "gemcitabine": "gemcitabine",
    "paclitaxel": "paclitaxel",
    "nab-paclitaxel": "nab_paclitaxel",
    "vinorelbine": "vinorelbine",
    "bevacizumab": "bevacizumab",
    "pembrolizumab": "pembrolizumab",
    "atezolizumab": "atezolizumab",
    "cemiplimab": "cemiplimab",
    "nivolumab": "nivolumab",
    "ipilimumab": "ipilimumab",
    "durvalumab": "durvalumab",
    "non-squamous preferred pemetrexed": "preferred_nonsquamous_pemetrexed",
    "squamous preferred gemcitabine": "preferred_squamous_gemcitabine",
}

def concept_category(norm: str) -> str:
    if norm in {"driver_negative", "driver_positive_EGFR", "driver_positive_ALK", "driver_positive_ROS1"}:
        return "driver_status"
    if norm in {"first_line", "second_line"}:
        return "therapy_line"
    if norm in {"resectable", "potentially_resectable", "unresectable"}:
        return "resectability"
    if norm in {"surgery_priority", "radical_surgery", "neoadjuvant", "adjuvant", "systemic_treatment", "best_supportive_care"}:
        return "treatment_intent"
    if norm in {"immunotherapy", "chemotherapy", "immunochemotherapy", "immunochemotherapy_platinum_doublet",
                "dual_immunotherapy", "dual_immunotherapy_plus_chemo", "chemo_plus_antiangiogenic",
                "targeted_therapy",
                "immuno_antiangiogenic_chemo"}:
        return "regimen_class"
    if norm in {"platinum_doublet", "cisplatin", "carboplatin"}:
        return "platinum_backbone"
    if norm in {"squamous", "nonsquamous", "pathology_unknown"}:
        return "pathology_branching"
    if norm in {"preferred_nonsquamous_pemetrexed"}:
        return "preferred_regimen_nonsquamous"
    if norm in {"preferred_squamous_gemcitabine"}:
        return "preferred_regimen_squamous"
    if norm in {"pembrolizumab", "atezolizumab", "cemiplimab", "nivolumab", "ipilimumab", "durvalumab",
                "bevacizumab", "pemetrexed", "gemcitabine", "paclitaxel", "nab_paclitaxel", "vinorelbine"}:
        return "specific_drugs"
    return "other"

def extract_concepts_rule_based(text: str) -> List[Dict[str, str]]:
    text = _safe_str(text)
    if not text:
        return []
    results = []
    for surface, norm in sorted(NORMALIZATION_LEXICON.items(), key=lambda x: len(x[0]), reverse=True):
        if _surface_in_text(text, surface):
            results.append({
                "surface_text": surface,
                "normalized_concept": norm,
                "category": concept_category(norm)
            })
    return results

def concepts_to_slots(concepts, normalized_case_context=None, *, seed_from_case_context: bool = False):
    context = normalized_case_context if isinstance(normalized_case_context, dict) else {}
    slots = {
        "disease_stage": context.get("disease_stage", "unknown") if seed_from_case_context else "unknown",
        "resectability": context.get("resectability", "unknown") if seed_from_case_context else "unknown",
        "driver_status": context.get("driver_status", "unknown") if seed_from_case_context else "unknown",
        "therapy_line": context.get("therapy_line", "unknown") if seed_from_case_context else "unknown",
        "ps_group": context.get("ps_group", "unknown") if seed_from_case_context else "unknown",
        "pdl1_group": context.get("pdl1_group", "PDL1_UNKNOWN") if seed_from_case_context else "PDL1_UNKNOWN",
        "immunotherapy_eligibility": context.get("immunotherapy_eligibility", "unknown") if seed_from_case_context else "unknown",
        "treatment_intent": "unknown",
        "regimen_class": [],
        "platinum_backbone": "unknown",
        "pathology_branching": [],
        "preferred_regimen_nonsquamous": [],
        "preferred_regimen_squamous": [],
        "specific_drugs": [],
    }

    for c in concepts:
        norm = c["normalized_concept"]
        cat = c["category"]

        if cat == "driver_status":
            slots["driver_status"] = norm
        elif cat == "therapy_line":
            slots["therapy_line"] = norm
        elif cat == "resectability":
            current = slots["resectability"]
            # Keep stronger resectability decisions when both generic and specific
            # terms are present (e.g., "potentially resectable" also contains
            # "resectable").
            if current == "unresectable":
                continue
            if current == "potentially_resectable" and norm == "resectable":
                continue
            slots["resectability"] = norm
        elif cat == "treatment_intent":
            if norm in {"surgery_priority", "systemic_treatment", "best_supportive_care"}:
                slots["treatment_intent"] = norm
            elif norm == "neoadjuvant":
                slots["treatment_intent"] = "neoadjuvant_then_surgery"
        elif cat == "regimen_class":
            if norm not in slots["regimen_class"]:
                slots["regimen_class"].append(norm)
        elif cat == "platinum_backbone":
            if norm == "platinum_doublet":
                slots["platinum_backbone"] = "platinum_doublet"
        elif cat == "pathology_branching":
            if norm not in slots["pathology_branching"]:
                slots["pathology_branching"].append(norm)
        elif cat == "preferred_regimen_nonsquamous":
            if "pemetrexed_platinum" not in slots["preferred_regimen_nonsquamous"]:
                slots["preferred_regimen_nonsquamous"].append("pemetrexed_platinum")
        elif cat == "preferred_regimen_squamous":
            if "gemcitabine_platinum" not in slots["preferred_regimen_squamous"]:
                slots["preferred_regimen_squamous"].append("gemcitabine_platinum")
        elif cat == "specific_drugs":
            if norm not in slots["specific_drugs"]:
                slots["specific_drugs"].append(norm)

    if not slots["pathology_branching"]:
        slots["pathology_branching"] = ["pathology_unknown"]

    if not slots["preferred_regimen_nonsquamous"]:
        slots["preferred_regimen_nonsquamous"] = ["unknown"]
    if not slots["preferred_regimen_squamous"]:
        slots["preferred_regimen_squamous"] = ["unknown"]

    return slots

MATCH_SCORE = {
    "exact": 1.0,
    "hierarchical_equivalent": 0.8,
    "clinically_acceptable": 0.6,
    "weak_partial": 0.3,
    "contradictory": 0.0,
    "missing": 0.0,
}

SLOT_WEIGHTS = {
    "disease_stage": 0.12,
    "resectability": 0.15,
    "driver_status": 0.12,
    "therapy_line": 0.08,
    "ps_group": 0.08,
    "pdl1_group": 0.05,
    "immunotherapy_eligibility": 0.08,
    "treatment_intent": 0.15,
    "regimen_class": 0.08,
    "platinum_backbone": 0.04,
    "pathology_branching": 0.05,
    "preferred_regimen_nonsquamous": 0.03,
    "preferred_regimen_squamous": 0.03,
    "specific_drugs": 0.02
}

SINGLE_SLOT_KEYS = [
    "disease_stage",
    "resectability",
    "driver_status",
    "therapy_line",
    "ps_group",
    "pdl1_group",
    "immunotherapy_eligibility",
    "treatment_intent",
    "platinum_backbone",
]

MULTI_SLOT_KEYS = [
    "regimen_class",
    "pathology_branching",
    "preferred_regimen_nonsquamous",
    "preferred_regimen_squamous",
    "specific_drugs",
]


def _single_slot_is_unknown(slot_name: str, value: Any) -> bool:
    text = _safe_str(value)
    if slot_name == "pdl1_group":
        return text in {"", "unknown", "PDL1_UNKNOWN"}
    return text in {"", "unknown"}


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_safe_str(v) for v in value if _safe_str(v)]
    text = _safe_str(value)
    if text:
        return [text]
    return []


def _multi_slot_has_signal(slot_name: str, value: Any) -> bool:
    vals = _as_str_list(value)
    if not vals:
        return False
    if slot_name == "pathology_branching":
        return any(v != "pathology_unknown" for v in vals)
    if slot_name in {"preferred_regimen_nonsquamous", "preferred_regimen_squamous"}:
        return any(v != "unknown" for v in vals)
    return any(v != "unknown" for v in vals)


def build_valid_slots(reference_slots: dict[str, Any], normalized_case_context: dict[str, Any]) -> list[str]:
    ctx = normalized_case_context if isinstance(normalized_case_context, dict) else {}
    valid: set[str] = set()

    # Core clinically-determinable slots can be evaluated from case context.
    for slot in ["disease_stage", "resectability", "driver_status", "therapy_line", "ps_group", "immunotherapy_eligibility"]:
        if not _single_slot_is_unknown(slot, ctx.get(slot)):
            valid.add(slot)

    if not _single_slot_is_unknown("pdl1_group", ctx.get("pdl1_group")):
        valid.add("pdl1_group")

    # GT/reference explicit mention enables evaluation even when context is weak.
    for slot in SINGLE_SLOT_KEYS:
        if not _single_slot_is_unknown(slot, reference_slots.get(slot)):
            valid.add(slot)

    for slot in MULTI_SLOT_KEYS:
        if _multi_slot_has_signal(slot, reference_slots.get(slot)):
            valid.add(slot)

    # Conditional slots: only evaluate when clinically meaningful.
    if _single_slot_is_unknown("pdl1_group", ctx.get("pdl1_group")) and _single_slot_is_unknown(
        "pdl1_group", reference_slots.get("pdl1_group")
    ):
        valid.discard("pdl1_group")

    pathology_known = _safe_str(ctx.get("pathology_type")) not in {"", "unknown", "pathology_unknown"}
    pathology_signaled = _multi_slot_has_signal("pathology_branching", reference_slots.get("pathology_branching"))
    if not pathology_known and not pathology_signaled:
        valid.discard("pathology_branching")
        valid.discard("preferred_regimen_nonsquamous")
        valid.discard("preferred_regimen_squamous")

    return [slot for slot in SLOT_WEIGHTS.keys() if slot in valid]


def match_single(ref, cand):
    if ref == cand:
        return "exact"

    hierarchical = {
        ("immunochemotherapy_platinum_doublet", "immunochemotherapy"),
        ("systemic_treatment", "immunochemotherapy"),
        ("systemic_treatment", "chemotherapy"),
        ("systemic_treatment", "targeted_therapy"),
        ("preferred_nonsquamous_pemetrexed", "pemetrexed_platinum"),
        ("preferred_squamous_gemcitabine", "gemcitabine_platinum"),
    }
    if (ref, cand) in hierarchical:
        return "hierarchical_equivalent"

    if ref == "unknown" or cand == "unknown":
        return "missing"

    return "contradictory"

def match_multi(ref_list, cand_list):
    ref_set = set(ref_list or [])
    cand_set = set(cand_list or [])
    if not ref_set and not cand_set:
        return "missing"
    if ref_set == cand_set:
        return "exact"
    if ref_set & cand_set:
        return "weak_partial"

    # hierarchical on regimen
    if "immunochemotherapy_platinum_doublet" in ref_set and "immunochemotherapy" in cand_set:
        return "hierarchical_equivalent"
    if "pathology_unknown" in ref_set and "pathology_branching_required" in cand_set:
        return "clinically_acceptable"

    return "contradictory"

def match_slots(reference_slots, candidate_slots, normalized_case_context):
    slot_match_results = {}

    for slot in SINGLE_SLOT_KEYS:
        ref = reference_slots.get(slot, "unknown")
        cand = candidate_slots.get(slot, "unknown")
        mt = match_single(ref, cand)
        slot_match_results[slot] = {
            "reference_value": ref,
            "candidate_value": cand,
            "match_type": mt
        }

    for slot in MULTI_SLOT_KEYS:
        ref = reference_slots.get(slot, [])
        cand = candidate_slots.get(slot, [])
        mt = match_multi(ref, cand)
        slot_match_results[slot] = {
            "reference_value": ref,
            "candidate_value": cand,
            "match_type": mt
        }

    # clinically acceptable rules
    if normalized_case_context.get("pdl1_group") == "PDL1_UNKNOWN":
        rc = slot_match_results.get("regimen_class", {})
        cand_vals = set(candidate_slots.get("regimen_class", []))
        if "immunochemotherapy" in cand_vals and rc["match_type"] == "contradictory":
            slot_match_results["regimen_class"]["match_type"] = "clinically_acceptable"

    if normalized_case_context.get("pathology_type") == "pathology_unknown":
        cand_vals = set(candidate_slots.get("pathology_branching", []))
        if "pathology_branching_required" in cand_vals:
            slot_match_results["pathology_branching"]["match_type"] = "clinically_acceptable"

    return slot_match_results

def detect_major_contradictions(reference_slots, candidate_slots):
    contradictions = []

    if reference_slots.get("treatment_intent") == "neoadjuvant_then_surgery" and candidate_slots.get("treatment_intent") == "systemic_treatment":
        contradictions.append({
            "type": "pathway_conflict",
            "description": "reference is neoadjuvant_then_surgery but candidate is systemic_treatment",
            "penalty": 0.30
        })

    if reference_slots.get("treatment_intent") == "systemic_treatment" and candidate_slots.get("treatment_intent") == "surgery_priority":
        contradictions.append({
            "type": "pathway_conflict",
            "description": "reference is systemic_treatment but candidate is surgery_priority",
            "penalty": 0.30
        })

    if reference_slots.get("driver_status") == "driver_negative" and "targeted_therapy" in candidate_slots.get("regimen_class", []):
        contradictions.append({
            "type": "driver_path_conflict",
            "description": "driver negative but candidate recommends targeted therapy",
            "penalty": 0.25
        })

    if reference_slots.get("immunotherapy_eligibility") == "immunotherapy_ineligible" and (
        "immunotherapy" in candidate_slots.get("regimen_class", []) or
        "immunochemotherapy" in candidate_slots.get("regimen_class", [])
    ):
        contradictions.append({
            "type": "eligibility_conflict",
            "description": "immunotherapy ineligible but candidate uses immunotherapy",
            "penalty": 0.25
        })

    return contradictions

def calc_score(slot_match_results, major_correct_pathways, major_contradictions, *, valid_slots: list[str] | None = None):
    score_breakdown = {}
    base_score = 0.0
    valid_set = set(valid_slots or [])
    slots_for_score = [slot for slot in SLOT_WEIGHTS.keys() if (not valid_set or slot in valid_set)]
    total_weight = sum(SLOT_WEIGHTS.get(slot, 0.0) for slot in slots_for_score)
    if total_weight <= 0:
        total_weight = 1.0

    for slot_name, weight in SLOT_WEIGHTS.items():
        included = slot_name in slots_for_score
        if included:
            effective_weight = weight / total_weight
        else:
            effective_weight = 0.0
        match_type = slot_match_results.get(slot_name, {}).get("match_type", "missing")
        match_score = MATCH_SCORE.get(match_type, 0.0)
        weighted_score = (match_score * effective_weight) if included else 0.0
        base_score += weighted_score
        score_breakdown[slot_name] = {
            "match_type": match_type,
            "match_score": match_score,
            "weight": weight,
            "effective_weight": round(effective_weight, 4),
            "included": included,
            "weighted_score": round(weighted_score, 4)
        }

    bonus = min(0.02 * len(major_correct_pathways), 0.10)
    penalty = round(sum(x.get("penalty", 0.30) for x in major_contradictions), 4)
    raw_final_score = base_score - penalty + bonus
    final_score = round(min(1.0, max(0.0, raw_final_score)), 4)

    return {
        "base_score": round(base_score, 4),
        "bonus": round(bonus, 4),
        "penalty": penalty,
        "raw_final_score": round(raw_final_score, 4),
        "final_score": final_score,
        "valid_slots": slots_for_score,
        "valid_slot_count": len(slots_for_score),
        "score_breakdown": score_breakdown
    }

def medical_eval_pipeline(case_json, reference_answer, candidate_answer,
                          ref_model_extra_concepts=None, cand_model_extra_concepts=None,
                          ref_model_corrected_slots=None, cand_model_corrected_slots=None,
                          final_judgment="roughly_equal", clinical_reasoning_summary=None):
    # 1. normalize case
    normalized_case_context = normalize_case_context(case_json)

    # 2. rule-based concept extraction
    reference_concepts = extract_concepts_rule_based(reference_answer)
    candidate_concepts = extract_concepts_rule_based(candidate_answer)

    # 3. add model-supplemented concepts if any
    if ref_model_extra_concepts:
        reference_concepts.extend(ref_model_extra_concepts)
    if cand_model_extra_concepts:
        candidate_concepts.extend(cand_model_extra_concepts)

    # 4. concept -> slots
    reference_slots = concepts_to_slots(reference_concepts, normalized_case_context, seed_from_case_context=False)
    candidate_slots = concepts_to_slots(candidate_concepts, normalized_case_context, seed_from_case_context=False)

    # 5. model slot correction if any
    if ref_model_corrected_slots:
        reference_slots.update(ref_model_corrected_slots)
    if cand_model_corrected_slots:
        candidate_slots.update(cand_model_corrected_slots)

    # 6. match
    slot_match_results = match_slots(reference_slots, candidate_slots, normalized_case_context)
    valid_slots = build_valid_slots(reference_slots, normalized_case_context)

    # 7. major pathway / contradiction
    major_correct_pathways = detect_major_correct_pathways(slot_match_results, valid_slots=valid_slots)
    major_contradictions = detect_major_contradictions(reference_slots, candidate_slots)

    # 8. score
    score_info = calc_score(
        slot_match_results,
        major_correct_pathways,
        major_contradictions,
        valid_slots=valid_slots,
    )

    # 9. output
    return {
        "normalized_case_context": normalized_case_context,
        "reference_slots": reference_slots,
        "candidate_slots": candidate_slots,
        "slot_match_results": slot_match_results,
        "major_correct_pathways": major_correct_pathways,
        "major_contradictions": major_contradictions,
        "valid_slots": valid_slots,
        "final_judgment": final_judgment,
        "clinical_reasoning_summary": clinical_reasoning_summary or [],
        "base_score": score_info["base_score"],
        "bonus": score_info["bonus"],
        "penalty": score_info["penalty"],
        "final_score": score_info["final_score"],
        "score_breakdown": score_info["score_breakdown"]
    }


def detect_major_correct_pathways(slot_match_results, valid_slots: list[str] | None = None):
    """
    Top-level helper for major pathway correctness.
    Keep this globally accessible for rule-flow scoring wrappers.
    """
    major_slots = [
        "disease_stage",
        "resectability",
        "driver_status",
        "therapy_line",
        "treatment_intent",
        "regimen_class",
    ]
    valid_set = set(valid_slots or [])
    res: list[str] = []
    for slot in major_slots:
        if valid_set and slot not in valid_set:
            continue
        mt = _safe_str((slot_match_results.get(slot) or {}).get("match_type")) or "missing"
        if mt in {"exact", "hierarchical_equivalent", "clinically_acceptable"}:
            res.append(slot)
    return res


def _extract_cdss_text_for_rule_f1(case_item: dict[str, Any], *, is_gt: bool) -> str:
    if not isinstance(case_item, dict):
        return ""

    # Prefer explicit cdss_result fields first.
    key_order = ["cdss_result", "treatment", "final_answer"]
    if is_gt:
        key_order = ["cdss_result", "final_answer", "treatment"]

    for key in key_order:
        text = _safe_str(case_item.get(key))
        if text:
            parsed_cdss, _ = _extract_embedded_answer_fields(text)
            if parsed_cdss:
                return parsed_cdss
            return text

    # Compatibility with metrics case format.
    base_model_cdss = case_item.get("base_model_cdss")
    if isinstance(base_model_cdss, dict):
        text = _safe_str(base_model_cdss.get("cdss_result"))
        if text:
            return text

    return ""


def _coerce_case_map_for_rule_f1(payload: Any, *, is_gt: bool) -> dict[str, dict[str, Any]]:
    if is_gt:
        raw_cases = _extract_cases(payload)
        if isinstance(raw_cases, dict):
            mapped: dict[str, dict[str, Any]] = {}
            for key, item in raw_cases.items():
                if not isinstance(item, dict):
                    continue
                cid = _safe_str(item.get("case_id")) or _safe_str(key)
                if cid:
                    mapped[cid] = item
            if mapped:
                return mapped
    else:
        pred_cases = _extract_pred_cases(payload)
        if pred_cases:
            return pred_cases

    if isinstance(payload, dict):
        cid = (
            _safe_str(payload.get("case_id"))
            or _safe_str(payload.get("id"))
            or _safe_str(payload.get("病例ID"))
            or _safe_str(payload.get("user"))
            or "single_case"
        )
        return {cid: payload}

    return {}


def _select_case_pair_for_rule_f1(
    candidate_cases: dict[str, dict[str, Any]],
    reference_cases: dict[str, dict[str, Any]],
    case_id: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    target_case_id = _safe_str(case_id)
    if target_case_id:
        return (
            target_case_id,
            candidate_cases.get(target_case_id, {}),
            reference_cases.get(target_case_id, {}),
        )

    shared_ids = [cid for cid in candidate_cases.keys() if cid in reference_cases]
    if shared_ids:
        target_case_id = shared_ids[0]
    elif candidate_cases:
        target_case_id = next(iter(candidate_cases.keys()))
    elif reference_cases:
        target_case_id = next(iter(reference_cases.keys()))
    else:
        raise ValueError("病例与GT中均未找到可评估病例。")

    return (
        target_case_id,
        candidate_cases.get(target_case_id, {}),
        reference_cases.get(target_case_id, {}),
    )


def _merge_case_context_for_rule_f1(
    candidate_case: dict[str, Any],
    reference_case: dict[str, Any],
    selected_case_id: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    # Prefer GT context first; fill from candidate if missing.
    for source in (reference_case, candidate_case):
        if not isinstance(source, dict):
            continue
        for k, v in source.items():
            if k not in merged or merged[k] in (None, "", [], {}):
                if v not in (None, "", [], {}):
                    merged[k] = v

    merged["case_id"] = _safe_str(merged.get("case_id")) or selected_case_id
    return merged


def compute_case_final_score_by_rule_flow(
    case_json: dict[str, Any],
    reference_answer: str,
    candidate_answer: str,
) -> dict[str, Any]:
    """
    Rule-flow scoring wrapper that follows the exact pipeline:
    normalize -> concept extraction -> slot mapping -> slot matching
    -> major pathway / contradiction -> final score.
    """
    reference_answer = _safe_str(reference_answer)
    candidate_answer = _safe_str(candidate_answer)
    if not reference_answer:
        raise ValueError("reference_answer 为空，无法计算 final_score。")
    if not candidate_answer:
        raise ValueError("candidate_answer 为空，无法计算 final_score。")

    normalized_case_context = normalize_case_context(case_json)

    reference_concepts = extract_concepts_rule_based(reference_answer)
    candidate_concepts = extract_concepts_rule_based(candidate_answer)

    reference_slots = concepts_to_slots(reference_concepts, normalized_case_context, seed_from_case_context=False)
    candidate_slots = concepts_to_slots(candidate_concepts, normalized_case_context, seed_from_case_context=False)

    slot_match_results = match_slots(reference_slots, candidate_slots, normalized_case_context)
    valid_slots = build_valid_slots(reference_slots, normalized_case_context)

    major_correct_pathways = detect_major_correct_pathways(slot_match_results, valid_slots=valid_slots)
    major_contradictions = detect_major_contradictions(reference_slots, candidate_slots)

    score_info = calc_score(
        slot_match_results,
        major_correct_pathways,
        major_contradictions,
        valid_slots=valid_slots,
    )

    final_score = _safe_float(score_info.get("final_score")) or 0.0
    return {
        "normalized_case_context": normalized_case_context,
        "reference_concepts": reference_concepts,
        "candidate_concepts": candidate_concepts,
        "reference_slots": reference_slots,
        "candidate_slots": candidate_slots,
        "slot_match_results": slot_match_results,
        "valid_slots": valid_slots,
        "major_correct_pathways": major_correct_pathways,
        "major_contradictions": major_contradictions,
        "score_info": score_info,
        "final_score": final_score,
        "f1_score": final_score,
    }


def compute_case_final_score_from_case_and_gt_files(
    case_file: str | Path,
    gt_file: str | Path,
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    """
    Public entry:
    Pass in a case file + GT file, then compute final_score (F1) for one case.
    """
    case_path = Path(case_file).resolve()
    gt_path = Path(gt_file).resolve()
    if not case_path.exists():
        raise FileNotFoundError(f"病例文件不存在: {case_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT文件不存在: {gt_path}")

    case_payload = _safe_read_json(case_path)
    gt_payload = _safe_read_json(gt_path)

    candidate_cases = _coerce_case_map_for_rule_f1(case_payload, is_gt=False)
    reference_cases = _coerce_case_map_for_rule_f1(gt_payload, is_gt=True)

    selected_case_id, candidate_case, reference_case = _select_case_pair_for_rule_f1(
        candidate_cases,
        reference_cases,
        case_id=case_id,
    )

    candidate_answer = _extract_cdss_text_for_rule_f1(candidate_case, is_gt=False)
    reference_answer = _extract_cdss_text_for_rule_f1(reference_case, is_gt=True)
    if not candidate_answer and isinstance(candidate_case, dict):
        # Keep compatibility with normalized pred structure from _extract_pred_cases.
        candidate_answer = _safe_str((_extract_pred_cdss(candidate_case) or {}).get("cdss_result"))

    merged_case_json = _merge_case_context_for_rule_f1(
        candidate_case=candidate_case,
        reference_case=reference_case,
        selected_case_id=selected_case_id,
    )

    result = compute_case_final_score_by_rule_flow(
        case_json=merged_case_json,
        reference_answer=reference_answer,
        candidate_answer=candidate_answer,
    )
    result["case_id"] = selected_case_id
    result["case_file"] = str(case_path)
    result["gt_file"] = str(gt_path)
    return result


def f1score(
    case_file: str | Path,
    gt_file: str | Path,
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    """
    Minimal wrapper:
    pure call to judge_cdss_metrics rule-flow functions for one-case evaluation.
    """
    return compute_case_final_score_from_case_and_gt_files(
        case_file=case_file,
        gt_file=gt_file,
        case_id=case_id,
    )


def main() -> None:
    args = parse_args()

    gt_dir = Path(args.gt_dir)
    api_config_path = Path(args.api_config)
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT目录不存在: {gt_dir}")
    if not api_config_path.exists():
        raise FileNotFoundError(f"API配置不存在: {api_config_path}")

    judge_provider = _load_judge_provider(api_config_path)
    print(f"[INFO] Judge provider: {judge_provider.name} ({judge_provider.model})", flush=True)

    if args.mode == "single":
        pred_dir = Path(args.pred_dir)
        metrics_dir = Path(args.metrics_dir)
        _evaluate_experiment(
            experiment_name=metrics_dir.name or "single",
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            output_dir=metrics_dir,
            include=args.include,
            max_files=args.max_files,
            max_cases=args.max_cases,
            default_language=args.default_language,
            judge_provider=judge_provider,
            workers=args.workers,
        )
        print("\n[INFO] Done.", flush=True)
        return

    # paired 模式：两组实验分别输出到 测评/result/<实验组名>
    result_root = Path(args.result_root)
    experiments = [
        (str(args.exp1_name), Path(args.exp1_pred_dir)),
        (str(args.exp2_name), Path(args.exp2_pred_dir)),
    ]

    all_summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result_root": str(result_root.resolve()),
        "experiments": [],
    }

    for exp_name, exp_pred_dir in experiments:
        exp_output_dir = result_root / exp_name
        exp_summary = _evaluate_experiment(
            experiment_name=exp_name,
            pred_dir=exp_pred_dir,
            gt_dir=gt_dir,
            output_dir=exp_output_dir,
            include=args.include,
            max_files=args.max_files,
            max_cases=args.max_cases,
            default_language=args.default_language,
            judge_provider=judge_provider,
            workers=args.workers,
        )
        all_summary["experiments"].append(
            {
                "name": exp_name,
                "pred_dir": str(exp_pred_dir.resolve()),
                "output_dir": str(exp_output_dir.resolve()),
                "record_count": len(exp_summary.get("records", [])),
            }
        )

    all_summary_path = result_root / "summary_all.json"
    _write_json(all_summary_path, all_summary)
    print(f"\n[INFO] Saved paired summary: {all_summary_path}", flush=True)
    print("[INFO] Done.", flush=True)


if __name__ == "__main__":
    main()

