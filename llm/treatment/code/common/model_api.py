from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
try:
    import importlib

    yaml = importlib.import_module("yaml")
except Exception:  # pragma: no cover
    yaml = None

try:
    from zai import ZaiClient
except Exception:  # pragma: no cover
    ZaiClient = None


# Fallback defaults when api_config.yaml is missing or incomplete.
DEFAULT_BASE_URL = "https://api.apiyi.com/v1"
DEFAULT_API_KEY = ""

GLM_BASE_URL = "https://api.z.ai/api/paas/v4/"
GLM_API_KEY = ""

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_API_KEY = ""

PROJECT_ROOT = Path(__file__).resolve().parents[3]
THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def _candidate_api_config_paths() -> list[Path]:
    env_path = str(os.getenv("API_CONFIG_PATH", "")).strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.append(PROJECT_ROOT / "api_config.yaml")
    candidates.append(PROJECT_ROOT / "code" / "code" / "api_config.yaml")
    candidates.append(PROJECT_ROOT / "code" / "api_config.yaml")

    uniq: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            key = str(path.resolve()) if path.exists() else str(path)
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(path)
    return uniq


def _load_api_config_dict() -> dict[str, Any]:
    if yaml is None:
        return {}
    for path in _candidate_api_config_paths():
        try:
            if not path.exists() or not path.is_file():
                continue
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def _provider_endpoint_from_config(provider_name: str, cfg: dict[str, Any]) -> tuple[str, str] | None:
    if not provider_name or not isinstance(cfg, dict):
        return None
    providers = cfg.get("providers")
    if not isinstance(providers, dict):
        return None
    provider = providers.get(provider_name)
    if not isinstance(provider, dict):
        return None

    base_url = str(provider.get("base_url", "")).strip()
    api_key = str(provider.get("api_key", "")).strip()
    if base_url and api_key:
        return base_url, api_key
    return None


def endpoint_from_provider(provider_name: str) -> tuple[str, str] | None:
    cfg = _load_api_config_dict()
    return _provider_endpoint_from_config(provider_name, cfg)


def extract_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def extract_reasoning_content(message: dict[str, Any]) -> str:
    candidates = [
        message.get("reasoning_content"),
        message.get("reasoning"),
        message.get("thinking"),
    ]
    for item in candidates:
        if item is None:
            continue
        text = extract_message_content(item).strip()
        if text:
            return text
    return ""


def extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw, re.IGNORECASE)
    if fenced:
        raw = fenced.group(1)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def split_think_from_text(text: str) -> tuple[str, str]:
    raw = str(text or "")
    thinks: list[str] = []
    for m in THINK_BLOCK_RE.finditer(raw):
        content = str(m.group(1) or "").strip()
        if content:
            thinks.append(content)
    cleaned = THINK_BLOCK_RE.sub("", raw).strip()
    return cleaned, "\n\n".join(thinks)


def is_glm46v_model(model_name: str) -> bool:
    name = str(model_name or "").strip().lower()
    return name in {"glm-4.6v", "glm4.6v", "glm-4-6v"}


def is_glm_model(model_name: str) -> bool:
    name = str(model_name or "").strip().lower()
    return name.startswith("glm")


def is_qwen397_model(model_name: str) -> bool:
    name = str(model_name or "").strip().lower()
    return "qwen3.5-397b-a17b" in name


def is_kimi_model(model_name: str) -> bool:
    name = str(model_name or "").strip().lower()
    return "kimi-k2.5" in name or name.startswith("kimi") or "moonshotai/kimi" in name


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


def _routing_provider_names(cfg: dict[str, Any]) -> tuple[str, str, str, str]:
    routing = cfg.get("routing") if isinstance(cfg, dict) else {}
    if not isinstance(routing, dict):
        routing = {}

    active_provider = str(cfg.get("active_provider", "")).strip() if isinstance(cfg, dict) else ""
    default_provider = str(routing.get("default_provider", "")).strip() or active_provider or "default_llm"
    glm_provider = str(routing.get("glm_provider", "")).strip() or "glm_route"
    qwen_provider = str(routing.get("qwen397_provider", "")).strip() or "qwen_route"
    kimi_provider = str(routing.get("kimi_provider", "")).strip() or "kimi_route"
    return default_provider, glm_provider, qwen_provider, kimi_provider


def endpoint_for_model(
    model_name: str,
    *,
    route_qwen_to_siliconflow: bool = True,
    route_kimi_to_siliconflow: bool = False,
) -> tuple[str, str]:
    cfg = _load_api_config_dict()
    default_provider, glm_provider, qwen_provider, kimi_provider = _routing_provider_names(cfg)

    if is_glm_model(model_name):
        endpoint = _provider_endpoint_from_config(glm_provider, cfg)
        if endpoint:
            return endpoint

    if route_qwen_to_siliconflow and is_qwen397_model(model_name):
        endpoint = _provider_endpoint_from_config(qwen_provider, cfg)
        if endpoint:
            return endpoint

    if route_kimi_to_siliconflow and is_kimi_model(model_name):
        endpoint = _provider_endpoint_from_config(kimi_provider, cfg)
        if endpoint:
            return endpoint

    # Fallback sequence: routing default -> active provider -> provider with same model name
    candidates = [default_provider]
    active_provider = str(cfg.get("active_provider", "")).strip() if isinstance(cfg, dict) else ""
    if active_provider:
        candidates.append(active_provider)
    candidates.append(str(model_name or "").strip())

    seen: set[str] = set()
    for provider_name in candidates:
        if not provider_name or provider_name in seen:
            continue
        seen.add(provider_name)
        endpoint = _provider_endpoint_from_config(provider_name, cfg)
        if endpoint:
            return endpoint

    # Ultimate fallback to hardcoded defaults.
    if is_glm_model(model_name):
        return GLM_BASE_URL, GLM_API_KEY
    if route_qwen_to_siliconflow and is_qwen397_model(model_name):
        return SILICONFLOW_BASE_URL, SILICONFLOW_API_KEY
    if route_kimi_to_siliconflow and is_kimi_model(model_name):
        return SILICONFLOW_BASE_URL, SILICONFLOW_API_KEY
    return DEFAULT_BASE_URL, DEFAULT_API_KEY


def endpoint_for_model_with_env(
    model_name: str,
    *,
    route_qwen_to_siliconflow: bool = True,
    route_kimi_to_siliconflow: bool = False,
) -> tuple[str, str]:
    base_url, api_key = endpoint_for_model(
        model_name,
        route_qwen_to_siliconflow=route_qwen_to_siliconflow,
        route_kimi_to_siliconflow=route_kimi_to_siliconflow,
    )
    return os.getenv("LLM_BASE_URL", base_url), os.getenv("LLM_API_KEY", api_key)


def is_model_not_exist_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "model does not exist" in text
        or "'code': 20012" in text
        or '"code": 20012' in text
    )


def is_thinking_unsupported_error(exc: Exception) -> bool:
    text = str(exc or "")
    lower = text.lower()
    return (
        "unrecognized request argument supplied: thinking" in text
        or ("unsupported" in lower and "thinking" in lower)
        or ("unknown" in lower and "thinking" in lower)
    )


class UnifiedOpenAIClient:
    def __init__(self, api_key: str, base_url: str, timeout: int = 480) -> None:
        if OpenAI is None:
            raise RuntimeError("依赖缺失: openai。请先安装后再运行，例如: pip install openai")
        self.api_key = str(api_key or "")
        self.base_url = str(base_url or "").rstrip("/")
        self.timeout = int(timeout)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _candidate_models(self, model_name: str, alias_retry: bool) -> list[str]:
        if not alias_retry:
            return [model_name]
        if is_qwen397_model(model_name) and "siliconflow.cn" in self.base_url.lower():
            cands = qwen397_candidate_models(model_name)
            return cands or [model_name]
        return [model_name]

    def _invoke(
        self,
        payload: dict[str, Any],
        *,
        stream: bool,
        alias_retry: bool,
        thinking_fallback: bool,
    ) -> Any:
        req = dict(payload)
        model_name = str(req.get("model", "")).strip()
        candidates = self._candidate_models(model_name, alias_retry=alias_retry)
        last_exc: Exception | None = None

        for idx, candidate in enumerate(candidates):
            call_req = dict(req)
            call_req["model"] = candidate
            if stream:
                call_req["stream"] = True

            try:
                return self.client.chat.completions.create(**call_req)
            except Exception as exc:
                last_exc = exc

                if thinking_fallback and call_req.get("thinking") is not None and is_thinking_unsupported_error(exc):
                    fallback_req = dict(call_req)
                    fallback_req.pop("thinking", None)
                    try:
                        return self.client.chat.completions.create(**fallback_req)
                    except Exception as exc2:
                        last_exc = exc2

                has_next = idx < len(candidates) - 1
                if has_next and is_model_not_exist_error(last_exc):
                    print(f"[ModelFallback] 模型不存在，尝试别名: {candidate} -> {candidates[idx + 1]}")
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("模型调用失败：未知错误")

    def chat_completions(
        self,
        payload: dict[str, Any],
        *,
        alias_retry: bool = True,
        thinking_fallback: bool = False,
    ) -> dict[str, Any]:
        resp = self._invoke(
            payload,
            stream=False,
            alias_retry=alias_retry,
            thinking_fallback=thinking_fallback,
        )
        return resp.model_dump()

    def chat_completions_stream(
        self,
        payload: dict[str, Any],
        *,
        alias_retry: bool = True,
        thinking_fallback: bool = False,
    ) -> Any:
        return self._invoke(
            payload,
            stream=True,
            alias_retry=alias_retry,
            thinking_fallback=thinking_fallback,
        )


def collect_stream_text(stream: Any, *, capture_thinking: bool) -> tuple[str, str]:
    completion_parts: list[str] = []
    thinking_parts: list[str] = []

    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if not delta:
            continue
        if capture_thinking:
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                thinking_parts.append(str(reasoning))
        content = getattr(delta, "content", None)
        if content:
            completion_parts.append(str(content))

    final_text = "".join(completion_parts).strip()
    thinking_text = "".join(thinking_parts).strip()
    return final_text, thinking_text


def call_glm46v_via_zai(
    *,
    api_key: str,
    model_name: str,
    messages: list[dict[str, Any]],
    thinking_type: str = "enabled",
) -> tuple[str, str]:
    if ZaiClient is None:
        raise RuntimeError("未找到 ZaiClient。请安装 zai-sdk: pip install -U zai-sdk")

    client = ZaiClient(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        thinking={"type": str(thinking_type or "enabled")},
    )

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError(f"glm-4.6v 响应无 choices: {response}")
    message_obj = getattr(choices[0], "message", None)
    if message_obj is None:
        raise RuntimeError(f"glm-4.6v 响应无 message: {response}")

    if isinstance(message_obj, dict):
        message_dict = message_obj
    elif hasattr(message_obj, "model_dump"):
        message_dict = message_obj.model_dump()
    else:
        message_dict = {
            "content": getattr(message_obj, "content", ""),
            "reasoning_content": getattr(message_obj, "reasoning_content", ""),
        }

    final_text = extract_message_content(message_dict.get("content", "")).strip()
    reasoning_text = extract_message_content(message_dict.get("reasoning_content", "")).strip()
    return final_text, reasoning_text


def chat_content_with_retry(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: int = 60,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    alias_retry: bool = True,
) -> str:
    client = UnifiedOpenAIClient(api_key=api_key, base_url=base_url, timeout=timeout)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": int(max_tokens),
    }

    attempts = max(1, int(max_retries))
    backoff = float(retry_backoff_seconds)
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            resp = client.chat_completions(payload, alias_retry=alias_retry, thinking_fallback=False)
            choices = resp.get("choices", [])
            if not choices:
                raise RuntimeError(f"响应缺少 choices: {resp}")
            message = choices[0].get("message", {})
            return extract_message_content(message.get("content", ""))
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            time.sleep(backoff * (2 ** (attempt - 1)))

    raise RuntimeError(f"调用失败（重试{attempts}次）: {last_exc}")




