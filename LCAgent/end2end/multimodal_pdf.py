import base64
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
import yaml


DEFAULT_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2


class MultimodalPdfExtractor:
    def __init__(
        self,
        config_path: Path,
        provider: str,
        max_pages: int,
        max_chars: int,
        render_dpi: int,
    ) -> None:
        self.config_path = config_path
        self.provider = provider.strip()
        self.max_pages = max_pages
        self.max_chars = max_chars
        self.render_dpi = render_dpi
        self.cache: dict[str, str] = {}

    def extract_text(self, pdf_path: str, language: str = "") -> str:
        return self.extract_key_info(pdf_path, language=language)

    def extract_key_info(self, pdf_path: str, language: str = "") -> str:
        path = Path(pdf_path)
        key = str(path.resolve()) if path.exists() else str(path)
        if key in self.cache:
            return self.cache[key]

        if not path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        images = self._render_pdf_to_base64_png(path)
        if not images:
            raise RuntimeError(f"PDF 渲染失败或页面为空: {pdf_path}")

        page_infos: list[str] = []
        for idx, image_b64 in enumerate(images, start=1):
            info = self._analyze_single_page(image_b64, page_index=idx, page_count=len(images), language=language)
            info = str(info or "").strip()
            if info:
                page_infos.append(info)

        if not page_infos:
            raise RuntimeError(f"多模态 OCR 结果为空: {pdf_path}")

        merged_info = self._merge_page_infos(page_infos, language)
        final_text = str(merged_info or "").strip() or "\n\n".join(page_infos).strip()
        final_text = final_text[: self.max_chars]

        self.cache[key] = final_text
        return final_text

    def _load_api_config(self) -> dict[str, Any]:
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError(f"配置文件格式错误: {self.config_path}")

        active_provider = self.provider or str(os.environ.get("ACTIVE_PROVIDER_OVERRIDE", "")).strip()
        if not active_provider:
            active_provider = str(data.get("active_provider", "")).strip()

        selected: dict[str, Any] = data
        providers = data.get("providers")
        if active_provider and isinstance(providers, dict):
            provider_cfg = providers.get(active_provider)
            if not isinstance(provider_cfg, dict):
                raise RuntimeError(f"providers 中不存在有效 provider: {active_provider}")
            selected = provider_cfg

        api_key = str(selected.get("api_key", "")).strip()
        base_url = str(selected.get("base_url", selected.get("api_url", ""))).strip()
        model = str(selected.get("model", "")).strip()
        timeout = int(selected.get("timeout", DEFAULT_TIMEOUT))
        max_retries = int(selected.get("max_retries", DEFAULT_MAX_RETRIES))
        retry_backoff_seconds = float(selected.get("retry_backoff_seconds", DEFAULT_RETRY_BACKOFF_SECONDS))

        if not api_key:
            raise RuntimeError(f"配置缺少 api_key: {self.config_path}")
        if not base_url:
            raise RuntimeError(f"配置缺少 base_url/api_url: {self.config_path}")
        if not model:
            raise RuntimeError(f"配置缺少 model: {self.config_path}")

        return {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "timeout": timeout,
            "max_retries": max(1, max_retries),
            "retry_backoff_seconds": max(0.0, retry_backoff_seconds),
        }

    def _candidate_endpoints(self, base_url: str) -> list[str]:
        url = base_url.rstrip("/")
        candidates: list[str] = []
        if url.endswith("/v1"):
            candidates.append(f"{url}/chat/completions")
        candidates.append(url)
        if not url.endswith("/chat/completions"):
            candidates.append(f"{url}/chat/completions")
        return list(dict.fromkeys(candidates))

    def _extract_content(self, data: Any) -> str:
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            return ""

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        chunks: list[str] = []
                        for item in content:
                            if isinstance(item, dict):
                                text = item.get("text")
                                if isinstance(text, str):
                                    chunks.append(text)
                        if chunks:
                            return "\n".join(chunks)
                text = first.get("text")
                if isinstance(text, str):
                    return text

        for key in ("content", "result", "answer", "output"):
            value = data.get(key)
            if isinstance(value, str):
                return value
        return ""

    def _page_extract_prompt(self, language: str) -> str:
        if str(language).strip().lower() == "english":
            return (
                "Extract ONLY clinically relevant facts for lung-cancer TNM staging and CDSS treatment planning. "
                "Do not transcribe the full page. Keep findings concise and evidence-grounded. "
                "Output in English using this template:\n"
                "- Primary diagnosis and pathology:\n"
                "- TNM-related evidence (tumor size/invasion, nodal findings, metastasis):\n"
                "- Stage cues:\n"
                "- Biomarkers (driver genes, PD-L1):\n"
                "- Performance status / contraindications:\n"
                "- Previous and current treatment:\n"
                "- Key imaging/lab findings:\n"
                "- Important negatives / exclusions:\n"
                "Use 'Not mentioned' for missing fields."
            )
        return (
            "请仅提取对肺癌TNM分期和CDSS治疗决策有价值的信息，不要全文转写。"
            "请用下面模板输出：\n"
            "- 主要诊断与病理：\n"
            "- TNM相关证据（肿瘤大小/侵犯、淋巴结、远处转移）：\n"
            "- 分期线索：\n"
            "- 生物标志物（驱动基因、PD-L1）：\n"
            "- 体力状态/治疗禁忌：\n"
            "- 既往及当前治疗：\n"
            "- 关键影像和检验结果：\n"
            "- 重要阴性信息：\n"
            "若缺失请写“未提及”。"
        )

    def _merge_prompt(self, language: str, page_infos: list[str]) -> str:
        pages_blob = "\n\n".join(
            f"[Page {idx}]\n{text}" for idx, text in enumerate(page_infos, start=1)
        )

        if str(language).strip().lower() == "english":
            return (
                "Merge the following page-level extracted facts into one concise, de-duplicated summary for downstream "
                "TNM/CDSS workflows. Keep only actionable clinical facts, preserve uncertainty when present, and avoid "
                "hallucinations.\n\n"
                "Output format:\n"
                "- Final structured key facts:\n"
                "- TNM evidence summary:\n"
                "- CDSS decision-critical factors:\n"
                "- Missing information requiring manual review:\n\n"
                f"{pages_blob}"
            )

        return (
            "请把以下各页提取结果合并为去重后的“关键事实摘要”，用于TNM/CDSS下游工作流。"
            "只保留可用于决策的医学信息，存在不确定性时请保留原不确定表述，不要编造。\n\n"
            "输出格式：\n"
            "- 最终结构化关键事实：\n"
            "- TNM证据汇总：\n"
            "- CDSS决策关键因素：\n"
            "- 仍需人工核查的缺失信息：\n\n"
            f"{pages_blob}"
        )

    def _call_chat_completion(self, payload: dict[str, Any], cfg: dict[str, Any]) -> str:
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }

        errors: list[str] = []
        for endpoint in self._candidate_endpoints(cfg["base_url"]):
            for _ in range(cfg["max_retries"]):
                try:
                    resp = requests.post(endpoint, headers=headers, json=payload, timeout=cfg["timeout"])
                except Exception as exc:
                    errors.append(f"{endpoint} -> 请求异常: {exc}")
                    time.sleep(cfg["retry_backoff_seconds"])
                    continue

                if resp.status_code >= 400:
                    errors.append(f"{endpoint} -> HTTP {resp.status_code}: {resp.text[:300]}")
                    time.sleep(cfg["retry_backoff_seconds"])
                    continue

                try:
                    data = resp.json()
                except Exception as exc:
                    errors.append(f"{endpoint} -> 响应非 JSON: {exc}; 原始响应: {resp.text[:300]}")
                    time.sleep(cfg["retry_backoff_seconds"])
                    continue

                content = self._extract_content(data)
                if content:
                    return content

                errors.append(f"{endpoint} -> JSON 无可解析内容: {str(data)[:300]}")
                time.sleep(cfg["retry_backoff_seconds"])

        raise RuntimeError("; ".join(errors)[:1200])

    def _analyze_single_page(self, image_b64: str, page_index: int, page_count: int, language: str) -> str:
        cfg = self._load_api_config()

        prompt = self._page_extract_prompt(language)
        user_text = f"Page {page_index}/{page_count}. {prompt}"
        payload = {
            "model": cfg["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
        }
        return self._call_chat_completion(payload, cfg)

    def _merge_page_infos(self, page_infos: list[str], language: str) -> str:
        if not page_infos:
            return ""
        if len(page_infos) == 1:
            return page_infos[0]

        cfg = self._load_api_config()
        merge_prompt = self._merge_prompt(language, page_infos)
        payload = {
            "model": cfg["model"],
            "messages": [
                {
                    "role": "user",
                    "content": merge_prompt,
                }
            ],
            "temperature": 0,
        }
        return self._call_chat_completion(payload, cfg)

    def _render_pdf_to_base64_png(self, pdf_path: Path) -> list[str]:
        try:
            import fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "缺少 PyMuPDF 依赖，无法将 PDF 渲染为图片。请安装: pip install pymupdf"
            ) from exc

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            raise RuntimeError(f"无法打开 PDF: {pdf_path}") from exc

        images: list[str] = []
        scale = max(1, self.render_dpi) / 72.0
        matrix = fitz.Matrix(scale, scale)

        page_count = min(len(doc), max(1, self.max_pages))
        for i in range(page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            png_bytes = pix.tobytes("png")
            images.append(base64.b64encode(png_bytes).decode("ascii"))

        doc.close()
        return images


def detect_language_from_path(path: str) -> str:
    lower = str(path).lower()
    if re.search(r"(^|[/\\])english([/\\]|$)", lower):
        return "English"
    if re.search(r"(^|[/\\])chinese([/\\]|$)", lower):
        return "Chinese"
    return "Chinese"
