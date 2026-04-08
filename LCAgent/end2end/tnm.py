import contextlib
import io
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import requests
import yaml
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader


class TNMState(TypedDict, total=False):
    case_id: str
    benchmark_key: str
    language: str
    query: str
    input_file_name: str
    input_file_exists: bool
    document_text: str
    doc_error: str
    extract_raw: str
    clean_text: str
    nm_split_raw: str
    nm_split_clean: str
    n_context: str
    m_context: str
    n_descriptions: str
    m_descriptions: str
    t_extract_raw: str
    t_extract_clean: str
    t_descriptions: str
    t_context: str
    t_stage_raw: str
    n_stage_raw: str
    m_stage_raw: str
    t_json_clean: str
    n_json_clean: str
    m_json_clean: str
    final_result: str
    T: str
    N: str
    M: str
    tnm_stage: str
    final_answer: str


API_CONFIG_PATH = Path("api_config.yaml")
DIFY_WORKFLOW_PATH = Path("files/tnm分期助手_分开_格式化.yml")


def _language_instruction(language: str) -> str:
    lang = str(language or "").strip().lower()
    if lang == "english":
        return (
            "All outputs must be in English only. Do not output any Chinese characters. "
            "Use English for headings, explanations, JSON values, and staging descriptions. "
            "If any step outputs structured JSON, keep JSON keys in the original schema language and only make the values English."
        )
    return "请全程使用中文输出。"


def set_api_config_path(path: str) -> None:
    global API_CONFIG_PATH
    API_CONFIG_PATH = Path(path)


def _load_api_config() -> dict[str, str]:
    if not API_CONFIG_PATH.exists():
        raise RuntimeError(f"未找到 API 配置文件: {API_CONFIG_PATH}")

    raw = API_CONFIG_PATH.read_text(encoding="utf-8")


    try:
        config = yaml.safe_load(raw)
    except Exception as exc:
        raise RuntimeError(f"API 配置文件不是有效 YAML/JSON: {exc}") from exc

    if not isinstance(config, dict):
        raise RuntimeError("API 配置文件格式错误，顶层必须是对象。")

    # 兼容两种结构：
    # 1) 平铺: api_key/api_url(or base_url)/model
    # 2) providers: active_provider + providers.<name>.{api_key,api_url|base_url,model}
    selected: dict[str, Any] = config
    active_provider = str(os.environ.get("ACTIVE_PROVIDER_OVERRIDE", "")).strip()
    if not active_provider:
        active_provider = str(config.get("active_provider", "")).strip()
    providers = config.get("providers")
    if active_provider and isinstance(providers, dict):
        provider_conf = providers.get(active_provider)
        if not isinstance(provider_conf, dict):
            raise RuntimeError(
                f"active_provider={active_provider} 未在 providers 中找到有效配置: {API_CONFIG_PATH}"
            )
        selected = provider_conf

    api_key = str(selected.get("api_key", "")).strip()
    api_url = str(selected.get("api_url", selected.get("base_url", "https://api.apiyi.com/v1"))).strip()
    model = str(selected.get("model", "gpt-4o")).strip()

    if not api_key:
        raise RuntimeError(f"API 配置文件缺少 api_key: {API_CONFIG_PATH}")
    if not api_url:
        raise RuntimeError(f"API 配置文件缺少 api_url: {API_CONFIG_PATH}")
    if not model:
        raise RuntimeError(f"API 配置文件缺少 model: {API_CONFIG_PATH}")

    return {"api_key": api_key, "api_url": api_url, "model": model}


def _call_model(system_prompt: str, user_prompt: str) -> str:
    config = _load_api_config()
    api_key = config["api_key"]
    api_url = config["api_url"]
    model = config["model"]
    

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    errors: list[str] = []
    for endpoint in _candidate_endpoints(api_url):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=180)
            if response.status_code >= 400:
                errors.append(f"{endpoint} -> HTTP {response.status_code}: {response.text[:300]}")
                continue

            try:
                data = response.json()
            except Exception as exc:
                errors.append(f"{endpoint} -> 响应非 JSON: {exc}; 原始响应: {response.text[:300]}")
                continue

            content = _extract_content(data)
            if content:
                return content
            errors.append(f"{endpoint} -> JSON 无可解析内容: {json.dumps(data, ensure_ascii=False)[:300]}")
        except Exception as exc:
            errors.append(f"{endpoint} -> 请求异常: {exc}")

    raise RuntimeError("; ".join(errors)[:1200])


def _candidate_endpoints(api_url: str) -> list[str]:
    url = api_url.rstrip("/")
    candidates = []
    if url.endswith("/token"):
        base = url[: -len("/token")]
        candidates.append(f"{base}/v1/chat/completions")
        candidates.append(f"{base}/chat/completions")
    elif url.endswith("/v1"):
        candidates.append(f"{url}/chat/completions")
    candidates.append(url)
    return list(dict.fromkeys(candidates))


def _extract_content(data: Any) -> str:
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
                    chunks = []
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
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

    nested = data.get("data")
    if isinstance(nested, dict):
        for key in ("content", "result", "answer", "output"):
            value = nested.get(key)
            if isinstance(value, str):
                return value

    return ""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {"raw": text}


def _read_pdf_text(file_name: str, max_chars: int = 20000) -> str:
    file_path = Path(file_name)
    if not file_path.exists():
        raise FileNotFoundError(f"病例文件不存在: {file_name}")

    reader = PdfReader(str(file_path))
    texts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            texts.append(page_text)

    full_text = "\n".join(texts).strip()
    if not full_text:
        raise RuntimeError("PDF 文本提取结果为空")
    return full_text[:max_chars]


@lru_cache(maxsize=1)
def _load_dify_workflow() -> dict[str, Any]:
    if not DIFY_WORKFLOW_PATH.exists():
        raise RuntimeError(f"未找到 Dify YAML 文件: {DIFY_WORKFLOW_PATH}")
    raw = DIFY_WORKFLOW_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise RuntimeError("Dify YAML 格式错误，顶层必须为对象")
    return data


@lru_cache(maxsize=1)
def _dify_nodes_index() -> dict[str, dict[str, Any]]:
    workflow = _load_dify_workflow()
    graph = workflow.get("workflow", {}).get("graph", {})
    nodes = graph.get("nodes", [])
    index: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if isinstance(node, dict):
            node_id = str(node.get("id", "")).strip()
            if node_id:
                index[node_id] = node
    return index


def _get_node_data(node_id: str) -> dict[str, Any]:
    node = _dify_nodes_index().get(node_id)
    if not isinstance(node, dict):
        raise RuntimeError(f"YAML 中未找到节点: {node_id}")
    data = node.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"节点 {node_id} 缺少 data")
    return data


def _get_llm_prompt(node_id: str, role: str) -> str:
    data = _get_node_data(node_id)
    prompt_template = data.get("prompt_template", [])
    if not isinstance(prompt_template, list):
        return ""
    for item in prompt_template:
        if isinstance(item, dict) and str(item.get("role", "")).strip() == role:
            return str(item.get("text", ""))
    return ""


def _get_code_script(node_id: str) -> str:
    data = _get_node_data(node_id)
    return str(data.get("code", ""))


def _get_answer_template() -> str:
    data = _get_node_data(N_ANSWER)
    return str(data.get("answer", ""))


def _template_values(state: TNMState) -> dict[str, str]:
    return {
        "sys.query": str(state.get("query", "")),
        f"{N_DOC}.text": str(state.get("document_text", "")),
        f"{N_FILTER_THINK}.result": str(state.get("clean_text", "")),
        f"{N_NM_SPLIT}.text": str(state.get("nm_split_raw", "")),
        f"{N_FILTER_THINK_NM}.result": str(state.get("nm_split_clean", "")),
        f"{N_CODE_7}.n_descriptions": str(state.get("n_descriptions", "")),
        f"{N_CODE_7}.m_descriptions": str(state.get("m_descriptions", "")),
        f"{N_T_EXTRACT}.text": str(state.get("t_extract_raw", "")),
        f"{N_FILTER_THINK_T}.result": str(state.get("t_extract_clean", "")),
        f"{N_CODE_9}.t_descriptions": str(state.get("t_descriptions", "")),
        f"{N_T_STAGE}.text": str(state.get("t_stage_raw", "")),
        f"{N_N_STAGE}.text": str(state.get("n_stage_raw", "")),
        f"{N_M_STAGE}.text": str(state.get("m_stage_raw", "")),
        f"{N_T_CODE}.result": str(state.get("t_json_clean", "")),
        f"{N_N_CODE}.result": str(state.get("n_json_clean", "")),
        f"{N_M_CODE}.result": str(state.get("m_json_clean", "")),
        f"{N_FINAL_CODE}.result": str(state.get("final_result", "")),
    }


def _render_dify_template(template: str, state: TNMState) -> str:
    values = _template_values(state)

    def replacer(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return values.get(key, "")

    return re.sub(r"\{\{#([^#]+)#\}\}", replacer, template)


def _execute_dify_code_node(node_id: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    script = _get_code_script(node_id)
    if not script.strip():
        return {}

    scope: dict[str, Any] = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(script, scope)
    main_func = scope.get("main")
    if not callable(main_func):
        raise RuntimeError(f"节点 {node_id} 代码缺少 main 函数")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = main_func(**kwargs)
    if not isinstance(result, dict):
        raise RuntimeError(f"节点 {node_id} main 返回值必须是 dict")
    return result


# 节点函数
def document_extractor(state: TNMState) -> dict[str, Any]:
    file_name = str(state.get("input_file_name", "")).strip()
    if not file_name:
        return {"doc_error": "缺少 input_file_name", "document_text": ""}

    try:
        document_text = _read_pdf_text(file_name)
        return {"document_text": document_text, "doc_error": ""}
    except Exception as exc:
        return {"document_text": "", "doc_error": str(exc)}


def extract_info_llm(state: TNMState) -> dict[str, Any]:
    document_text = str(state.get("document_text", "")).strip()
    if not document_text:
        return {"extract_raw": state.get("doc_error", "文档为空")}

    system_prompt = _get_llm_prompt(N_EXTRACT, "system")
    user_prompt = _render_dify_template(_get_llm_prompt(N_EXTRACT, "user"), state)
    try:
        raw = _call_model(system_prompt, user_prompt)
        return {"extract_raw": raw}
    except Exception as exc:
        return {"extract_raw": f"模型调用失败: {exc}"}


def filter_think(state: TNMState) -> dict[str, Any]:
    result = _execute_dify_code_node(N_FILTER_THINK, {"arg1": str(state.get("extract_raw", ""))})
    return {"clean_text": str(result.get("result", ""))}


def nm_distributor_llm(state: TNMState) -> dict[str, Any]:
    system_prompt = _get_llm_prompt(N_NM_SPLIT, "system")
    user_prompt = _render_dify_template(_get_llm_prompt(N_NM_SPLIT, "user"), state)
    try:
        raw = _call_model(system_prompt, user_prompt)
        return {"nm_split_raw": raw}
    except Exception as exc:
        return {"nm_split_raw": f"模型调用失败: {exc}"}


def filter_think_nm(state: TNMState) -> dict[str, Any]:
    result = _execute_dify_code_node(N_FILTER_THINK_NM, {"arg1": str(state.get("nm_split_raw", ""))})
    return {"nm_split_clean": str(result.get("result", ""))}


def code_exec_7(state: TNMState) -> dict[str, Any]:
    result = _execute_dify_code_node(N_CODE_7, {"arg1": str(state.get("nm_split_clean", ""))})
    return {
        "n_descriptions": str(result.get("n_descriptions", "")),
        "m_descriptions": str(result.get("m_descriptions", "")),
    }


def t_extract_llm(state: TNMState) -> dict[str, Any]:
    system_prompt = _get_llm_prompt(N_T_EXTRACT, "system")
    user_prompt = _render_dify_template(_get_llm_prompt(N_T_EXTRACT, "user"), state)
    try:
        raw = _call_model(system_prompt, user_prompt)
        return {"t_extract_raw": raw}
    except Exception as exc:
        return {"t_extract_raw": f"模型调用失败: {exc}"}


def filter_think_t(state: TNMState) -> dict[str, Any]:
    result = _execute_dify_code_node(N_FILTER_THINK_T, {"arg1": str(state.get("t_extract_raw", ""))})
    return {"t_extract_clean": str(result.get("result", ""))}


def code_exec_9(state: TNMState) -> dict[str, Any]:
    result = _execute_dify_code_node(N_CODE_9, {"arg1": str(state.get("t_extract_clean", ""))})
    return {"t_descriptions": str(result.get("t_descriptions", ""))}


def t_staging_llm(state: TNMState) -> dict[str, Any]:
    system_prompt = _get_llm_prompt(N_T_STAGE, "system")
    user_prompt = _render_dify_template(_get_llm_prompt(N_T_STAGE, "user"), state)
    try:
        raw = _call_model(system_prompt, user_prompt)
        return {"t_stage_raw": raw}
    except Exception as exc:
        return {"t_stage_raw": f"模型调用失败: {exc}"}


def n_staging_llm(state: TNMState) -> dict[str, Any]:
    system_prompt = _get_llm_prompt(N_N_STAGE, "system")
    user_prompt = _render_dify_template(_get_llm_prompt(N_N_STAGE, "user"), state)
    try:
        raw = _call_model(system_prompt, user_prompt)
        return {"n_stage_raw": raw}
    except Exception as exc:
        return {"n_stage_raw": f"模型调用失败: {exc}"}


def m_staging_llm(state: TNMState) -> dict[str, Any]:
    system_prompt = _get_llm_prompt(N_M_STAGE, "system")
    user_prompt = _render_dify_template(_get_llm_prompt(N_M_STAGE, "user"), state)
    try:
        raw = _call_model(system_prompt, user_prompt)
        return {"m_stage_raw": raw}
    except Exception as exc:
        return {"m_stage_raw": f"模型调用失败: {exc}"}


def t_code(state: TNMState) -> dict[str, Any]:
    result = _execute_dify_code_node(N_T_CODE, {"arg1": str(state.get("t_stage_raw", ""))})
    return {"t_json_clean": str(result.get("result", ""))}


def n_code(state: TNMState) -> dict[str, Any]:
    result = _execute_dify_code_node(N_N_CODE, {"arg1": str(state.get("n_stage_raw", ""))})
    return {"n_json_clean": str(result.get("result", ""))}


def m_code(state: TNMState) -> dict[str, Any]:
    result = _execute_dify_code_node(N_M_CODE, {"arg1": str(state.get("m_stage_raw", ""))})
    return {"m_json_clean": str(result.get("result", ""))}


def final_staging_code(state: TNMState) -> dict[str, Any]:
    result = _execute_dify_code_node(
        N_FINAL_CODE,
        {
            "t": str(state.get("t_json_clean", "")),
            "n": str(state.get("n_json_clean", "")),
            "m": str(state.get("m_json_clean", "")),
        },
    )
    return {"final_result": str(result.get("result", ""))}


def answer(state: TNMState) -> dict[str, Any]:
    t_obj = _extract_json(str(state.get("t_json_clean", "")))
    n_obj = _extract_json(str(state.get("n_json_clean", "")))
    m_obj = _extract_json(str(state.get("m_json_clean", "")))

    t_stage = str(t_obj.get("final_t_stage", "")).strip()
    n_stage = str(n_obj.get("final_n_stage", "")).strip()
    m_stage = str(m_obj.get("final_m_stage", "")).strip()

    answer_template = _get_answer_template()
    final_answer = _render_dify_template(answer_template, state).strip()

    tnm_stage = ""
    if t_stage or n_stage or m_stage:
        tnm_stage = f"{t_stage}{n_stage}{m_stage}"

    return {
        "T": t_stage,
        "N": n_stage,
        "M": m_stage,
        "tnm_stage": tnm_stage,
        "final_answer": final_answer,
    }


# 与 Dify 文件中的 node id 1:1 对齐
N_DOC = "1756804402464"
N_START = "1756804397059"
N_EXTRACT = "1758186451078"
N_T_STAGE = "1758861864675"
N_N_STAGE = "1758861867069"
N_M_STAGE = "1758861898205"
N_FILTER_THINK = "1764323445709"
N_T_CODE = "1764323555201"
N_N_CODE = "17643235703550"
N_M_CODE = "17643235815850"
N_FINAL_CODE = "1768228656389"
N_NM_SPLIT = "1769599924411"
N_FILTER_THINK_NM = "1769603565594"
N_CODE_7 = "1769604169192"
N_T_EXTRACT = "1770127326622"
N_FILTER_THINK_T = "1770127431059"
N_CODE_9 = "1770127481088"
N_ANSWER = "answer"

NODE_TITLES = {
    N_START: "开始",
    N_DOC: "文档提取器",
    N_EXTRACT: "提取信息",
    N_FILTER_THINK: "过滤think",
    N_NM_SPLIT: "N.M分流节点",
    N_FILTER_THINK_NM: "过滤think-NM",
    N_CODE_7: "代码执行 7",
    N_N_STAGE: "N分期",
    N_M_STAGE: "M分期",
    N_N_CODE: "N",
    N_M_CODE: "M",
    N_T_EXTRACT: "T提取",
    N_FILTER_THINK_T: "过滤think-t",
    N_CODE_9: "代码执行 9",
    N_T_STAGE: "T分期",
    N_T_CODE: "T",
    N_FINAL_CODE: "代码执行 5",
    N_ANSWER: "直接回复",
}

DIFY_EDGES = [
    (N_START, N_DOC),
    (N_DOC, N_EXTRACT),
    (N_T_STAGE, N_T_CODE),
    (N_N_STAGE, N_N_CODE),
    (N_M_STAGE, N_M_CODE),
    (N_T_CODE, N_FINAL_CODE),
    (N_N_CODE, N_FINAL_CODE),
    (N_M_CODE, N_FINAL_CODE),
    (N_FINAL_CODE, N_ANSWER),
    (N_EXTRACT, N_FILTER_THINK),
    (N_FILTER_THINK, N_NM_SPLIT),
    (N_NM_SPLIT, N_FILTER_THINK_NM),
    (N_FILTER_THINK_NM, N_CODE_7),
    (N_CODE_7, N_N_STAGE),
    (N_CODE_7, N_M_STAGE),
    (N_FILTER_THINK, N_T_EXTRACT),
    (N_T_EXTRACT, N_FILTER_THINK_T),
    (N_FILTER_THINK_T, N_CODE_9),
    (N_CODE_9, N_T_STAGE),
]


def draw_dify_mermaid() -> str:
    lines = ["graph TD;"]
    for node_id, title in NODE_TITLES.items():
        lines.append(f'    {node_id}["{title}"]')
    for source, target in DIFY_EDGES:
        lines.append(f"    {source} --> {target};")
    return "\n".join(lines)


def draw_dify_png(png_path: Path) -> None:
    import pygraphviz as pgv

    graph = pgv.AGraph(directed=True)
    graph.graph_attr.update(fontname="Noto Sans CJK SC")
    graph.node_attr.update(fontname="Noto Sans CJK SC")
    graph.edge_attr.update(fontname="Noto Sans CJK SC")
    for node_id, title in NODE_TITLES.items():
        graph.add_node(node_id, label=title)
    for source, target in DIFY_EDGES:
        graph.add_edge(source, target)
    graph.layout("dot")
    graph.draw(str(png_path))


workflow = StateGraph(TNMState)

# 添加节点（与 YAML 节点一一对应）
workflow.add_node(N_DOC, document_extractor)
workflow.add_node(N_EXTRACT, extract_info_llm)
workflow.add_node(N_T_STAGE, t_staging_llm)
workflow.add_node(N_N_STAGE, n_staging_llm)
workflow.add_node(N_M_STAGE, m_staging_llm)
workflow.add_node(N_FILTER_THINK, filter_think)
workflow.add_node(N_T_CODE, t_code)
workflow.add_node(N_N_CODE, n_code)
workflow.add_node(N_M_CODE, m_code)
workflow.add_node(N_FINAL_CODE, final_staging_code)
workflow.add_node(N_NM_SPLIT, nm_distributor_llm)
workflow.add_node(N_FILTER_THINK_NM, filter_think_nm)
workflow.add_node(N_CODE_7, code_exec_7)
workflow.add_node(N_T_EXTRACT, t_extract_llm)
workflow.add_node(N_FILTER_THINK_T, filter_think_t)
workflow.add_node(N_CODE_9, code_exec_9)
workflow.add_node(N_ANSWER, answer)

# 边构建（与 files/tnm分期助手_分开_格式化.yml 中 edges 1:1 对齐）
for source, target in DIFY_EDGES:
    if source == N_START:
        workflow.add_edge(START, target)
    else:
        workflow.add_edge(source, target)
workflow.add_edge(N_ANSWER, END)


app1 = workflow.compile()


def main() -> None:
    graph = app1.get_graph()
    print(draw_dify_mermaid())

    png_path = Path.cwd() / "tnm_graph.png"
    try:
        draw_dify_png(png_path)
        print(f"PNG 流程图已保存: {png_path}")
    except Exception as exc:
        print(f"Dify 风格 PNG 渲染失败: {exc}")
        try:
            png_data = graph.draw_png()
            if isinstance(png_data, (bytes, bytearray)):
                png_path.write_bytes(png_data)
            else:
                graph.draw_png(str(png_path))
            print(f"已使用 LangGraph 回退导出 PNG: {png_path}")
        except Exception as fallback_exc:
            print(f"PNG 渲染失败: {fallback_exc}")


if __name__ == "__main__":
    main()
