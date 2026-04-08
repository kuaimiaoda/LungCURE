import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict

import requests
import yaml
from langgraph.graph import END, START, StateGraph

try:
	from pypdf import PdfReader
except Exception:
	PdfReader = None


class CDSSState(TypedDict, total=False):
	query: str
	files: list[Any]
	TNM_ret: str
	node_outputs: dict[str, dict[str, Any]]
	final_answer: str
	doc_error: str


API_CONFIG_PATH = Path("api_config.yaml")
WORKFLOW_PATH = Path("files/非小细胞癌治疗-免提示.yml")


def set_api_config_path(path: str) -> None:
	global API_CONFIG_PATH
	API_CONFIG_PATH = Path(path)


def set_workflow_path(path: str) -> None:
	global WORKFLOW_PATH
	WORKFLOW_PATH = Path(path)


def _infer_language(state: CDSSState) -> str:
	query = str(state.get("query", ""))
	if "English" in query or "english" in query:
		return "english"
	files = state.get("files", [])
	if isinstance(files, list):
		for item in files:
			text = str(item)
			if "/English/" in text or "\\English\\" in text:
				return "english"
			if "/Chinese/" in text or "\\Chinese\\" in text:
				return "chinese"
	return "chinese"


def _language_instruction(language: str) -> str:
	lang = str(language or "").strip().lower()
	if lang == "english":
		return (
			"All outputs must be in English only. Do not output any Chinese characters. "
			"Use English for headings, explanations, JSON values, and final answers. "
			"If you output structured JSON for the first extraction node, keep the JSON keys exactly as the original Chinese schema and only make the values English."
		)
	return "请全程使用中文输出。"


def _load_yaml(path: Path) -> dict[str, Any]:
	if not path.exists():
		raise FileNotFoundError(f"文件不存在: {path}")
	data = yaml.safe_load(path.read_text(encoding="utf-8"))
	if not isinstance(data, dict):
		raise RuntimeError(f"YAML 顶层必须是对象: {path}")
	return data


def _workflow_data() -> dict[str, Any]:
	return _load_yaml(WORKFLOW_PATH)


def _graph_data() -> dict[str, Any]:
	data = _workflow_data()
	graph = data.get("workflow", {}).get("graph", {})
	if not isinstance(graph, dict):
		raise RuntimeError("workflow.graph 结构错误")
	return graph


def _nodes() -> list[dict[str, Any]]:
	nodes = _graph_data().get("nodes", [])
	if not isinstance(nodes, list):
		raise RuntimeError("workflow.graph.nodes 结构错误")
	return [n for n in nodes if isinstance(n, dict)]


def _edges() -> list[dict[str, Any]]:
	edges = _graph_data().get("edges", [])
	if not isinstance(edges, list):
		raise RuntimeError("workflow.graph.edges 结构错误")
	return [e for e in edges if isinstance(e, dict)]


def _node_index() -> dict[str, dict[str, Any]]:
	index: dict[str, dict[str, Any]] = {}
	for node in _nodes():
		node_id = str(node.get("id", "")).strip()
		if node_id:
			index[node_id] = node
	return index


def _start_node_id() -> str:
	for node in _nodes():
		data = node.get("data", {})
		if isinstance(data, dict) and str(data.get("type", "")) == "start":
			return str(node.get("id", "")).strip()
	raise RuntimeError("未找到 start 节点")


def _load_api_config() -> dict[str, Any]:
	raw_cfg = _load_yaml(API_CONFIG_PATH)
	selected: dict[str, Any] = raw_cfg

	active_provider = str(os.environ.get("ACTIVE_PROVIDER_OVERRIDE", "")).strip()
	if not active_provider:
		active_provider = str(raw_cfg.get("active_provider", "")).strip()
	providers = raw_cfg.get("providers")
	if active_provider and isinstance(providers, dict):
		provider_cfg = providers.get(active_provider)
		if not isinstance(provider_cfg, dict):
			raise RuntimeError(
				f"active_provider={active_provider} 未在 providers 中找到有效配置: {API_CONFIG_PATH}"
			)
		selected = provider_cfg

	api_key = str(selected.get("api_key", "")).strip()
	base_url = str(selected.get("base_url", selected.get("api_url", ""))).strip()
	model = str(selected.get("model", "gpt-5.2")).strip() or "gpt-5.2"
	timeout = int(selected.get("timeout", 300))
	max_retries = int(selected.get("max_retries", 3))

	if not api_key:
		raise RuntimeError(f"API 配置缺少 api_key: {API_CONFIG_PATH}")
	if not base_url:
		raise RuntimeError(f"API 配置缺少 base_url/api_url: {API_CONFIG_PATH}")

	return {
		"api_key": api_key,
		"base_url": base_url,
		"model": model,
		"timeout": timeout,
		"max_retries": max_retries,
	}


def _candidate_endpoints(base_url: str) -> list[str]:
	url = base_url.rstrip("/")
	candidates: list[str] = []
	if url.endswith("/v1"):
		candidates.append(f"{url}/chat/completions")
	candidates.append(url)
	if not url.endswith("/chat/completions"):
		candidates.append(f"{url}/chat/completions")
	unique: list[str] = []
	seen: set[str] = set()
	for item in candidates:
		if item not in seen:
			unique.append(item)
			seen.add(item)
	return unique


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


def _call_llm(messages: list[dict[str, str]], temperature: float | None, language: str = "") -> str:
	cfg = _load_api_config()
	lang_messages = [{"role": "system", "content": _language_instruction(language)}]
	payload: dict[str, Any] = {
		"model": cfg["model"],
		"messages": lang_messages + messages,
	}
	if temperature is not None:
		payload["temperature"] = temperature

	headers = {
		"Authorization": f"Bearer {cfg['api_key']}",
		"Content-Type": "application/json",
	}

	errors: list[str] = []
	for endpoint in _candidate_endpoints(cfg["base_url"]):
		for _ in range(cfg["max_retries"]):
			try:
				resp = requests.post(endpoint, headers=headers, json=payload, timeout=cfg["timeout"])
			except Exception as exc:
				errors.append(f"{endpoint} -> 请求异常: {exc}")
				continue

			if resp.status_code >= 400:
				errors.append(f"{endpoint} -> HTTP {resp.status_code}: {resp.text[:300]}")
				continue

			try:
				data = resp.json()
			except Exception as exc:
				errors.append(f"{endpoint} -> 响应非 JSON: {exc}; 原始响应: {resp.text[:300]}")
				continue

			content = _extract_content(data)
			if content:
				return content
			errors.append(f"{endpoint} -> 未解析到文本内容")

	raise RuntimeError("; ".join(errors)[:1200])


def _stringify(value: Any) -> str:
	if value is None:
		return ""
	if isinstance(value, str):
		return value
	if isinstance(value, (dict, list)):
		return json.dumps(value, ensure_ascii=False)
	return str(value)


def _resolve_selector(state: CDSSState, selector: list[Any] | tuple[Any, ...] | None) -> Any:
	if not selector:
		return ""
	if len(selector) < 2:
		return ""

	left = str(selector[0])
	right = str(selector[1])

	if left == "sys":
		if right == "query":
			return state.get("query", "")
		if right == "files":
			return state.get("files", [])
		return state.get(right, "")

	node_outputs = state.get("node_outputs", {})
	if left in node_outputs:
		return node_outputs.get(left, {}).get(right, "")
	return ""


def _render_template(template: str, state: CDSSState) -> str:
	if not template:
		return ""

	def replace(match: re.Match[str]) -> str:
		token = match.group(1).strip()
		if "." not in token:
			return ""
		left, right = token.split(".", 1)
		value = _resolve_selector(state, [left, right])
		return _stringify(value)

	return re.sub(r"\{\{#([^#]+)#\}\}", replace, template)


def _read_file_text(path: Path) -> str:
	suffix = path.suffix.lower()
	if suffix == ".pdf":
		if PdfReader is None:
			return ""
		try:
			reader = PdfReader(str(path))
			texts: list[str] = []
			for page in reader.pages:
				page_text = page.extract_text() or ""
				if page_text:
					texts.append(page_text)
			return "\n".join(texts).strip()
		except Exception:
			return ""

	try:
		return path.read_text(encoding="utf-8", errors="ignore").strip()
	except Exception:
		return ""


def _normalize_files(raw: Any) -> list[str]:
	if raw is None:
		return []
	if isinstance(raw, str):
		return [raw]
	if isinstance(raw, list):
		items: list[str] = []
		for item in raw:
			if isinstance(item, str):
				items.append(item)
			elif isinstance(item, dict):
				for key in ("path", "file_path", "name"):
					val = item.get(key)
					if isinstance(val, str) and val.strip():
						items.append(val)
						break
		return items
	return []


def _execute_code(code_text: str, kwargs: dict[str, Any]) -> dict[str, Any]:
	scope: dict[str, Any] = {}
	exec(code_text, scope)
	main_fn = scope.get("main")
	if not callable(main_fn):
		raise RuntimeError("代码节点缺少 main 函数")
	result = main_fn(**kwargs)
	if not isinstance(result, dict):
		raise RuntimeError("代码节点 main 返回值必须是 dict")
	return result


def _update_node_output(state: CDSSState, node_id: str, output: dict[str, Any]) -> dict[str, Any]:
	merged = dict(state.get("node_outputs", {}))
	merged[node_id] = output
	update: dict[str, Any] = {"node_outputs": merged}
	if "text" in output:
		update["final_answer"] = _stringify(output.get("text", ""))
	return update


def _run_node(node_id: str, state: CDSSState) -> dict[str, Any]:
	node = _node_index()[node_id]
	data = node.get("data", {})
	if not isinstance(data, dict):
		return _update_node_output(state, node_id, {})

	node_type = str(data.get("type", ""))

	if node_type == "start":
		output: dict[str, Any] = {}
		variables = data.get("variables", [])
		if isinstance(variables, list):
			for var in variables:
				if not isinstance(var, dict):
					continue
				name = str(var.get("variable", "")).strip()
				if not name:
					continue
				output[name] = state.get(name, "")
		return _update_node_output(state, node_id, output)

	if node_type == "document-extractor":
		selector = data.get("variable_selector", ["sys", "files"])
		raw_files = _resolve_selector(state, selector if isinstance(selector, list) else ["sys", "files"])
		file_list = _normalize_files(raw_files)
		text_chunks: list[str] = []
		errors: list[str] = []
		for item in file_list:
			path = Path(item)
			if not path.exists():
				errors.append(f"文件不存在: {item}")
				continue
			content = _read_file_text(path)
			if content:
				text_chunks.append(content)
			elif path.suffix.lower() == ".pdf" and PdfReader is None:
				errors.append(f"未安装 pypdf，无法解析 PDF: {item}")
			else:
				errors.append(f"文档解析结果为空: {item}")
		update = _update_node_output(state, node_id, {"text": "\n\n".join(text_chunks).strip(), "error": "; ".join(errors)})
		if errors:
			update["doc_error"] = "; ".join(errors)
		return update

	if node_type == "llm":
		language = _infer_language(state)
		prompt_template = data.get("prompt_template", [])
		messages: list[dict[str, str]] = []
		if isinstance(prompt_template, list):
			for item in prompt_template:
				if not isinstance(item, dict):
					continue
				role = str(item.get("role", "user")).strip().lower()
				if role not in {"system", "user", "assistant"}:
					role = "user"
				text = _render_template(str(item.get("text", "")), state)
				messages.append({"role": role, "content": text})

		# Dify 的高级聊天节点会把 memory.query_prompt_template 作为当前轮用户输入。
		memory = data.get("memory", {})
		if isinstance(memory, dict):
			query_prompt_template = str(memory.get("query_prompt_template", ""))
			query_text = _render_template(query_prompt_template, state)
			if query_text.strip():
				messages.append({"role": "user", "content": query_text})

		model_cfg = data.get("model", {})
		completion_params = model_cfg.get("completion_params", {}) if isinstance(model_cfg, dict) else {}
		temperature = None
		if isinstance(completion_params, dict) and "temperature" in completion_params:
			try:
				temp_raw = completion_params.get("temperature")
				if temp_raw is not None:
					temperature = float(temp_raw)
			except Exception:
				temperature = None

		text = _call_llm(messages, temperature, language)
		return _update_node_output(state, node_id, {"text": text})

	if node_type == "code":
		kwargs: dict[str, Any] = {}
		variables = data.get("variables", [])
		if isinstance(variables, list):
			for var in variables:
				if not isinstance(var, dict):
					continue
				name = str(var.get("variable", "")).strip()
				selector = var.get("value_selector", [])
				if name:
					kwargs[name] = _resolve_selector(state, selector if isinstance(selector, list) else [])
		output = _execute_code(str(data.get("code", "")), kwargs)
		return _update_node_output(state, node_id, output)

	if node_type == "if-else":
		return _update_node_output(state, node_id, {})

	if node_type == "answer":
		answer_text = _render_template(str(data.get("answer", "")), state)
		return _update_node_output(state, node_id, {"text": answer_text})

	return _update_node_output(state, node_id, {})


def _to_number(value: Any) -> float | None:
	try:
		return float(str(value).strip())
	except Exception:
		return None


def _eval_condition(condition: dict[str, Any], state: CDSSState) -> bool:
	selector = condition.get("variable_selector", [])
	left = _resolve_selector(state, selector if isinstance(selector, list) else [])
	operator = str(condition.get("comparison_operator", "=")).strip().lower()
	right = condition.get("value")
	var_type = str(condition.get("varType", "string")).strip().lower()

	if operator in {"empty", "is empty"}:
		return _stringify(left).strip() == ""
	if operator in {"not empty", "is not empty"}:
		return _stringify(left).strip() != ""

	if var_type == "number":
		left_num = _to_number(left)
		right_num = _to_number(right)
		if left_num is not None and right_num is not None:
			if operator in {"=", "=="}:
				return left_num == right_num
			if operator in {"!=", "<>"}:
				return left_num != right_num
			if operator == ">":
				return left_num > right_num
			if operator == ">=":
				return left_num >= right_num
			if operator == "<":
				return left_num < right_num
			if operator == "<=":
				return left_num <= right_num

	left_str = _stringify(left)
	right_str = _stringify(right)
	if operator in {"=", "=="}:
		return left_str == right_str
	if operator in {"!=", "<>"}:
		return left_str != right_str
	if operator == "contains":
		return right_str in left_str
	if operator in {"not contains", "not_contains"}:
		return right_str not in left_str
	if operator == ">":
		return left_str > right_str
	if operator == ">=":
		return left_str >= right_str
	if operator == "<":
		return left_str < right_str
	if operator == "<=":
		return left_str <= right_str
	return False


def _match_case_id(node_id: str, state: CDSSState) -> str:
	node = _node_index()[node_id]
	data = node.get("data", {})
	cases = data.get("cases", []) if isinstance(data, dict) else []
	if not isinstance(cases, list):
		return "false"

	for case in cases:
		if not isinstance(case, dict):
			continue
		case_id = str(case.get("case_id", "")).strip()
		conditions = case.get("conditions", [])
		logical_op = str(case.get("logical_operator", "and")).strip().lower()
		if not case_id or not isinstance(conditions, list):
			continue

		results = [_eval_condition(c, state) for c in conditions if isinstance(c, dict)]
		if not results:
			continue

		matched = any(results) if logical_op == "or" else all(results)
		if matched:
			return case_id

	return "false"


def _make_node_runner(node_id: str):
	def runner(state: CDSSState) -> dict[str, Any]:
		return _run_node(node_id, state)

	return runner


def _make_router(node_id: str, path_map: dict[str, str]):
	def router(state: CDSSState) -> str:
		case_id = _match_case_id(node_id, state)
		if case_id in path_map:
			return case_id
		if "false" in path_map:
			return "false"
		return "__end__"

	return router


def build_cdss_graph():
	nodes = _nodes()
	edges = _edges()
	node_by_id = _node_index()

	workflow = StateGraph(CDSSState)
	for node in nodes:
		node_id = str(node.get("id", "")).strip()
		if node_id:
			workflow.add_node(node_id, _make_node_runner(node_id))

	start_id = _start_node_id()
	workflow.add_edge(START, start_id)

	if_else_nodes: set[str] = set()
	for node in nodes:
		node_id = str(node.get("id", "")).strip()
		data = node.get("data", {})
		if isinstance(data, dict) and str(data.get("type", "")) == "if-else":
			if_else_nodes.add(node_id)

	for edge in edges:
		source = str(edge.get("source", "")).strip()
		target = str(edge.get("target", "")).strip()
		if not source or not target:
			continue
		if source in if_else_nodes:
			continue
		if source not in node_by_id or target not in node_by_id:
			continue
		workflow.add_edge(source, target)

	grouped: dict[str, dict[str, str]] = {}
	for edge in edges:
		source = str(edge.get("source", "")).strip()
		if source not in if_else_nodes:
			continue
		handle = str(edge.get("sourceHandle", "source")).strip() or "source"
		target = str(edge.get("target", "")).strip()
		if not target:
			continue
		grouped.setdefault(source, {})[handle] = target

	for node_id, path_map in grouped.items():
		full_map = dict(path_map)
		full_map["__end__"] = END
		workflow.add_conditional_edges(node_id, _make_router(node_id, full_map), full_map)

	return workflow.compile()


def render_workflow_png(output_path: Path) -> None:
	graph = _graph_data()
	nodes = graph.get("nodes", [])
	edges = graph.get("edges", [])

	try:
		import pygraphviz as pgv

		dot = pgv.AGraph(directed=True, strict=False)
		dot.graph_attr.update(rankdir="LR", fontname="Noto Sans CJK SC")
		dot.node_attr.update(shape="box", style="rounded", fontname="Noto Sans CJK SC")
		dot.edge_attr.update(fontname="Noto Sans CJK SC")

		for node in nodes:
			if not isinstance(node, dict):
				continue
			node_id = str(node.get("id", "")).strip()
			data = node.get("data", {})
			title = str(data.get("title", node_id)) if isinstance(data, dict) else node_id
			node_type = str(data.get("type", "")) if isinstance(data, dict) else ""
			label = f"{title}\\n({node_type})"
			if node_id:
				dot.add_node(node_id, label=label)

		for edge in edges:
			if not isinstance(edge, dict):
				continue
			source = str(edge.get("source", "")).strip()
			target = str(edge.get("target", "")).strip()
			if not source or not target:
				continue
			handle = str(edge.get("sourceHandle", "source")).strip() or "source"
			kwargs: dict[str, Any] = {}
			if handle != "source":
				kwargs["label"] = handle
			dot.add_edge(source, target, **kwargs)

		output_path.parent.mkdir(parents=True, exist_ok=True)
		dot.layout("dot")
		dot.draw(str(output_path))
		return
	except Exception:
		pass

	try:
		from graphviz import Digraph

		dot = Digraph(comment="cdss-workflow", format="png")
		dot.attr(rankdir="LR", fontname="Noto Sans CJK SC")
		dot.attr("node", shape="box", style="rounded", fontname="Noto Sans CJK SC")
		dot.attr("edge", fontname="Noto Sans CJK SC")

		for node in nodes:
			if not isinstance(node, dict):
				continue
			node_id = str(node.get("id", "")).strip()
			data = node.get("data", {})
			title = str(data.get("title", node_id)) if isinstance(data, dict) else node_id
			node_type = str(data.get("type", "")) if isinstance(data, dict) else ""
			if node_id:
				dot.node(node_id, f"{title}\n({node_type})")

		for edge in edges:
			if not isinstance(edge, dict):
				continue
			source = str(edge.get("source", "")).strip()
			target = str(edge.get("target", "")).strip()
			if not source or not target:
				continue
			handle = str(edge.get("sourceHandle", "source")).strip() or "source"
			label = handle if handle != "source" else ""
			dot.edge(source, target, label=label)

		output_path.parent.mkdir(parents=True, exist_ok=True)
		rendered = dot.render(filename=output_path.stem, directory=str(output_path.parent), cleanup=True)
		rendered_path = Path(rendered)
		if rendered_path != output_path and rendered_path.exists():
			output_path.write_bytes(rendered_path.read_bytes())
			rendered_path.unlink(missing_ok=True)
		return
	except Exception as exc:
		raise RuntimeError(f"PNG 渲染失败: {exc}") from exc


def run_once(app, query: str, tnm_ret: str, files: list[str]) -> dict[str, Any]:
	state: CDSSState = {
		"query": query,
		"TNM_ret": tnm_ret,
		"files": files,
		"node_outputs": {},
	}
	return app.invoke(state)


def main() -> None:
	parser = argparse.ArgumentParser(description="使用 LangGraph 复刻 Dify CDSS 工作流并导出 PNG")
	parser.add_argument("--workflow", default="files/非小细胞癌治疗-免提示.yml", help="Dify YAML 路径")
	parser.add_argument("--api-config", default="api_config.yaml", help="API 配置文件路径")
	parser.add_argument("--png", default="outputs/cdss_graph.png", help="输出 PNG 路径")
	parser.add_argument("--run", action="store_true", help="是否执行一次工作流")
	parser.add_argument("--query", default="", help="对话 query（对应 sys.query）")
	parser.add_argument("--tnm-ret", default="", help="开始节点输入 TNM_ret")
	parser.add_argument("--files", nargs="*", default=[], help="文档路径列表（对应 sys.files）")
	args = parser.parse_args()

	set_workflow_path(args.workflow)
	set_api_config_path(args.api_config)

	app = build_cdss_graph()
	render_workflow_png(Path(args.png))
	print(f"工作流 PNG 已生成: {args.png}")

	if args.run:
		result = run_once(app, query=args.query, tnm_ret=args.tnm_ret, files=args.files)
		print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
