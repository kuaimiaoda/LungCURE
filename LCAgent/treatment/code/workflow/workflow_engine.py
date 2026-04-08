from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph


DEFAULT_PATIENT = {
    "病理类型": "非鳞癌",
    "体力评分": "0-1分",
    "驱动基因": "0",
    "PDL1表达": "低表达(1-49%)",
    "高危因素": "无",
    "治疗阶段": "0",
    "既往治疗方案": "1",
    "免疫禁忌": "无",
    "转移类型": "1",
    "是否一线治疗": "1",
    "TNM分期": "T2N0M0",
    "综合分期": "IIIA",
}

# 全局 LLM 配置：统一控制当前脚本内所有 llm 节点的模型配置
# 与 llm/llmpy.py 中 LLM_UNIFIED_CONFIG 对齐的字段：
# - model: kimi-k2.5
# - completion_params: max_tokens/temperature/top_p/presence_penalty/frequency_penalty
GLOBAL_LLM_CONFIG = {
    "model": "kimi-k2.5",
    "provider": None,  # None 表示保留 YAML 原 provider
    "mode": "chat",
    "show_prompt_template": True,
    "prompt_preview_chars": 1200,
    "completion_params": {
        # 与 llmpy 对齐：默认仅下发 max_tokens，其余默认不下发
        "max_tokens": 8192,
        "temperature": None,
        "top_p": None,
        "presence_penalty": None,
        "frequency_penalty": None,
    },
}


class FlowState(TypedDict, total=False):
    tnm_ret: str
    document_text: str
    query: str
    extracted_json_text: str
    ctx: dict[str, dict[str, Any]]
    route: list[str]
    final_answer: str


class NSCLCWorkflowLangGraphReplay:
    def __init__(
        self,
        workflow_path: str,
        llm_executor: Callable[..., str] | None = None,
        log_enabled: bool = True,
    ) -> None:
        self.workflow_path = Path(workflow_path)
        raw = yaml.safe_load(self.workflow_path.read_text(encoding="utf-8"))
        graph = raw["workflow"]["graph"]

        self.nodes = graph["nodes"]
        self.edges = graph["edges"]
        self.node_by_id = {n["id"]: n for n in self.nodes}

        self.outgoing: dict[str, list[dict[str, Any]]] = {}
        for edge in self.edges:
            self.outgoing.setdefault(edge["source"], []).append(edge)

        start_nodes = [n for n in self.nodes if n["data"]["type"] == "start"]
        if len(start_nodes) != 1:
            raise ValueError(f"期望 1 个 start 节点，实际 {len(start_nodes)}")
        self.start_id = start_nodes[0]["id"]
        self.llm_executor = llm_executor
        self.log_enabled = log_enabled
        self.llm_config = dict(GLOBAL_LLM_CONFIG)
        self._apply_global_llm_config()

        self.app = self._compile_graph()

    def _log(self, text: str) -> None:
        if self.log_enabled:
            print(text)

    def _apply_global_llm_config(self) -> None:
        for node in self.nodes:
            data = node.get("data", {})
            if data.get("type") != "llm":
                continue

            model_cfg = data.setdefault("model", {})
            model_cfg["name"] = self.llm_config["model"]
            if self.llm_config.get("mode"):
                model_cfg["mode"] = self.llm_config["mode"]

            provider = self.llm_config.get("provider")
            if provider:
                model_cfg["provider"] = provider

            completion_params = model_cfg.setdefault("completion_params", {})
            global_params = self.llm_config.get("completion_params", {})
            for key, value in global_params.items():
                if value is None:
                    completion_params.pop(key, None)
                else:
                    completion_params[key] = value

    def _compile_graph(self):
        builder: StateGraph = StateGraph(FlowState)

        for node in self.nodes:
            node_id = node["id"]
            builder.add_node(node_id, self._build_node_fn(node_id))

        builder.add_edge(START, self.start_id)

        for node in self.nodes:
            node_id = node["id"]
            node_type = node["data"]["type"]

            if node_type == "if-else":
                mapping = {
                    edge.get("sourceHandle", "source"): edge["target"]
                    for edge in self.outgoing.get(node_id, [])
                }
                if not mapping:
                    builder.add_edge(node_id, END)
                    continue

                builder.add_conditional_edges(
                    node_id,
                    self._build_router_fn(node_id, mapping),
                    mapping,
                )
                continue

            if node_type == "answer":
                builder.add_edge(node_id, END)
                continue

            next_edge = self._pick_single_source_edge(node_id)
            if next_edge is None:
                builder.add_edge(node_id, END)
            else:
                builder.add_edge(node_id, next_edge["target"])

        return builder.compile()

    def _build_node_fn(self, node_id: str):
        node = self.node_by_id[node_id]
        data = node["data"]
        node_type = data["type"]
        title = data.get("title", "")

        def fn(state: FlowState) -> FlowState:
            ctx = self._copy_ctx(state.get("ctx", {}))
            route = list(state.get("route", []))
            step = len(route) + 1
            route_item = f"{step:02d}. {title} [{node_type}] ({node_id})"
            route.append(route_item)

            self._log(f"\n[{step:02d}] {title} [{node_type}] ({node_id})")

            update: FlowState = {"ctx": ctx, "route": route}

            if node_type == "start":
                values: dict[str, Any] = {}
                for var in data.get("variables", []):
                    name = var["variable"]
                    values[name] = state.get("tnm_ret", "") if name == "TNM_ret" else ""
                ctx[node_id] = values
                self._log(f"  输出: {values}")

            elif node_type == "document-extractor":
                doc_text = state.get("document_text", "")
                ctx[node_id] = {"text": doc_text}
                self._log(f"  输出: text(len={len(doc_text)})")

            elif node_type == "llm":
                model_name = self._node_model_name(node_id)
                params = self._node_completion_params(node_id)
                self._log(f"  模型: {model_name}")
                self._log(f"  参数: {params}")
                if self.llm_config.get("show_prompt_template", True):
                    prompt_text = self._node_prompt_summary(
                        node_id,
                        max_chars=int(self.llm_config.get("prompt_preview_chars", 1200)),
                    )
                    if prompt_text:
                        self._log("  提示词:")
                        self._log(prompt_text)

                override_text = state.get("extracted_json_text")
                if node_id == "llm" and override_text:
                    text = str(override_text)
                else:
                    messages = self._build_llm_messages(node_id, state, ctx)
                    if self.llm_executor is not None:
                        text = self.llm_executor(
                            node_id=node_id,
                            node_title=title,
                            model_name=model_name,
                            params=params,
                            messages=messages,
                            node_data=data,
                            state=state,
                            context=ctx,
                        )
                    else:
                        text = self._mock_llm_text(node_id, title, ctx)
                ctx[node_id] = {"text": text}
                preview = text if len(text) <= 180 else text[:180] + "..."
                self._log(f"  输出: text={preview}")

            elif node_type == "code":
                kwargs = self._build_code_kwargs(data.get("variables", []), ctx)
                result = self._run_embedded_code(data["code"], kwargs)
                ctx[node_id] = result
                self._log(f"  入参: {kwargs}")
                self._log(f"  输出: {result}")

            elif node_type == "if-else":
                branch = self._pick_branch(data, ctx)
                ctx[node_id] = {"branch": branch}
                self._log(f"  决策: branch={branch}")

            elif node_type == "answer":
                answer_text = self._render_template(data.get("answer", ""), ctx)
                ctx[node_id] = {"text": answer_text}
                update["final_answer"] = answer_text
                self._log(f"  最终回复:\n{answer_text}")

            else:
                raise ValueError(f"暂不支持节点类型: {node_type} ({node_id})")

            return update

        return fn

    def _build_router_fn(self, node_id: str, mapping: dict[str, str]):
        def router(state: FlowState) -> str:
            ctx = state.get("ctx", {})
            data = self.node_by_id[node_id]["data"]
            branch = self._pick_branch(data, ctx)
            if branch in mapping:
                target = mapping[branch]
                target_title = self.node_by_id[target]["data"].get("title", "")
                self._log(f"  跳转: ({branch}) -> {target_title} ({target})")
                return branch

            if "false" in mapping:
                target = mapping["false"]
                target_title = self.node_by_id[target]["data"].get("title", "")
                self._log(f"  跳转: (false fallback) -> {target_title} ({target})")
                return "false"

            fallback = next(iter(mapping))
            target = mapping[fallback]
            target_title = self.node_by_id[target]["data"].get("title", "")
            self._log(f"  跳转: (fallback={fallback}) -> {target_title} ({target})")
            return fallback

        return router

    def invoke(
        self,
        extracted_json_text: str | None = None,
        tnm_ret: str = "",
        document_text: str = "",
        query: str = "",
    ) -> FlowState:
        initial_state: FlowState = {
            "tnm_ret": tnm_ret,
            "document_text": document_text,
            "query": query,
            "ctx": {
                "sys": {
                    "query": query,
                    "files": document_text,
                }
            },
            "route": [],
            "final_answer": "",
        }
        if extracted_json_text is not None:
            initial_state["extracted_json_text"] = extracted_json_text
        result = self.app.invoke(initial_state)
        return result

    def export_llm_prompt_catalog(
        self,
        output_path: str = "llm_prompts_catalog.md",
        max_chars_per_block: int = 10000,
    ) -> str:
        lines: list[str] = []
        lines.append("# LLM 节点提示词清单")
        lines.append("")

        llm_nodes = [n for n in self.nodes if n.get("data", {}).get("type") == "llm"]
        for node in llm_nodes:
            node_id = node["id"]
            data = node["data"]
            title = data.get("title", "")
            model = data.get("model", {})
            provider = model.get("provider", "")
            model_name = model.get("name", "")

            lines.append(f"## {title} ({node_id})")
            lines.append(f"- model: `{model_name}`")
            lines.append(f"- provider: `{provider}`")
            lines.append("")

            memory = data.get("memory", {})
            query_prompt_template = memory.get("query_prompt_template", "")
            if query_prompt_template:
                lines.append("### memory.query_prompt_template")
                lines.append("")
                lines.append("```text")
                lines.append(self._shorten_text(str(query_prompt_template), max_chars_per_block))
                lines.append("```")
                lines.append("")

            prompt_template = data.get("prompt_template", [])
            if isinstance(prompt_template, list):
                for idx, msg in enumerate(prompt_template, start=1):
                    role = str(msg.get("role", "unknown"))
                    text = str(msg.get("text", ""))
                    lines.append(f"### prompt_template[{idx}] role={role}")
                    lines.append("")
                    lines.append("```text")
                    lines.append(self._shorten_text(text, max_chars_per_block))
                    lines.append("```")
                    lines.append("")

            if not query_prompt_template and not prompt_template:
                lines.append("_该节点未配置提示词内容_")
                lines.append("")

        out = Path(output_path).resolve()
        out.write_text("\n".join(lines), encoding="utf-8")
        return str(out)

    def export_visual_structure(self, output_prefix: str = "workflow_structure") -> dict[str, str]:
        mermaid_text = self._build_mermaid()
        mmd_path = Path(f"{output_prefix}.mmd").resolve()
        html_path = Path(f"{output_prefix}.html").resolve()

        mmd_path.write_text(mermaid_text, encoding="utf-8")
        html_path.write_text(self._wrap_mermaid_html(mermaid_text), encoding="utf-8")

        return {
            "mmd": str(mmd_path),
            "html": str(html_path),
        }

    def _build_mermaid(self) -> str:
        lines: list[str] = []
        lines.append("flowchart LR")
        lines.append("    classDef start fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px;")
        lines.append("    classDef llm fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.5px;")
        lines.append("    classDef code fill:#fff3e0,stroke:#ef6c00,stroke-width:1.5px;")
        lines.append("    classDef ifelse fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1.5px;")
        lines.append("    classDef answer fill:#ffebee,stroke:#c62828,stroke-width:2px;")
        lines.append("    classDef doc fill:#e0f7fa,stroke:#00838f,stroke-width:1.5px;")

        mermaid_id: dict[str, str] = {}
        for idx, node in enumerate(self.nodes):
            mermaid_id[node["id"]] = f"n{idx}"

        for node in self.nodes:
            node_id = node["id"]
            data = node["data"]
            node_type = data["type"]
            title = data.get("title", "")
            if node_type == "if-else":
                case_count = len(data.get("cases", []))
                title = f"{title}（多分支:{case_count}）"
            if node_type == "llm":
                title = f"{title}[{self._node_model_name(node_id)}]"
            safe_title = title.replace("\"", "'")
            safe_type = node_type.replace("\"", "'")
            label = f"{safe_title}<br/>({safe_type})"
            lines.append(f'    {mermaid_id[node_id]}["{label}"]')

        for edge in self.edges:
            src = mermaid_id[edge["source"]]
            dst = mermaid_id[edge["target"]]
            handle = edge.get("sourceHandle", "source")
            display_label = self._edge_display_label(edge["source"], str(handle))
            safe_label = display_label.replace("\"", "'")
            if safe_label == "":
                lines.append(f"    {src} --> {dst}")
            else:
                lines.append(f'    {src} -- "{safe_label}" --> {dst}')

        for node in self.nodes:
            node_id = node["id"]
            node_type = node["data"]["type"]
            cls = self._mermaid_class(node_type)
            if cls:
                lines.append(f"    class {mermaid_id[node_id]} {cls};")

        return "\n".join(lines) + "\n"

    def _edge_display_label(self, source_node_id: str, handle: str) -> str:
        source = self.node_by_id[source_node_id]
        source_type = source["data"]["type"]
        if source_type != "if-else":
            return "" if handle == "source" else handle

        if handle == "false":
            return "else"

        case_labels = self._if_case_label_map(source_node_id)
        return case_labels.get(handle, handle)

    def _if_case_label_map(self, node_id: str) -> dict[str, str]:
        data = self.node_by_id[node_id]["data"]
        labels: dict[str, str] = {}

        for case in data.get("cases", []):
            case_id = case.get("case_id")
            conds = case.get("conditions", [])
            if not case_id or not conds:
                continue

            logical_op = str(case.get("logical_operator", "and")).upper()
            pieces: list[str] = []
            for cond in conds:
                selector = cond.get("variable_selector", [])
                left = self._selector_display(selector)
                op = cond.get("comparison_operator", "=")
                right = cond.get("value", "")
                pieces.append(f"{left} {op} {right}")

            if len(pieces) == 1:
                labels[str(case_id)] = pieces[0]
            else:
                labels[str(case_id)] = f" {logical_op} ".join(pieces)

        return labels

    def _selector_display(self, selector: list[Any]) -> str:
        if len(selector) != 2:
            return "unknown"
        ref_node_id, key = str(selector[0]), str(selector[1])
        ref_node = self.node_by_id.get(ref_node_id, {})
        title = ref_node.get("data", {}).get("title", ref_node_id)
        return f"{title}.{key}"

    def _build_llm_messages(
        self,
        node_id: str,
        state: FlowState,
        ctx: dict[str, dict[str, Any]],
    ) -> list[dict[str, str]]:
        node = self.node_by_id.get(node_id, {})
        data = node.get("data", {})
        messages: list[dict[str, str]] = []

        prompt_template = data.get("prompt_template", [])
        if isinstance(prompt_template, list):
            for item in prompt_template:
                role = str(item.get("role", "user")).strip() or "user"
                content = self._render_dify_text(str(item.get("text", "")), state, ctx).strip()
                if content:
                    messages.append({"role": role, "content": content})

        memory = data.get("memory", {})
        query_prompt_template = str(memory.get("query_prompt_template", "")).strip()
        if query_prompt_template:
            rendered_query_prompt = self._render_dify_text(query_prompt_template, state, ctx).strip()
            if rendered_query_prompt:
                messages.append({"role": "user", "content": rendered_query_prompt})

        if not messages:
            fallback_query = str(ctx.get("sys", {}).get("query", "")).strip()
            if fallback_query:
                messages.append({"role": "user", "content": fallback_query})
            else:
                messages.append({"role": "user", "content": "请根据输入信息给出结果。"})

        return messages

    def _render_dify_text(
        self,
        text: str,
        state: FlowState,
        ctx: dict[str, dict[str, Any]],
    ) -> str:
        pattern = re.compile(r"\{\{#([^#}]+)#\}\}")

        def repl(match: re.Match[str]) -> str:
            token = match.group(1)
            if "." not in token:
                return ""

            node_or_scope, key = token.split(".", 1)
            node_or_scope = node_or_scope.strip()
            key = key.strip()

            if node_or_scope == "sys":
                value = ctx.get("sys", {}).get(key, state.get(key, ""))
                return "" if value is None else str(value)

            value = ctx.get(node_or_scope, {}).get(key, "")
            return "" if value is None else str(value)

        return pattern.sub(repl, text)

    def _node_model_name(self, node_id: str) -> str:
        node = self.node_by_id.get(node_id, {})
        model = node.get("data", {}).get("model", {})
        return str(model.get("name", "unknown"))

    def _node_completion_params(self, node_id: str) -> dict[str, Any]:
        node = self.node_by_id.get(node_id, {})
        model = node.get("data", {}).get("model", {})
        params = model.get("completion_params", {})
        if isinstance(params, dict):
            return dict(params)
        return {}

    def _node_prompt_summary(self, node_id: str, max_chars: int = 1200) -> str:
        node = self.node_by_id.get(node_id, {})
        data = node.get("data", {})
        items: list[str] = []

        memory = data.get("memory", {})
        query_prompt_template = memory.get("query_prompt_template", "")
        if query_prompt_template:
            items.append(
                f"    [memory.query_prompt_template]\n{self._shorten_text(query_prompt_template, max_chars)}"
            )

        prompt_template = data.get("prompt_template", [])
        if isinstance(prompt_template, list):
            for idx, msg in enumerate(prompt_template, start=1):
                role = str(msg.get("role", "unknown"))
                text = str(msg.get("text", ""))
                items.append(f"    [{idx}] {role}\n{self._shorten_text(text, max_chars)}")

        return "\n".join(items)

    @staticmethod
    def _shorten_text(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return text
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n...(已截断，原始长度 {len(text)} 字符)"

    @staticmethod
    def _wrap_mermaid_html(mermaid_text: str) -> str:
        return (
            "<!doctype html>\n"
            "<html lang=\"zh-CN\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\" />\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
            "  <title>Workflow Visualization</title>\n"
            "  <style>\n"
            "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 24px; background: #f7f9fb; }\n"
            "    .panel { background: #fff; border: 1px solid #d9e2ec; border-radius: 10px; padding: 20px; overflow-x: auto; }\n"
            "    .note { color: #445; margin: 0 0 12px 0; }\n"
            "  </style>\n"
            "  <script type=\"module\">\n"
            "    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';\n"
            "    mermaid.initialize({ startOnLoad: true, theme: 'default', flowchart: { curve: 'linear' } });\n"
            "  </script>\n"
            "</head>\n"
            "<body>\n"
            "  <p class=\"note\">若图未渲染，请检查网络后直接打开同目录的 .mmd 文件。</p>\n"
            "  <div class=\"panel\">\n"
            "    <pre class=\"mermaid\">\n"
            f"{mermaid_text}"
            "    </pre>\n"
            "  </div>\n"
            "</body>\n"
            "</html>\n"
        )

    @staticmethod
    def _mermaid_class(node_type: str) -> str:
        mapping = {
            "start": "start",
            "llm": "llm",
            "code": "code",
            "if-else": "ifelse",
            "answer": "answer",
            "document-extractor": "doc",
        }
        return mapping.get(node_type, "")

    def _pick_single_source_edge(self, node_id: str) -> dict[str, Any] | None:
        outs = self.outgoing.get(node_id, [])
        if not outs:
            return None

        source_edges = [e for e in outs if e.get("sourceHandle") == "source"]
        if len(source_edges) == 1:
            return source_edges[0]

        if len(outs) == 1:
            return outs[0]

        raise ValueError(f"节点 {node_id} 非条件节点但有多个出边")

    def _build_code_kwargs(
        self,
        variables: list[dict[str, Any]],
        ctx: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for var in variables:
            kwargs[var["variable"]] = self._get_by_selector(var.get("value_selector", []), ctx)
        return kwargs

    def _run_embedded_code(self, code: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        env: dict[str, Any] = {}
        exec(code, env, env)
        fn = env.get("main")
        if not callable(fn):
            raise ValueError("code 节点内未找到 main 函数")
        result = fn(**kwargs)
        if not isinstance(result, dict):
            raise ValueError(f"code 节点返回值必须是 dict，实际为 {type(result)}")
        return result

    def _pick_branch(self, if_data: dict[str, Any], ctx: dict[str, dict[str, Any]]) -> str:
        for case in if_data.get("cases", []):
            if self._case_match(case, ctx):
                return case["case_id"]
        return "false"

    def _case_match(self, case: dict[str, Any], ctx: dict[str, dict[str, Any]]) -> bool:
        conditions = case.get("conditions", [])
        if not conditions:
            return False

        op = case.get("logical_operator", "and").lower()
        matches = [self._condition_match(cond, ctx) for cond in conditions]
        if op == "or":
            return any(matches)
        return all(matches)

    def _condition_match(self, cond: dict[str, Any], ctx: dict[str, dict[str, Any]]) -> bool:
        left = self._get_by_selector(cond.get("variable_selector", []), ctx)
        right = cond.get("value")
        comp = cond.get("comparison_operator")
        var_type = cond.get("varType")

        if var_type == "number":
            left_num = self._to_number(left)
            right_num = self._to_number(right)
            if left_num is not None and right_num is not None:
                return self._compare(left_num, right_num, comp)

        return self._compare(str(left), str(right), comp)

    @staticmethod
    def _compare(left: Any, right: Any, comp: str | None) -> bool:
        if comp == "=":
            return left == right
        if comp == "!=":
            return left != right
        if comp == ">":
            return left > right
        if comp == ">=":
            return left >= right
        if comp == "<":
            return left < right
        if comp == "<=":
            return left <= right
        raise ValueError(f"不支持的比较符: {comp}")

    @staticmethod
    def _to_number(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_by_selector(selector: list[Any], ctx: dict[str, dict[str, Any]]) -> Any:
        if len(selector) != 2:
            return ""
        node_id, key = selector
        return ctx.get(str(node_id), {}).get(str(key), "")

    def _render_template(self, template: str, ctx: dict[str, dict[str, Any]]) -> str:
        state: FlowState = {"ctx": ctx}
        return self._render_dify_text(template, state, ctx)

    @staticmethod
    def _copy_ctx(ctx: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        copied: dict[str, dict[str, Any]] = {}
        for key, value in ctx.items():
            copied[key] = dict(value)
        return copied

    def _mock_llm_text(self, node_id: str, title: str, ctx: dict[str, dict[str, Any]]) -> str:
        patient_json = ctx.get("1765803586857", {}).get("result", "")
        prompt_preview = self._node_prompt_summary(node_id, max_chars=280).replace("\n", " | ")
        return (
            f"[模拟LLM节点: {title}]\n"
            "此输出用于复现流程，不调用真实模型。\n"
            f"节点提示词摘要: {prompt_preview}\n"
            f"病人信息: {patient_json}"
        )


def load_extracted_json_text(file_path: str | None, inline_json: str | None) -> str:
    if inline_json:
        return inline_json

    if file_path:
        path = Path(file_path)
        raw = path.read_text(encoding="utf-8").strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return json.dumps(obj, ensure_ascii=False)
            return raw
        except json.JSONDecodeError:
            return raw

    return json.dumps(DEFAULT_PATIENT, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 LangGraph 复现 Dify YAML 工作流流程。"
    )
    parser.add_argument("--workflow", default="非小细胞癌治疗-免提示.yml", help="YAML 路径")
    parser.add_argument("--tnm-ret", default="", help="start 节点 TNM_ret")
    parser.add_argument("--document-text", default="", help="文档提取器输出")
    parser.add_argument("--query", default="", help="sys.query")
    parser.add_argument("--extracted-json", default=None, help="病人信息提取节点输出(JSON字符串)")
    parser.add_argument("--extracted-json-file", default=None, help="病人信息提取节点输出文件")
    parser.add_argument(
        "--visual-output",
        default="workflow_structure",
        help="可视化结构输出文件前缀（会生成 .mmd 和 .html）",
    )
    parser.add_argument(
        "--llm-prompt-output",
        default="llm_prompts_catalog.md",
        help="导出全部 llm 节点提示词清单（Markdown）",
    )
    parser.add_argument("--graph-only", action="store_true", help="仅查看结构")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay = NSCLCWorkflowLangGraphReplay(args.workflow)

    print("\n### 1) 工作流可视化")
    files = replay.export_visual_structure(args.visual_output)
    print(f"已生成 Mermaid 文件: {files['mmd']}")
    print(f"已生成 HTML 可视化: {files['html']}")
    prompt_catalog_file = replay.export_llm_prompt_catalog(args.llm_prompt_output)
    print(f"已导出 LLM 提示词清单: {prompt_catalog_file}")

    if args.graph_only:
        return

    extracted_json_text = load_extracted_json_text(args.extracted_json_file, args.extracted_json)

    print("\n### 2) 执行流程")
    result = replay.invoke(
        extracted_json_text=extracted_json_text,
        tnm_ret=args.tnm_ret,
        document_text=args.document_text,
        query=args.query,
    )

    print("\n### 3) 路径摘要")
    for item in result.get("route", []):
        print(item)

    print("\n### 4) 最终输出")
    print(result.get("final_answer", ""))


if __name__ == "__main__":
    main()
