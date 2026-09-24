"""
工具注册系统
============
通过 @tool 装饰器把普通 Python 函数注册成 Agent 可调用的工具。
自动从函数签名 + 类型注解 + docstring 生成 OpenAI function calling 的 JSON Schema。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, get_type_hints, get_origin, get_args


_TYPE_MAP: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _parse_docstring_params(doc: str) -> Dict[str, str]:
    """简单解析 docstring 中 'param: 说明' 形式的参数描述。"""
    desc: Dict[str, str] = {}
    if not doc:
        return desc
    for line in doc.splitlines():
        line = line.strip()
        for prefix in (":param ", "@param "):
            if line.startswith(prefix):
                rest = line[len(prefix):]
                if ":" in rest:
                    name, _, text = rest.partition(":")
                    desc[name.strip()] = text.strip()
                break
        else:
            if ":" in line and not line.startswith(("Returns", "Args", "Raises")):
                head, _, text = line.partition(":")
                head = head.strip()
                if head and " " not in head and head[0].isalpha():
                    desc[head] = text.strip()
    return desc


def _json_schema_for_param(annotation: Any) -> Dict[str, Any]:
    origin = get_origin(annotation)
    if origin is list or annotation is list:
        args = get_args(annotation)
        item_type = _json_schema_for_param(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_type}
    if origin is dict or annotation is dict:
        return {"type": "object"}
    if origin is not None:
        return {"type": "string"}
    return {"type": _TYPE_MAP.get(annotation, "string")}


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., Any] = field(repr=False)

    @property
    def openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, **kwargs: Any) -> str:
        try:
            result = self.func(**kwargs)
            return str(result) if result is not None else "(无返回值)"
        except Exception as e:  # noqa: BLE001
            return f"[工具执行出错] {type(e).__name__}: {e}"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, func: Callable[..., Any], name: str | None = None,
                 description: str | None = None) -> Tool:
        tool = self._build_tool(func, name=name, description=description)
        self._tools[tool.name] = tool
        return tool

    def _build_tool(self, func: Callable[..., Any],
                    name: str | None, description: str | None) -> Tool:
        tool_name = name or func.__name__
        doc = inspect.getdoc(func) or ""
        tool_desc = description or doc.split("\n\n")[0].strip()

        sig = inspect.signature(func)
        hints = get_type_hints(func)
        param_descs = _parse_docstring_params(doc)

        properties: Dict[str, Any] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            ann = hints.get(pname, str)
            properties[pname] = _json_schema_for_param(ann)
            if pname in param_descs:
                properties[pname]["description"] = param_descs[pname]
            if p.default is inspect.Parameter.empty:
                required.append(pname)

        parameters = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        return Tool(name=tool_name, description=tool_desc,
                    parameters=parameters, func=func)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def openai_schemas(self) -> list[Dict[str, Any]]:
        return [t.openai_schema for t in self._tools.values()]


registry = ToolRegistry()


def tool(func: Callable[..., Any] | None = None, *,
         name: str | None = None, description: str | None = None):
    """@tool 装饰器。用法::

        @tool
        def add(a: int, b: int) -> int:
            "两数相加"
            return a + b
    """
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        registry.register(f, name=name, description=description)
        return f

    if func is not None:
        return decorator(func)
    return decorator
