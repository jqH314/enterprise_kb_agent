"""
LLM 客户端
==========
封装 OpenAI 兼容接口（DeepSeek / 通义 / OpenAI 等均可），
同时支持：
  - function calling（tools 参数，给 Agent 主循环用）
  - 流式输出（给 Web 界面用）
"""
from __future__ import annotations

from typing import Any, Generator

from openai import OpenAI

from config import config


class LLMClient:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self.model = config.llm_model

    def chat(self, messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """非流式调用，返回标准化后的 message dict。

        返回结构::
            {"content": str|None, "tool_calls": [...], "finish_reason": str}
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        return {
            "content": msg.content,
            "tool_calls": tool_calls,
            "finish_reason": choice.finish_reason,
        }

    def stream_chat(self, messages: list[dict[str, Any]]) -> Generator[str, None, None]:
        """纯文本流式调用（无工具调用场景，用于 UI 逐字打印）。"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                yield delta.content
