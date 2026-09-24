"""
Agent 主循环
============
自实现的 Function Calling 调度器（不套 LangChain Agent，面试可讲清每一步）：

  用户问题
    → LLM 决定是否调用工具
    → 调工具（知识库检索 / 沙箱跑代码 / 读文件）
    → 结果回灌给 LLM
    → 循环直到 LLM 给出最终答案，或达到 max_steps 上限

同时把每一步记录成 trace，供 Web 界面展示"思考过程"。
"""
from __future__ import annotations

import json
from typing import Any

from config import config
from core.llm import LLMClient
from core.tool import registry
from tools import register_all_tools


DEFAULT_SYSTEM_PROMPT = """你是一个企业知识库问答助手。你的工作规则：

1. 涉及公司制度、产品资料、业务规则、内部数据等问题，**必须**先调用 search_knowledge_base 检索相关文档，
   再基于检索到的内容回答，禁止凭记忆编造。
2. 当需要做数值计算、数据统计、格式转换时，调用 run_python 在沙箱里执行代码。
3. 回答要严谨，必须基于检索到的资料。如果知识库中没有相关信息，直接如实说明"知识库中未找到相关内容"。
4. 给出最终答案时，请在关键结论后用【来源:文件名】标注信息出处。
5. 不要重复调用相同工具；拿到足够信息后直接回答用户。
"""


class KBAgent:
    def __init__(self, system_prompt: str | None = None) -> None:
        register_all_tools()
        self.llm = LLMClient()
        self.system_prompt = system_prompt or config.system_prompt or DEFAULT_SYSTEM_PROMPT
        self.tools_schema = registry.openai_schemas()

    def run(self, user_query: str) -> dict[str, Any]:
        """执行一轮完整问答，返回 {answer, steps}。"""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        steps: list[dict[str, Any]] = []
        final_answer = ""

        for step in range(1, config.max_steps + 1):
            resp = self.llm.chat(messages, tools=self.tools_schema)

            # 没有工具调用 → LLM 给出了最终答案
            if not resp["tool_calls"]:
                final_answer = resp["content"] or ""
                steps.append({"step": step, "type": "final", "content": final_answer})
                break

            # 把 assistant 的 tool_calls 意图加进历史
            messages.append({
                "role": "assistant",
                "content": resp["content"],
                "tool_calls": resp["tool_calls"],
            })

            # 逐个执行工具
            for tc in resp["tool_calls"]:
                fname = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}

                tool = registry.get(fname)
                if tool is None:
                    observation = f"[错误] 未知工具: {fname}"
                else:
                    observation = tool.run(**args)

                steps.append({
                    "step": step,
                    "type": "tool_call",
                    "tool": fname,
                    "args": args,
                    "observation": observation,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": fname,
                    "content": observation,
                })
        else:
            # 达到步数上限还没收敛，强制 LLM 总结
            messages.append({
                "role": "user",
                "content": "你已经调用了多次工具，请基于已有信息直接给出最终答案，不要再调用工具。",
            })
            resp = self.llm.chat(messages)
            final_answer = resp["content"] or ""
            steps.append({"step": config.max_steps + 1, "type": "forced_final",
                          "content": final_answer})

        return {"answer": final_answer, "steps": steps}
