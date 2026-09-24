"""
命令行入口
==========
用法：
  1) 建索引：  python main.py --build
  2) 单轮问答：python main.py --ask "公司的报销制度是什么？"
  3) 交互模式：python main.py
"""
from __future__ import annotations

import sys

from config import config


def cmd_build() -> None:
    from rag.pipeline import RAGPipeline
    print(f"[建索引] 扫描目录: {config.docs_dir}")
    rag = RAGPipeline()
    n = rag.build_index()
    print(f"[完成] 入库 {n} 个文本块，当前库内共 {rag.count()} 块。")


def cmd_ask(question: str) -> None:
    from core.agent import KBAgent
    agent = KBAgent()
    print(f"\n[用户] {question}\n")
    result = agent.run(question)

    # 打印思考过程
    for s in result["steps"]:
        if s["type"] == "tool_call":
            print(f"  🛠  调用工具 {s['tool']}({s['args']})")
            print(f"     ↳ {s['observation'][:200]}{'...' if len(s['observation'])>200 else ''}")
    print(f"\n[助手] {result['answer']}\n")


def interactive() -> None:
    from core.agent import KBAgent
    print("=" * 60)
    print(" 企业知识库问答 Agent（输入 exit 退出）")
    print("=" * 60)
    agent = KBAgent()
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出"):
            break
        result = agent.run(q)
        print(f"\n助手> {result['answer']}\n")


def main() -> None:
    config.ensure_dirs()
    args = sys.argv[1:]

    if not args:
        interactive()
        return
    if args[0] == "--build":
        cmd_build()
    elif args[0] == "--ask" and len(args) >= 2:
        config.validate()
        cmd_ask(" ".join(args[1:]))
    elif args[0] == "--ask":
        print("用法: python main.py --ask '你的问题'")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
