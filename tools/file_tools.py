"""
受限文件工具
============
只允许读写沙箱 workspace 目录，禁止路径逃逸（../../etc/passwd 这类会被拦截）。
"""
from __future__ import annotations

from pathlib import Path

from config import config
from core.tool import tool


def _safe_resolve(filename: str) -> Path:
    root = Path(config.workspace_dir).resolve()
    target = (root / filename).resolve()
    if not str(target).startswith(str(root)):
        raise ValueError("非法路径：不允许访问 workspace 之外的文件")
    return target


@tool
def read_workspace_file(filename: str) -> str:
    """读取沙箱工作目录下的文本文件内容。
    适合读取 Agent 之前用 run_python 生成的中间结果文件。

    :param filename: 相对工作目录的文件名，如 result.txt
    """
    target = _safe_resolve(filename)
    if not target.exists():
        return f"文件不存在: {filename}"
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > 8000:
        text = text[:8000] + "\n...[过长已截断]"
    return text or "(空文件)"


@tool
def list_workspace_files() -> str:
    """列出沙箱工作目录下的所有文件及大小。"""
    root = Path(config.workspace_dir)
    files = []
    for p in sorted(root.iterdir()):
        if p.is_file():
            files.append(f"{p.name}  ({p.stat().st_size} bytes)")
    return "\n".join(files) if files else "(工作目录为空)"
