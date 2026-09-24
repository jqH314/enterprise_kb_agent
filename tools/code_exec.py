"""
沙箱代码执行工具
================
让 Agent 能在隔离子进程里跑 Python 做计算/数据处理，
例如：对检索到的表格数据做统计、写脚本算差值、格式化输出。
"""
from __future__ import annotations

from core.tool import tool
from tools import get_sandbox


@tool
def run_python(code: str) -> str:
    """在沙箱中执行一段 Python 代码并返回 stdout/stderr。
    适合做计算、数据处理、格式转换等。工作目录已隔离，超时 15 秒。
    代码里请用 print() 输出结果。

    :param code: 要执行的完整 Python 代码
    """
    sandbox = get_sandbox()
    result = sandbox.run_python(code)
    return str(result)
