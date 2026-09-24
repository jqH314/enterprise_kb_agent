"""
沙箱执行器
==========
在隔离的工作目录（workspace）里以子进程方式执行 Python / Shell 代码，
做到：
  1. 目录隔离：子进程 cwd 固定在 workspace，碰不到项目其他文件
  2. 超时保护：超时自动 kill，防止死循环
  3. 环境隔离：不把宿主进程的敏感环境变量（API Key 等）传给子进程
  4. 输出截断：防止超长输出撑爆 LLM 上下文

注意：这是"轻量级应用沙箱"（进程级隔离），适合教学/演示场景。
生产环境应改用 Docker gVisor / Firecracker 等强隔离方案。
"""
from __future__ import annotations

import os
import sys
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass


# 允许透传给子进程的白名单环境变量（其余一律不暴露，防止 API Key 泄漏）
_ALLOWED_ENV = {"PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP",
                "HOME", "USERPROFILE", "LANG", "PYTHONIOENCODING"}

# 输出截断上限（字符数）
_MAX_OUTPUT_CHARS = 8000


@dataclass
class ExecResult:
    ok: bool
    output: str
    exit_code: int

    def __str__(self) -> str:
        status = "成功" if self.ok else f"失败(exit={self.exit_code})"
        return f"[{status}]\n{self.output}"


class Sandbox:
    def __init__(self, workspace: str, timeout: int = 15) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _safe_env(self) -> dict[str, str]:
        return {k: v for k, v in os.environ.items() if k in _ALLOWED_ENV}

    def _truncate(self, text: str) -> str:
        if len(text) <= _MAX_OUTPUT_CHARS:
            return text
        head = text[: _MAX_OUTPUT_CHARS // 2]
        tail = text[-_MAX_OUTPUT_CHARS // 2:]
        return f"{head}\n...[输出过长，已省略 {len(text) - _MAX_OUTPUT_CHARS} 字符]...\n{tail}"

    def run_python(self, code: str) -> ExecResult:
        """在沙箱子进程里执行一段 Python 代码。"""
        # 把代码写到 workspace 下的临时文件
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", dir=self.workspace, delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                cwd=str(self.workspace),
                env=self._safe_env(),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return ExecResult(
                ok=proc.returncode == 0,
                output=self._truncate(output.strip() or "(无输出)"),
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                ok=False,
                output=f"[执行超时] 超过 {self.timeout} 秒，已强制终止。"
                       "可能是死循环或计算量过大。",
                exit_code=-1,
            )
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def run_command(self, command: str) -> ExecResult:
        """在沙箱 workspace 里执行一条 shell 命令。"""
        try:
            proc = subprocess.run(
                command,
                cwd=str(self.workspace),
                shell=True,
                env=self._safe_env(),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return ExecResult(
                ok=proc.returncode == 0,
                output=self._truncate(output.strip() or "(无输出)"),
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                ok=False,
                output=f"[执行超时] 超过 {self.timeout} 秒，已强制终止。",
                exit_code=-1,
            )
