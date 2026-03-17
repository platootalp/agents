"""
Shell工具 - 执行bash命令

提供安全的shell命令执行功能
"""

import asyncio
import subprocess
from typing import List, Optional

from pydantic import BaseModel, Field

from ..base import BaseTool, tool


class ShellInput(BaseModel):
    """Shell命令输入参数"""

    command: str = Field(description="要执行的shell命令")
    timeout: Optional[int] = Field(default=60, description="超时时间（秒）", ge=1, le=300)
    working_dir: Optional[str] = Field(default=None, description="工作目录")
    env_vars: Optional[dict] = Field(default=None, description="环境变量")


class ShellOutput(BaseModel):
    """Shell命令输出"""

    stdout: str = Field(description="标准输出")
    stderr: str = Field(description="标准错误")
    returncode: int = Field(description="返回码")
    command: str = Field(description="执行的命令")
    elapsed_ms: float = Field(description="执行时间（毫秒）")


class ShellTool(BaseTool):
    """
    Shell命令执行工具

    安全地执行shell命令并返回结果，支持最小权限执行

    Usage:
        shell = ShellTool(
            allowed_paths=["/home/user/projects"],
            allowed_env_vars=["PATH", "HOME", "USER"]
        )
        result = shell.run(command="ls -la", timeout=30)
    """

    DANGEROUS_COMMANDS = [
        "rm -rf /",
        "rm -rf /*",
        "> /dev/sda",
        "mkfs",
        "dd if=/dev/zero",
        ":(){ :|:& };:",
    ]

    def __init__(
        self,
        allowed_paths: Optional[List[str]] = None,
        allowed_env_vars: Optional[List[str]] = None,
    ):
        super().__init__(
            name="shell",
            description="执行shell/bash命令。支持指定超时时间、工作目录和环境变量。",
            args_schema=ShellInput,
        )
        self.allowed_commands: Optional[List[str]] = None
        self.blocked_commands: List[str] = self.DANGEROUS_COMMANDS.copy()
        self.allowed_paths = allowed_paths
        self.allowed_env_vars = allowed_env_vars

    def _validate_command(self, command: str) -> None:
        """
        验证命令安全性

        Args:
            command: 要验证的命令

        Raises:
            ValueError: 如果命令不安全
        """
        # 检查危险命令
        cmd_lower = command.lower().strip()
        for dangerous in self.blocked_commands:
            if dangerous in cmd_lower:
                raise ValueError(f"命令包含危险操作: {dangerous}")

        if self.allowed_commands:
            base_cmd = cmd_lower.split()[0] if cmd_lower.split() else ""
            if base_cmd not in self.allowed_commands:
                raise ValueError(f"命令 '{base_cmd}' 不在允许列表中")

    def _validate_path_access(self, path: Optional[str]) -> None:
        """
        验证路径访问权限

        Args:
            path: 要验证的路径

        Raises:
            ValueError: 如果路径不在允许的范围内
        """
        if not path or not self.allowed_paths:
            return

        from pathlib import Path

        target_path = Path(path).expanduser().resolve()

        for allowed_path in self.allowed_paths:
            allowed = Path(allowed_path).expanduser().resolve()
            try:
                target_path.relative_to(allowed)
                return
            except ValueError:
                continue

        raise ValueError(f"路径 '{path}' 不在允许的访问范围内")

    def _sanitize_env_vars(self, env_vars: Optional[dict]) -> dict:
        """
        清理环境变量，只保留允许的变量

        Args:
            env_vars: 原始环境变量

        Returns:
            dict: 清理后的环境变量
        """
        if not env_vars:
            return env_vars

        if not self.allowed_env_vars:
            return env_vars

        return {key: value for key, value in env_vars.items() if key in self.allowed_env_vars}

    def _run(
        self,
        command: str,
        timeout: int = 60,
        working_dir: Optional[str] = None,
        env_vars: Optional[dict] = None,
    ) -> str:
        """
        执行shell命令

        Args:
            command: 命令字符串
            timeout: 超时时间
            working_dir: 工作目录
            env_vars: 环境变量

        Returns:
            str: JSON格式的执行结果
        """
        import json
        import time

        start_time = time.time()

        self._validate_command(command)
        self._validate_path_access(working_dir)
        sanitized_env = self._sanitize_env_vars(env_vars)

        try:
            env = None
            if sanitized_env:
                import os

                env = os.environ.copy()
                env.update(sanitized_env)

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
                env=env,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            output = ShellOutput(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                command=command,
                elapsed_ms=elapsed_ms,
            )

            return json.dumps(output.model_dump(), ensure_ascii=False, indent=2)

        except subprocess.TimeoutExpired:
            elapsed_ms = (time.time() - start_time) * 1000
            output = ShellOutput(
                stdout="",
                stderr=f"命令执行超时（{timeout}秒）",
                returncode=-1,
                command=command,
                elapsed_ms=elapsed_ms,
            )
            return json.dumps(output.model_dump(), ensure_ascii=False, indent=2)

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            output = ShellOutput(
                stdout="",
                stderr=str(e),
                returncode=-1,
                command=command,
                elapsed_ms=elapsed_ms,
            )
            return json.dumps(output.model_dump(), ensure_ascii=False, indent=2)

    async def _arun(
        self,
        command: str,
        timeout: int = 60,
        working_dir: Optional[str] = None,
        env_vars: Optional[dict] = None,
    ) -> str:
        """
        异步执行shell命令
        """
        import json
        import time

        start_time = time.time()

        self._validate_command(command)
        self._validate_path_access(working_dir)
        sanitized_env = self._sanitize_env_vars(env_vars)

        try:
            env = None
            if sanitized_env:
                import os

                env = os.environ.copy()
                env.update(sanitized_env)

            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )

                elapsed_ms = (time.time() - start_time) * 1000

                output = ShellOutput(
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                    returncode=process.returncode,
                    command=command,
                    elapsed_ms=elapsed_ms,
                )

                return json.dumps(output.model_dump(), ensure_ascii=False, indent=2)

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

                elapsed_ms = (time.time() - start_time) * 1000
                output = ShellOutput(
                    stdout="",
                    stderr=f"命令执行超时（{timeout}秒）",
                    returncode=-1,
                    command=command,
                    elapsed_ms=elapsed_ms,
                )
                return json.dumps(output.model_dump(), ensure_ascii=False, indent=2)

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            output = ShellOutput(
                stdout="",
                stderr=str(e),
                returncode=-1,
                command=command,
                elapsed_ms=elapsed_ms,
            )
            return json.dumps(output.model_dump(), ensure_ascii=False, indent=2)


# 便捷的装饰器版本
@tool(name="bash", description="执行bash/shell命令，支持超时和工作目录设置")
def bash(command: str, timeout: int = 60, working_dir: Optional[str] = None) -> str:
    """
    便捷的bash命令执行

    Args:
        command: 命令
        timeout: 超时时间
        working_dir: 工作目录
    """
    shell = ShellTool()
    result = shell.run(command=command, timeout=timeout, working_dir=working_dir)
    return str(result.output) if result.success else f"Error: {result.error}"
