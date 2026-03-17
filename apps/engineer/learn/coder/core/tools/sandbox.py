"""
沙箱执行模块

提供隔离的执行环境，限制资源访问和系统调用
"""

import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .base import ToolResult
from .builtin.shell import ShellTool


@dataclass
class SandboxConfig:
    """沙箱配置"""

    temp_dir: Optional[str] = None
    allowed_paths: List[str] = field(default_factory=list)
    read_only_paths: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    network_access: bool = False
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_files: int = 1000
    cleanup_on_exit: bool = True


class SandboxEnvironment:
    """
    沙箱执行环境

    提供隔离的文件系统和环境变量
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        初始化沙箱环境

        Args:
            config: 沙箱配置
        """
        self.config = config or SandboxConfig()
        self._temp_dir: Optional[str] = None
        self._original_env: Dict[str, str] = {}
        self._is_active = False

    def __enter__(self) -> "SandboxEnvironment":
        """进入沙箱环境"""
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出沙箱环境"""
        self.deactivate()

    def activate(self):
        """激活沙箱环境"""
        if self._is_active:
            return

        # 创建临时目录
        if self.config.temp_dir:
            self._temp_dir = self.config.temp_dir
            Path(self._temp_dir).mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = tempfile.mkdtemp(prefix="agent_sandbox_")

        # 保存原始环境变量
        self._original_env = dict(os.environ)

        # 设置沙箱环境变量
        sandbox_env = {
            "SANDBOX": "1",
            "SANDBOX_DIR": self._temp_dir,
            "HOME": self._temp_dir,
            "TMPDIR": self._temp_dir,
        }
        sandbox_env.update(self.config.env_vars)

        # 只保留必要的系统环境变量
        allowed_system_vars = ["PATH", "LANG", "LC_ALL", "TZ"]
        if self.config.network_access:
            allowed_system_vars.extend(["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"])

        for var in allowed_system_vars:
            if var in self._original_env:
                sandbox_env[var] = self._original_env[var]

        os.environ.update(sandbox_env)

        self._is_active = True

    def deactivate(self):
        """停用沙箱环境"""
        if not self._is_active:
            return

        # 恢复原始环境变量
        os.environ.clear()
        os.environ.update(self._original_env)

        # 清理临时目录
        if self.config.cleanup_on_exit and self._temp_dir:
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass

        self._is_active = False
        self._temp_dir = None

    @property
    def working_directory(self) -> str:
        """获取沙箱工作目录"""
        if not self._temp_dir:
            raise RuntimeError("沙箱环境未激活")
        return self._temp_dir

    def resolve_path(self, path: str) -> Path:
        """
        解析路径，确保在沙箱内

        Args:
            path: 相对或绝对路径

        Returns:
            Path: 解析后的沙箱内路径
        """
        if not self._temp_dir:
            raise RuntimeError("沙箱环境未激活")

        target = Path(path)
        if target.is_absolute():
            # 如果是绝对路径，在沙箱内重新创建
            target = Path(self._temp_dir) / target.relative_to(target.anchor)
        else:
            target = Path(self._temp_dir) / target

        resolved = target.resolve()
        sandbox_root = Path(self._temp_dir).resolve()

        # 确保路径在沙箱内
        try:
            resolved.relative_to(sandbox_root)
        except ValueError:
            raise ValueError(f"路径 '{path}' 超出沙箱范围")

        return resolved

    def is_path_allowed(self, path: str) -> bool:
        """
        检查路径是否允许访问

        Args:
            path: 要检查的路径

        Returns:
            bool: 是否允许访问
        """
        if not self._temp_dir:
            return False

        try:
            target = Path(path).resolve()

            # 检查是否在沙箱内
            sandbox_root = Path(self._temp_dir).resolve()
            try:
                target.relative_to(sandbox_root)
                return True
            except ValueError:
                pass

            # 检查是否在允许的额外路径内
            for allowed in self.config.allowed_paths:
                allowed_path = Path(allowed).resolve()
                try:
                    target.relative_to(allowed_path)
                    return True
                except ValueError:
                    pass

            # 检查是否是只读路径
            for ro_path in self.config.read_only_paths:
                ro = Path(ro_path).resolve()
                try:
                    target.relative_to(ro)
                    return True
                except ValueError:
                    pass

            return False
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        获取沙箱统计信息

        Returns:
            Dict: 统计信息
        """
        if not self._temp_dir:
            return {"active": False}

        try:
            total_size = 0
            file_count = 0

            for root, dirs, files in os.walk(self._temp_dir):
                for f in files:
                    fp = Path(root) / f
                    try:
                        total_size += fp.stat().st_size
                        file_count += 1
                    except Exception:
                        pass

            return {
                "active": True,
                "directory": self._temp_dir,
                "file_count": file_count,
                "total_size": total_size,
                "max_files": self.config.max_files,
                "max_file_size": self.config.max_file_size,
            }
        except Exception as e:
            return {"active": True, "error": str(e)}


@contextmanager
def sandbox_environment(
    temp_dir: Optional[str] = None,
    allowed_paths: Optional[List[str]] = None,
    read_only_paths: Optional[List[str]] = None,
    env_vars: Optional[Dict[str, str]] = None,
    network_access: bool = False,
    cleanup_on_exit: bool = True,
) -> Generator[SandboxEnvironment, None, None]:
    """
    沙箱环境上下文管理器

    Args:
        temp_dir: 临时目录路径，None则自动创建
        allowed_paths: 额外允许访问的路径列表
        read_only_paths: 只读路径列表
        env_vars: 额外的环境变量
        network_access: 是否允许网络访问
        cleanup_on_exit: 退出时是否清理临时目录

    Yields:
        SandboxEnvironment: 沙箱环境实例

    Example:
        with sandbox_environment() as sandbox:
            result = sandbox.execute("ls -la")
    """
    config = SandboxConfig(
        temp_dir=temp_dir,
        allowed_paths=allowed_paths or [],
        read_only_paths=read_only_paths or [],
        env_vars=env_vars or {},
        network_access=network_access,
        cleanup_on_exit=cleanup_on_exit,
    )

    sandbox = SandboxEnvironment(config)
    try:
        sandbox.activate()
        yield sandbox
    finally:
        sandbox.deactivate()


class SandboxedShellTool(ShellTool):
    """
    沙箱化的Shell工具

    在隔离的沙箱环境中执行命令
    """

    def __init__(
        self,
        sandbox_config: Optional[SandboxConfig] = None,
        allowed_paths: Optional[List[str]] = None,
        allowed_env_vars: Optional[List[str]] = None,
    ):
        """
        初始化沙箱Shell工具

        Args:
            sandbox_config: 沙箱配置
            allowed_paths: 允许的路径（除沙箱外）
            allowed_env_vars: 允许的环境变量
        """
        # 初始化父类，但不允许任何额外路径
        super().__init__(
            allowed_paths=[],  # 沙箱工具在沙箱内执行
            allowed_env_vars=allowed_env_vars or ["PATH", "HOME", "TMPDIR", "SANDBOX"],
        )
        self.sandbox_config = sandbox_config or SandboxConfig()

        # 添加沙箱临时目录到允许路径
        self.allowed_paths = allowed_paths or []

    def _run(
        self,
        command: str,
        timeout: int = 60,
        working_dir: Optional[str] = None,
        env_vars: Optional[dict] = None,
    ) -> str:
        """
        在沙箱中执行shell命令
        """
        with sandbox_environment(
            temp_dir=self.sandbox_config.temp_dir,
            allowed_paths=self.allowed_paths + self.sandbox_config.allowed_paths,
            read_only_paths=self.sandbox_config.read_only_paths,
            env_vars=env_vars,
            network_access=self.sandbox_config.network_access,
            cleanup_on_exit=self.sandbox_config.cleanup_on_exit,
        ) as sandbox:
            # 在沙箱中执行命令
            return super()._run(
                command=command,
                timeout=timeout,
                working_dir=sandbox.working_directory,
                env_vars=env_vars,
            )

    async def _arun(
        self,
        command: str,
        timeout: int = 60,
        working_dir: Optional[str] = None,
        env_vars: Optional[dict] = None,
    ) -> str:
        """
        在沙箱中异步执行shell命令
        """
        # 由于sandbox_environment是同步上下文管理器，我们需要手动管理
        sandbox = SandboxEnvironment(self.sandbox_config)
        sandbox.activate()

        try:
            result = await super()._arun(
                command=command,
                timeout=timeout,
                working_dir=sandbox.working_directory,
                env_vars=env_vars,
            )
            return result
        finally:
            sandbox.deactivate()


def create_sandboxed_tools(
    allowed_paths: Optional[List[str]] = None,
    network_access: bool = False,
) -> Dict[str, Any]:
    """
    创建沙箱化的工具集合

    Args:
        allowed_paths: 额外允许的路径
        network_access: 是否允许网络访问

    Returns:
        Dict: 工具字典
    """
    config = SandboxConfig(
        allowed_paths=allowed_paths or [],
        network_access=network_access,
    )

    return {
        "shell": SandboxedShellTool(sandbox_config=config),
    }


# 便捷的独立函数
def execute_in_sandbox(
    command: str,
    timeout: int = 60,
    allowed_paths: Optional[List[str]] = None,
    env_vars: Optional[Dict[str, str]] = None,
) -> ToolResult:
    """
    在沙箱中执行命令

    Args:
        command: 要执行的命令
        timeout: 超时时间
        allowed_paths: 允许访问的路径
        env_vars: 环境变量

    Returns:
        ToolResult: 执行结果
    """
    config = SandboxConfig(
        allowed_paths=allowed_paths or [],
        env_vars=env_vars or {},
    )

    tool = SandboxedShellTool(sandbox_config=config)
    return tool.run(command=command, timeout=timeout)


class ResourceLimiter:
    """
    资源限制器

    限制CPU、内存等资源使用
    """

    def __init__(
        self,
        max_cpu_time: Optional[int] = None,
        max_memory_mb: Optional[int] = None,
        max_processes: Optional[int] = None,
    ):
        """
        初始化资源限制器

        Args:
            max_cpu_time: 最大CPU时间（秒）
            max_memory_mb: 最大内存使用（MB）
            max_processes: 最大进程数
        """
        self.max_cpu_time = max_cpu_time
        self.max_memory_mb = max_memory_mb
        self.max_processes = max_processes

    def apply_limits(self):
        """应用资源限制（仅在Unix系统有效）"""
        try:
            import resource

            if self.max_cpu_time:
                resource.setrlimit(resource.RLIMIT_CPU, (self.max_cpu_time, self.max_cpu_time))

            if self.max_memory_mb:
                max_bytes = self.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

            if self.max_processes:
                resource.setrlimit(resource.RLIMIT_NPROC, (self.max_processes, self.max_processes))

        except (ImportError, OSError):
            pass


def create_restricted_shell(
    allowed_commands: Optional[List[str]] = None,
    allowed_paths: Optional[List[str]] = None,
    max_timeout: int = 60,
) -> ShellTool:
    """
    创建受限的shell工具

    Args:
        allowed_commands: 允许的命令列表
        allowed_paths: 允许的路径列表
        max_timeout: 最大超时时间

    Returns:
        ShellTool: 配置好的受限shell工具
    """
    shell = ShellTool(
        allowed_paths=allowed_paths,
        allowed_env_vars=["PATH", "HOME", "USER", "LANG"],
    )

    if allowed_commands:
        shell.allowed_commands = allowed_commands

    return shell
