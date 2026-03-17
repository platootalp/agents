"""
输入过滤安全模块

提供多层输入验证和过滤机制，防止路径遍历、SQL注入等攻击
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Union

from pydantic import BaseModel, Field, validator


class InputFilter(ABC):
    """输入过滤器基类"""

    @abstractmethod
    def filter(self, input_value: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        过滤输入值

        Args:
            input_value: 输入字符串
            context: 上下文信息（如参数名、工具名等）

        Returns:
            str: 过滤后的值，或抛出SecurityError
        """
        pass

    @abstractmethod
    def validate(self, input_value: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        验证输入是否合法

        Args:
            input_value: 输入字符串
            context: 上下文信息

        Returns:
            bool: 是否通过验证
        """
        pass


class SecurityError(Exception):
    """安全错误异常"""

    def __init__(
        self, message: str, filter_name: Optional[str] = None, context: Optional[Dict] = None
    ):
        super().__init__(message)
        self.filter_name = filter_name
        self.context = context or {}


class RegexFilter(InputFilter):
    """
    正则表达式过滤器

    基于正则表达式模式匹配进行输入验证
    """

    def __init__(
        self,
        name: str,
        pattern: Union[str, Pattern],
        allow_mode: bool = True,
        error_message: Optional[str] = None,
    ):
        """
        初始化正则过滤器

        Args:
            name: 过滤器名称
            pattern: 正则表达式模式
            allow_mode: True表示允许匹配的内容，False表示禁止匹配的内容
            error_message: 自定义错误信息
        """
        self.name = name
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.allow_mode = allow_mode
        self.error_message = error_message or f"输入未通过 {name} 验证"

    def filter(self, input_value: str, context: Optional[Dict[str, Any]] = None) -> str:
        if not self.validate(input_value, context):
            raise SecurityError(self.error_message, filter_name=self.name, context=context)
        return input_value

    def validate(self, input_value: str, context: Optional[Dict[str, Any]] = None) -> bool:
        match = self.pattern.search(input_value)
        if self.allow_mode:
            return match is not None
        else:
            return match is None


class PathTraversalFilter(InputFilter):
    """
    路径遍历过滤器

    防止路径遍历攻击（如 ../../../etc/passwd）
    """

    # 危险模式：路径遍历相关
    DANGEROUS_PATTERNS = [
        r"\.\./",  # ../
        r"\.\.\\",  # ..\
        r"%2e%2e[/\\]",  # URL编码的 ../
        r"\.{2,}[/\\]",  # 两个或多个点
        r"~",  # 家目录
        r"\$\w+",  # 环境变量
    ]

    def __init__(
        self,
        allowed_base_paths: Optional[List[str]] = None,
        allow_absolute: bool = False,
    ):
        """
        初始化路径过滤器

        Args:
            allowed_base_paths: 允许的基础路径列表
            allow_absolute: 是否允许绝对路径
        """
        self.allowed_base_paths = allowed_base_paths or []
        self.allow_absolute = allow_absolute
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]

    def filter(self, input_value: str, context: Optional[Dict[str, Any]] = None) -> str:
        if not self.validate(input_value, context):
            raise SecurityError(
                f"路径包含不安全内容: {input_value}",
                filter_name="PathTraversalFilter",
                context=context,
            )
        return input_value

    def validate(self, input_value: str, context: Optional[Dict[str, Any]] = None) -> bool:
        # 检查危险模式
        for pattern in self._patterns:
            if pattern.search(input_value):
                return False

        from pathlib import Path

        try:
            path = Path(input_value)

            # 检查绝对路径
            if path.is_absolute() and not self.allow_absolute:
                return False

            # 检查是否在允许的基路径内
            if self.allowed_base_paths and path.is_absolute():
                resolved = path.resolve()
                for base in self.allowed_base_paths:
                    base_path = Path(base).resolve()
                    try:
                        resolved.relative_to(base_path)
                        return True
                    except ValueError:
                        continue
                return False

            return True
        except Exception:
            return False


class SQLInjectionFilter(InputFilter):
    """
    SQL注入过滤器

    检测和阻止常见的SQL注入攻击模式
    """

    # SQL注入危险模式
    SQL_PATTERNS = [
        r"(\s|^)(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|TABLE|DATABASE)(\s|$)",
        r"(\s|^)(OR|AND)\s+['\"\d]\s*=\s*['\"\d]",
        r"--",
        r"/\*",
        r"\*/",
        r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP)",
        r"'\s*OR\s*['\"\d]",
        r"\"\s*OR\s*['\"\d]",
        r"'\s*AND\s*['\"\d]",
        r"\"\s*AND\s*['\"\d]",
        r"1\s*=\s*1",
        r"1\s*=\s*'1'",
        r"SLEEP\s*\(",
        r"BENCHMARK\s*\(",
        r"WAITFOR\s+DELAY",
        r"xp_cmdshell",
        r"sp_executesql",
    ]

    def __init__(self, case_sensitive: bool = False):
        """
        初始化SQL注入过滤器

        Args:
            case_sensitive: 是否区分大小写
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        self._patterns = [re.compile(p, flags) for p in self.SQL_PATTERNS]

    def filter(self, input_value: str, context: Optional[Dict[str, Any]] = None) -> str:
        if not self.validate(input_value, context):
            raise SecurityError(
                "检测到潜在的SQL注入攻击",
                filter_name="SQLInjectionFilter",
                context=context,
            )
        return input_value

    def validate(self, input_value: str, context: Optional[Dict[str, Any]] = None) -> bool:
        for pattern in self._patterns:
            if pattern.search(input_value):
                return False
        return True


class CommandInjectionFilter(InputFilter):
    """
    命令注入过滤器

    检测和阻止命令注入攻击
    """

    # 命令注入危险字符和模式
    DANGEROUS_CHARS = [";", "|", "&", "$", "`", "(", ")", "{", "}", "<", ">"]
    DANGEROUS_PATTERNS = [
        r"`[^`]*`",  # 反引号命令替换
        r"\$\([^)]*\)",  # $() 命令替换
        r"\$\{[^}]*\}",  # ${} 变量扩展
        r"\|\s*\w+",  # 管道
        r";\s*\w+",  # 命令分隔
        r"&&\s*\w+",  # 逻辑与
        r"\|\|\s*\w+",  # 逻辑或
    ]

    def __init__(self, strict_mode: bool = False):
        """
        初始化命令注入过滤器

        Args:
            strict_mode: 严格模式，禁止更多特殊字符
        """
        self.strict_mode = strict_mode
        self._patterns = [re.compile(p) for p in self.DANGEROUS_PATTERNS]

        if strict_mode:
            self.DANGEROUS_CHARS.extend(["'", '"', "\\", "\n", "\r"])

    def filter(self, input_value: str, context: Optional[Dict[str, Any]] = None) -> str:
        if not self.validate(input_value, context):
            raise SecurityError(
                "检测到潜在的命令注入攻击",
                filter_name="CommandInjectionFilter",
                context=context,
            )
        return input_value

    def validate(self, input_value: str, context: Optional[Dict[str, Any]] = None) -> bool:
        # 检查危险字符
        for char in self.DANGEROUS_CHARS:
            if char in input_value:
                return False

        # 检查危险模式
        for pattern in self._patterns:
            if pattern.search(input_value):
                return False

        return True


class CompositeFilter(InputFilter):
    """
    组合过滤器

    组合多个过滤器，按顺序执行
    """

    def __init__(
        self,
        filters: List[InputFilter],
        mode: str = "all",  # "all" 或 "any"
    ):
        """
        初始化组合过滤器

        Args:
            filters: 过滤器列表
            mode: 验证模式，"all"表示全部通过，"any"表示任一通过
        """
        self.filters = filters
        self.mode = mode

    def filter(self, input_value: str, context: Optional[Dict[str, Any]] = None) -> str:
        if self.mode == "all":
            for f in self.filters:
                input_value = f.filter(input_value, context)
            return input_value
        else:  # any mode
            last_error = None
            for f in self.filters:
                try:
                    return f.filter(input_value, context)
                except SecurityError as e:
                    last_error = e
            raise last_error or SecurityError("所有过滤器都拒绝了输入")

    def validate(self, input_value: str, context: Optional[Dict[str, Any]] = None) -> bool:
        if self.mode == "all":
            return all(f.validate(input_value, context) for f in self.filters)
        else:
            return any(f.validate(input_value, context) for f in self.filters)


class FilterConfig(BaseModel):
    """过滤器配置"""

    enable_path_filter: bool = Field(default=True, description="启用路径遍历过滤")
    enable_sql_filter: bool = Field(default=True, description="启用SQL注入过滤")
    enable_command_filter: bool = Field(default=True, description="启用命令注入过滤")
    allowed_base_paths: List[str] = Field(default_factory=list, description="允许的基础路径")
    custom_patterns: Dict[str, str] = Field(default_factory=dict, description="自定义正则模式")
    strict_mode: bool = Field(default=False, description="严格模式")


class SecureInputValidator:
    """
    安全输入验证器

    提供统一的输入验证接口
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        """
        初始化验证器

        Args:
            config: 过滤器配置
        """
        self.config = config or FilterConfig()
        self._filters: Dict[str, InputFilter] = {}
        self._setup_filters()

    def _setup_filters(self):
        """根据配置设置过滤器"""
        if self.config.enable_path_filter:
            self._filters["path"] = PathTraversalFilter(
                allowed_base_paths=self.config.allowed_base_paths
            )

        if self.config.enable_sql_filter:
            self._filters["sql"] = SQLInjectionFilter()

        if self.config.enable_command_filter:
            self._filters["command"] = CommandInjectionFilter(strict_mode=self.config.strict_mode)

        # 添加自定义过滤器
        for name, pattern in self.config.custom_patterns.items():
            self._filters[name] = RegexFilter(
                name=name,
                pattern=pattern,
                allow_mode=False,
                error_message=f"输入包含禁止内容: {name}",
            )

    def validate(self, input_value: str, filter_types: Optional[List[str]] = None) -> bool:
        """
        验证输入

        Args:
            input_value: 输入值
            filter_types: 要应用的过滤器类型，None表示全部

        Returns:
            bool: 是否通过验证
        """
        filters_to_apply = filter_types or list(self._filters.keys())

        for filter_type in filters_to_apply:
            filter_obj = self._filters.get(filter_type)
            if filter_obj and not filter_obj.validate(input_value):
                return False

        return True

    def filter(
        self,
        input_value: str,
        filter_types: Optional[List[str]] = None,
        context: Optional[Dict] = None,
    ) -> str:
        """
        过滤输入

        Args:
            input_value: 输入值
            filter_types: 要应用的过滤器类型，None表示全部
            context: 上下文信息

        Returns:
            str: 过滤后的值

        Raises:
            SecurityError: 验证失败时抛出
        """
        filters_to_apply = filter_types or list(self._filters.keys())
        result = input_value

        for filter_type in filters_to_apply:
            filter_obj = self._filters.get(filter_type)
            if filter_obj:
                result = filter_obj.filter(result, context)

        return result

    def add_filter(self, name: str, filter_obj: InputFilter):
        """
        添加自定义过滤器

        Args:
            name: 过滤器名称
            filter_obj: 过滤器实例
        """
        self._filters[name] = filter_obj

    def remove_filter(self, name: str):
        """
        移除过滤器

        Args:
            name: 过滤器名称
        """
        self._filters.pop(name, None)


def with_input_filter(
    filter_obj: Optional[InputFilter] = None,
    filter_types: Optional[List[str]] = None,
    param_name: Optional[str] = None,
):
    """
    输入过滤装饰器

    用于装饰工具函数，自动过滤输入参数

    Args:
        filter_obj: 过滤器实例
        filter_types: 过滤器类型列表
        param_name: 要过滤的参数名，None表示所有字符串参数

    Returns:
        Callable: 装饰器函数
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            validator = SecureInputValidator()

            # 过滤kwargs
            if param_name:
                if param_name in kwargs and isinstance(kwargs[param_name], str):
                    kwargs[param_name] = validator.filter(
                        kwargs[param_name], filter_types, context={"param": param_name}
                    )
            else:
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        kwargs[key] = validator.filter(value, filter_types, context={"param": key})

            return func(*args, **kwargs)

        return wrapper

    return decorator


# 预定义的常用过滤器组合
COMMON_FILTERS = {
    "path": PathTraversalFilter(),
    "sql": SQLInjectionFilter(),
    "command": CommandInjectionFilter(),
    "strict_command": CommandInjectionFilter(strict_mode=True),
}


def get_filter(name: str) -> Optional[InputFilter]:
    """
    获取预定义过滤器

    Args:
        name: 过滤器名称

    Returns:
        InputFilter: 过滤器实例，不存在则返回None
    """
    return COMMON_FILTERS.get(name)


def create_default_validator(allowed_paths: Optional[List[str]] = None) -> SecureInputValidator:
    """
    创建默认验证器

    Args:
        allowed_paths: 允许的路径列表

    Returns:
        SecureInputValidator: 配置好的验证器
    """
    config = FilterConfig(
        enable_path_filter=True,
        enable_sql_filter=True,
        enable_command_filter=True,
        allowed_base_paths=allowed_paths or [],
    )
    return SecureInputValidator(config)
