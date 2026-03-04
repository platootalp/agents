"""
工具基础抽象类和接口定义
定义所有工具必须实现的标准接口，参考LangChain工具设计
"""

# genAI_main_start
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, Union, Type
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ValidationError
import json
import asyncio
from enum import Enum


class ToolCallbackType(Enum):
    """工具回调类型枚举"""
    ON_TOOL_START = "on_tool_start"
    ON_TOOL_END = "on_tool_end"
    ON_TOOL_ERROR = "on_tool_error"


@dataclass
class ToolResult:
    """工具执行结果
    
    封装工具执行的结果信息
    
    Attributes:
        output: 工具执行的输出结果（字符串或字典）
        success: 执行是否成功
        error: 错误信息（如果失败）
        metadata: 额外的元数据信息
    """
    output: Union[str, Dict[str, Any]]
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """转换为字符串格式
        
        Returns:
            格式化的结果字符串
        """
        if isinstance(self.output, str):
            return self.output
        return json.dumps(self.output, ensure_ascii=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            包含所有结果信息的字典
        """
        return {
            "output": self.output,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata
        }


class BaseTool(ABC):
    """工具基础抽象类
    
    所有工具必须继承此类并实现必要的抽象方法。
    提供统一的工具执行接口和参数验证机制。
    
    Attributes:
        name: 工具名称（唯一标识）
        description: 工具描述（用于LLM理解工具功能）
        args_schema: 参数模式（Pydantic BaseModel类，用于参数验证）
        return_direct: 是否直接返回工具结果（不再调用LLM）
        verbose: 是否输出详细日志
        callbacks: 回调函数字典
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = False,
        verbose: bool = False
    ):
        """初始化工具实例
        
        Args:
            name: 工具名称
            description: 工具描述
            args_schema: 参数模式（Pydantic模型类）
            return_direct: 是否直接返回结果
            verbose: 是否输出详细日志
        """
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.return_direct = return_direct
        self.verbose = verbose
        self.callbacks: Dict[ToolCallbackType, Callable] = {}
    
    @abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        """执行工具逻辑（抽象方法）
        
        子类必须实现此方法来定义工具的核心功能
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            工具执行结果（任意类型）
        
        Raises:
            NotImplementedError: 子类未实现此方法
        """
        pass
    
    async def _arun(self, *args, **kwargs) -> Any:
        """异步执行工具逻辑（可选实现）
        
        子类可以重写此方法以支持异步执行
        默认实现使用同步方法在异步环境中执行
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            工具执行结果（任意类型）
        """
        return await asyncio.to_thread(self._run, *args, **kwargs)
    
    def validate_args(self, **kwargs) -> Dict[str, Any]:
        """验证工具参数
        
        使用Pydantic模型验证输入参数
        
        Args:
            **kwargs: 待验证的参数
        
        Returns:
            验证后的参数字典
        
        Raises:
            ValidationError: 参数验证失败
        """
        if self.args_schema is None:
            return kwargs
        
        try:
            validated = self.args_schema(**kwargs)
            return validated.dict()
        except ValidationError as e:
            raise ValueError(f"参数验证失败: {e}")
    
    def run(self, *args, **kwargs) -> ToolResult:
        """运行工具（同步）
        
        执行工具并返回标准化的结果对象
        自动处理异常和回调
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            ToolResult对象，包含执行结果和状态
        """
        try:
            # 触发开始回调
            self._trigger_callback(ToolCallbackType.ON_TOOL_START, kwargs)
            
            # 验证参数
            if self.args_schema:
                validated_kwargs = self.validate_args(**kwargs)
                result = self._run(*args, **validated_kwargs)
            else:
                result = self._run(*args, **kwargs)
            
            # 构建结果对象
            tool_result = ToolResult(
                output=result,
                success=True,
                metadata={"tool_name": self.name}
            )
            
            # 触发成功回调
            self._trigger_callback(ToolCallbackType.ON_TOOL_END, tool_result)
            
            if self.verbose:
                print(f"[{self.name}] 执行成功: {tool_result.to_string()[:100]}...")
            
            return tool_result
            
        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            tool_result = ToolResult(
                output="",
                success=False,
                error=error_msg,
                metadata={"tool_name": self.name}
            )
            
            # 触发错误回调
            self._trigger_callback(ToolCallbackType.ON_TOOL_ERROR, e)
            
            if self.verbose:
                print(f"[{self.name}] 执行失败: {error_msg}")
            
            return tool_result
    
    async def arun(self, *args, **kwargs) -> ToolResult:
        """运行工具（异步）
        
        异步执行工具并返回标准化的结果对象
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            ToolResult对象，包含执行结果和状态
        """
        try:
            # 触发开始回调
            self._trigger_callback(ToolCallbackType.ON_TOOL_START, kwargs)

            # 验证参数
            if self.args_schema:
                validated_kwargs = self.validate_args(**kwargs)
                result = await self._arun(*args, **validated_kwargs)
            else:
                result = await self._arun(*args, **kwargs)
            
            # 构建结果对象
            tool_result = ToolResult(
                output=result,
                success=True,
                metadata={"tool_name": self.name}
            )
            
            # 触发成功回调
            self._trigger_callback(ToolCallbackType.ON_TOOL_END, tool_result)
            
            if self.verbose:
                print(f"[{self.name}] 执行成功: {tool_result.to_string()[:100]}...")
            
            return tool_result
            
        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            tool_result = ToolResult(
                output="",
                success=False,
                error=error_msg,
                metadata={"tool_name": self.name}
            )
            
            # 触发错误回调
            self._trigger_callback(ToolCallbackType.ON_TOOL_ERROR, e)
            
            if self.verbose:
                print(f"[{self.name}] 执行失败: {error_msg}")
            
            return tool_result
    
    def register_callback(self, callback_type: ToolCallbackType, callback: Callable):
        """注册回调函数
        
        Args:
            callback_type: 回调类型
            callback: 回调函数
        """
        self.callbacks[callback_type] = callback
    
    def _trigger_callback(self, callback_type: ToolCallbackType, data: Any):
        """触发回调函数
        
        Args:
            callback_type: 回调类型
            data: 传递给回调函数的数据
        """
        callback = self.callbacks.get(callback_type)
        if callback:
            try:
                callback(data)
            except Exception as e:
                if self.verbose:
                    print(f"回调执行失败: {e}")
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """转换为OpenAI工具格式
        
        将工具转换为OpenAI函数调用格式
        
        Returns:
            OpenAI工具格式的字典
        """
        tool_def = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
            }
        }
        
        # 添加参数模式
        if self.args_schema:
            tool_def["function"]["parameters"] = self.args_schema.schema()
        
        return tool_def
    
    def to_anthropic_tool(self) -> Dict[str, Any]:
        """转换为Anthropic工具格式
        
        将工具转换为Anthropic工具使用格式
        
        Returns:
            Anthropic工具格式的字典
        """
        tool_def = {
            "name": self.name,
            "description": self.description,
        }
        
        # 添加参数模式
        if self.args_schema:
            tool_def["input_schema"] = self.args_schema.schema()
        
        return tool_def
    
    def __call__(self, *args, **kwargs) -> ToolResult:
        """使工具实例可调用
        
        允许直接调用工具实例：tool(arg1, arg2)
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            ToolResult对象
        """
        return self.run(*args, **kwargs)
    
    def __repr__(self) -> str:
        """工具的字符串表示
        
        Returns:
            格式化的工具信息字符串
        """
        return f"Tool(name='{self.name}', description='{self.description}')"
# genAI_main_end

