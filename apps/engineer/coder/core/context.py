"""
上下文管理模块

提供 Agent 执行过程中的上下文管理，包括：
- 执行上下文（ExecutionContext）
- 变量存储和检索
- 上下文生命周期管理
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from contextvars import ContextVar
import copy


@dataclass
class ExecutionContext:
    """
    执行上下文

    存储 Agent 执行过程中的状态和数据
    """

    # 上下文 ID
    context_id: str = ""

    # 父上下文（支持嵌套）
    parent: Optional["ExecutionContext"] = None

    # 变量存储
    variables: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 执行历史
    history: List[Dict[str, Any]] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """获取变量值（优先从当前上下文，然后递归查找父上下文）"""
        if key in self.variables:
            return self.variables[key]
        if self.parent:
            return self.parent.get(key, default)
        return default

    def set(self, key: str, value: Any) -> None:
        """设置变量值"""
        self.variables[key] = value

    def update(self, data: Dict[str, Any]) -> None:
        """批量更新变量"""
        self.variables.update(data)

    def delete(self, key: str) -> bool:
        """删除变量"""
        if key in self.variables:
            del self.variables[key]
            return True
        return False

    def has(self, key: str) -> bool:
        """检查变量是否存在"""
        if key in self.variables:
            return True
        if self.parent:
            return self.parent.has(key)
        return False

    def push_history(self, event: Dict[str, Any]) -> None:
        """添加历史记录"""
        self.history.append(event)

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取历史记录"""
        if limit:
            return self.history[-limit:]
        return copy.deepcopy(self.history)

    def clear_history(self) -> None:
        """清空历史记录"""
        self.history.clear()

    def fork(self, context_id: Optional[str] = None) -> "ExecutionContext":
        """创建子上下文（分支）"""
        return ExecutionContext(
            context_id=context_id or f"{self.context_id}_fork",
            parent=self,
            variables=copy.deepcopy(self.variables),
            metadata=copy.deepcopy(self.metadata),
        )

    def flatten(self) -> Dict[str, Any]:
        """扁平化所有变量（包括父上下文的）"""
        result = {}
        if self.parent:
            result.update(self.parent.flatten())
        result.update(self.variables)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "context_id": self.context_id,
            "variables": copy.deepcopy(self.variables),
            "metadata": copy.deepcopy(self.metadata),
            "history_length": len(self.history),
        }


class ContextManager:
    """
    上下文管理器

    管理多个执行上下文的创建、存储和检索
    """

    def __init__(self):
        self._contexts: Dict[str, ExecutionContext] = {}
        self._current_context: Optional[ExecutionContext] = None

    def create(self, context_id: str, parent_id: Optional[str] = None) -> ExecutionContext:
        """创建新上下文"""
        parent = None
        if parent_id and parent_id in self._contexts:
            parent = self._contexts[parent_id]

        context = ExecutionContext(context_id=context_id, parent=parent)
        self._contexts[context_id] = context
        return context

    def get(self, context_id: str) -> Optional[ExecutionContext]:
        """获取上下文"""
        return self._contexts.get(context_id)

    def get_or_create(self, context_id: str) -> ExecutionContext:
        """获取或创建上下文"""
        if context_id not in self._contexts:
            return self.create(context_id)
        return self._contexts[context_id]

    def delete(self, context_id: str) -> bool:
        """删除上下文"""
        if context_id in self._contexts:
            del self._contexts[context_id]
            return True
        return False

    def set_current(self, context: ExecutionContext) -> None:
        """设置当前上下文"""
        self._current_context = context

    def get_current(self) -> Optional[ExecutionContext]:
        """获取当前上下文"""
        return self._current_context

    def clear(self) -> None:
        """清空所有上下文"""
        self._contexts.clear()
        self._current_context = None

    def list_contexts(self) -> List[str]:
        """列出所有上下文 ID"""
        return list(self._contexts.keys())


# 全局上下文变量（用于异步上下文传递）
current_context_var: ContextVar[Optional[ExecutionContext]] = ContextVar(
    "current_context", default=None
)


def get_current_context() -> Optional[ExecutionContext]:
    """获取当前线程/任务的上下文"""
    return current_context_var.get()


def set_current_context(context: Optional[ExecutionContext]) -> None:
    """设置当前线程/任务的上下文"""
    current_context_var.set(context)
