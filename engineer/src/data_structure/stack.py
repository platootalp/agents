from typing import TypeVar, Generic, Iterator

T = TypeVar('T')  # 声明类型变量

class Stack(Generic[T]):
    def __init__(self):
        self._items: list[T] = []

    def push(self, item: T) -> None:
        """将元素压入栈顶"""
        self._items.append(item)

    def pop(self) -> T:
        """弹出栈顶元素"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> T:
        """查看栈顶元素（不移除）"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        """判断栈是否为空"""
        return len(self._items) == 0

    def __len__(self) -> int:
        """返回栈的大小（len(stack)）"""
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        """支持迭代（for x in stack）"""
        # 从栈顶到栈底迭代
        return reversed(self._items)

    def __repr__(self) -> str:
        return f"Stack({self._items})"

class Solution:
    # 844. Backspace String Compare
    def backspaceCompare(self, s: str, t: str) -> bool:
        sStack: Stack[str] = Stack()
        tStack: Stack[str] = Stack()

        self.add_item(s, sStack)
        self.add_item(t, tStack)
        while not sStack.is_empty() and not tStack.is_empty():
            pop1 = sStack.pop()
            pop2 = tStack.pop()
            if pop1 != pop2:
                return False
        return sStack.is_empty() and tStack.is_empty()

    def add_item(self, s: str, stack: Stack[str]):
        temp = '#'
        for c in s:
            if c != temp:
                stack.push(c)
            else:
                if not stack.is_empty():
                    stack.pop()
