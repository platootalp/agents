from collections import defaultdict
from math import gcd
from typing import Optional, List

from data_structure.stack import Stack


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """
        遍历链表
    """

    # 1290 二进制链表转整数
    def getDecimalValue(self, head: Optional[ListNode]) -> int:
        ans = 0
        p = head
        while p != None:
            ans += ans * 2 + p.val
            p = p.next
        return ans

    # 725 分隔链表
    def splitListToParts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        # Step 1: Get the length of the list
        length = 0
        cur = head
        while cur:
            length += 1
            cur = cur.next

        # Step 2: Determine size of each part
        part_size = length // k
        remainder = length % k  # Extra nodes to distribute

        res = []
        current = head
        for i in range(k):
            part_head = current
            current_part_size = part_size + (1 if i < remainder else 0)

            # Traverse current part
            for j in range(current_part_size - 1):
                if current:
                    current = current.next

            # Cut the list
            if current:
                next_part = current.next
                current.next = None
                current = next_part

            res.append(part_head)

        return res

    # 817 链表组件
    def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
        dict = defaultdict()
        for num in nums:
            dict[num] += 1

    """
        删除节点
    """

    # 203 移除链表元素
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        p = dummy

        while p.next:
            if p.next.val == val:
                p.next = p.next.next
            else:
                p = p.next

        return dummy.next

    # 3217 从链表中移除在数组中存在的节点
    def modifiedList(self, nums: List[int], head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head

        p = dummy
        dit = set(nums)

        while p.next:
            if p.next.val in dit:
                p.next = p.next.next
            else:
                p = p.next

        return dummy.next

    # 82 删除排序链表中的重复元素 II
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = dummy = ListNode(next=head)

        while cur.next and cur.next.next:
            val = cur.next.val
            if cur.next.next.val == val:
                # 将中间节点相等的节点全部删除
                while cur.next and cur.next.val == val:
                    cur.next = cur.next.next
            else:
                cur = cur.next

        return dummy.next

    # 2487 从链表中移除节点
    def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        ans = ListNode(float('inf'))  # 哨兵节点，值为正无穷
        stack = Stack[ListNode]()  # 使用你的 Stack 类
        stack.push(ans)  # 初始压入哨兵节点

        while head:
            # 维护单调栈：弹出所有比当前节点小的栈顶节点
            while not stack.is_empty() and head.val > stack.peek().val:
                stack.pop()
            # 更新指针并压入当前节点
            stack.peek().next = head
            stack.push(head)
            head = head.next

        return ans.next  # 返回处理后的链表头

    """
        插入节点
    """

    # 2807 在链表中插入最大公约数
    def insertGreatestCommonDivisors(self, head: Optional[ListNode]) -> Optional[ListNode]:

        dummy = ListNode(next=head)
        while head and head.next:
            q = head.next
            gcd_node = ListNode(next=q)
            gcd_node.val = gcd(head.val, q.val)
            head.next = gcd_node
            head = q

        return dummy.next
