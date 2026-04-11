from collections import Counter, deque
from typing import List, Optional

from apps.engineer.framework.data_structure.linked_list import ListNode


class Trending100:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        l = len(nums)
        res = list()
        for i in range(l):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            j = i + 1
            k = l - 1
            while j < k:
                if nums[j] + nums[k] > -nums[i]:
                    k -= 1
                elif nums[j] + nums[k] < -nums[i]:
                    j += 1
                else:
                    res.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
        return res

    # 3. 无重复字符的最长子串
    def lengthOfLongestSubstring(self, s: str) -> int:
        length = len(s)
        ans = 0
        r = 0
        st = set()
        for l in range(length):
            while r < length and s[r] not in st:
                st.add(s[r])
                r += 1
            ans = max(ans, r - l)
            st.remove(s[l])
        return ans

    # 428 找到字符串中所有字母异位词
    def findAnagrams(self, s: str, p: str) -> List[int]:
        len_s = len(s)
        len_p = len(p)
        window = Counter()
        count = Counter(p)

        r = 0
        ans = []
        for l in range(len_s):
            c = r - l
            # 扩大窗口到 len_p 大小
            while r < len_s and c < len_p:
                window[s[r]] += 1
                r += 1
                c += 1
            if c == len_p and window == count:
                ans.append(l)
            # 缩小窗口
            window[s[l]] -= 1

        return ans

    # 206. 反转链表（递归法-尾插法）
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 边界
        if not head or not head.next:
            return head

        # 翻转 head 之后的节点
        rev_head = self.reverseList(head.next)
        tail = head.next
        tail.next = head
        head.next = None

        return rev_head

    # 206. 反转链表（迭代法-头插法）
    def reverseList2(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        cur = head

        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp

        return pre

    # 53. 最大子数组和
    def maxSubArray(self, nums: List[int]) -> int:
        # f[x]为以x结尾的子数组的最大和
        res = nums[0]
        dp = []

        for i, num in enumerate(nums):
            if i == 0:
                dp[0] = num
            else:
                dp[i] = max(dp[i - 1], 0) + num
            res = max(res, dp[i])

        return res

    # 56. 合并区间
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:

        intervals.sort(key=lambda p: p[0])
        res = []
        for i in intervals:
            # 能够合并
            if res and i[0] <= res[-1][1]:
                res[-1][1] = max(i[1], res[-1][1])
            else:
                res.append(i)
        return res

    def isPalindrome(self, head: Optional[ListNode]) -> bool:

        # 翻转链表
        p = self.reverseList(head)

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        ans = []
        q = deque([root])
        while q:
            vals = []
            for _ in range(len(q)):
                node = q.popleft()
                vals.append(node.val)
                if node.left:  q.append(node.left)
                if node.right: q.append(node.right)
            ans.append(vals)
        return ans



if __name__ == '__main__':
    matrix = [
        [1, 0, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]

    # 1. 检查行
    # 第 1 行有 0 -> True
    # 第 2 行没 0 -> False
    # 第 3 行有 0 -> True
    row_has_zero = [0 in row for row in matrix]
    # 结果：[True, False, True]
    print(row_has_zero)
    # 2. 检查列
    # zip(*matrix) 会把列提取出来：
    # 第 1 列：(1, 4, 7) -> 无 0 -> False
    # 第 2 列：(0, 5, 8) -> 有 0 -> True
    # 第 3 列：(3, 6, 0) -> 有 0 -> True
    col_has_zero = [0 in col for col in zip(*matrix)]
    # 结果：[False, True, True]
    print(col_has_zero)