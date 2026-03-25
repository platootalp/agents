from collections import Counter
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

    # 189. 轮转数组
    def rotate(self, nums: List[int], k: int) -> None:
        k %= len(nums)
        if k == 0:
            return

        nums.reverse()
        nums[:k] = reversed(nums[:k])
        nums[k:] = reversed(nums[k:])

    # 238. 除了自身以外数组的乘积
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        le = len(nums)
        res = [1] * le

        # 前缀乘积
        for i in range(1, len(nums)):
            res[i] = res[i - 1] * nums[i - 1]

        # 后缀乘积
        suf = 1
        for i in range(len(nums) - 2, -1, -1):
            suf *= nums[i + 1]
            res[i] *= suf

        return res

    # 73. 矩阵置零
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """