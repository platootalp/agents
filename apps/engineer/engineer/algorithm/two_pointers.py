from typing import List

from data_structure.stack import Stack


class TwoPointers:
    """
        1.单序列双指针
    """
    """ 相向双指针（对撞指针）"""

    # 1750 Minimum Length of String After Deleting Similar Ends
    def minimumLength(self, s: str) -> int:
        left, right = 0, len(s) - 1
        while left < right and s[left] == s[right]:
            c = s[left]
            while left <= right and s[left] == c:
                left += 1
            while right >= left and s[right] == c:
                right -= 1
        return right - left + 1

    # 658 Find K Closest Elements
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        left, right = 0, len(arr) - 1

        while left < right:
            if right - left + 1 == k:
                break
            if abs(arr[left] - x) <= abs(arr[right] - x):
                right -= 1
            else:
                left += 1
        return arr[left:right + 1]

    # 167 Two Sum II - Input Array Is Sorted
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1

        while left < right:
            if numbers[left] + numbers[right] < target:
                left += 1
            elif numbers[left] + numbers[right] > target:
                right -= 1
            else:
                return [left, right]

        return []

    # 2563 Count the Number of Fair Pairs
    def countFairPairs(self, nums: List[int], lower: int, upper: int) -> int:
        pass

    # LCP28 采购方案
    def purchasePlans(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        nums.sort()

        ans = 0
        while left < right:
            while left < right and nums[left] + nums[right] > target:
                right -= 1
            ans += (right - left) % 1000000007
            left += 1
        return ans

    def threeSumMulti(self, arr: List[int], target: int) -> int:
        arr.sort()
        ans = 0

        for i, a in enumerate(arr):
            left, right = i + 1, len(arr) - 1
            t = target - a
            if t <= 0:
                continue
            while left < right:
                if arr[left] + arr[right] < t:
                    left += 1
                elif arr[left] + arr[right] > t:
                    right -= 1
                else:
                    tl, tr = left, right
                    al, ar = arr[left], arr[right]
                    while left < right and arr[left] == al:
                        left += 1
                    while right >= left and arr[right] == ar:
                        right -= 1
                    ans = (ans + (left - tl) * (tr - right)) % 1000000007

        return ans

    """ 原地修改（快慢指针）"""

    # 27 Remove Element
    def removeElement(self, nums: List[int], val: int) -> int:
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
        return slow

    # 26 Remove Duplicates from Sorted Array
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0

        slow = 1
        for fast in range(1, len(nums)):
            if nums[fast] != nums[fast - 1]:
                nums[slow] = nums[fast]
                slow += 1

        return slow

    # 80 Remove Duplicates from Sorted Array II
    def removeDuplicates2(self, nums: List[int]) -> int:
        if len(nums) <= 2:
            return len(nums)

        slow = 2  # 前两个肯定保留
        for fast in range(2, len(nums)):
            if nums[fast] != nums[slow - 2]:
                nums[slow] = nums[fast]
                slow += 1

        return slow

    # 922 Sort Array By Parity II
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        even, odd = 0, 1
        n = len(nums)

        while even < n and odd < n:
            # 找到错误的 even 索引
            while even < n and nums[even] % 2 == 0:
                even += 2
            # 找到错误的 odd 索引
            while odd < n and nums[odd] % 2 == 1:
                odd += 2

            # 交换错误位置的奇偶值
            if even < n and odd < n:
                nums[even], nums[odd] = nums[odd], nums[even]
                even += 2
                odd += 2

        return nums

    """
        2.双序列双指针
    """

    # 2109 Adding Spaces to a String
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        p1, p2 = 0, 0
        ans: list[str] = []

        while p1 < len(s):
            if p2 < len(spaces) and p1 == spaces[p2]:
                ans.append(" ")
                p2 += 1
            ans.append(s[p1])
            p1 += 1

        return ''.join(ans)

    # 1855 Maximum Distance Between a Pair of Values
    def maxDistance(self, nums1: List[int], nums2: List[int]) -> int:
        p1, p2 = 0, 0
        ans: int = 0
        while p1 < len(nums1) and p2 < len(nums2):
            if nums1[p1] <= nums2[p2]:
                p2 += 1
                ans = max(ans, p2 - p1)
            elif nums1[p1] > nums2[p2]:
                p1 += 1

        return ans

    # 2337 Move Pieces to Obtain a String
    def canChange(self, start: str, target: str) -> bool:
        n = len(start)
        p1 = p2 = 0

        while p1 < n or p2 < n:
            # 跳过空格
            while p1 < n and start[p1] == '_':
                p1 += 1
            while p2 < n and target[p2] == '_':
                p2 += 1

            # 都走到头了，说明匹配成功
            if p1 == n and p2 == n:
                return True
            # 一个走到头另一个没走到，匹配失败
            if p1 == n or p2 == n:
                return False

            if start[p1] != target[p2]:
                return False
            if start[p1] == 'L' and p1 < p2:
                return False  # 'L' 只能左移
            if start[p1] == 'R' and p1 > p2:
                return False  # 'R' 只能右移
            p1 += 1
            p2 += 1
        return True

    # 844. Backspace String Compare
    def backspaceCompare(self, s: str, t: str) -> bool:
        p1, p2 = len(s) - 1, len(t) - 1
        b1, b2 = 0, 0
        while p1 >= 0 and p2 >= 0:
            # 获取
            while p1 >= 0 and s[p1] == '#':
                b1 += 1
                p1 -= 1
            while p2 >= 0 and s[p2] == '#':
                b2 += 1
                p2 -= 1
            # 消耗'#'
            while p1 >= 0 and b1 > 0:
                p1 -= 1
                b1 -= 1
            while p2 >= 0 and b2 > 0:
                p2 -= 1
                b2 -= 1
            if s[p1] != s[p2]:
                return False

        return True

    """ 判断子序列 """

    # 392. Is Subsequence
    def isSubsequence(self, s: str, t: str) -> bool:
        p1, p2 = 0, 0
        while p1 < len(s) or p2 < len(t):
            if s[p1] == t[p2]:
                p1 += 1
            p2 += 1

        return p1 == len(s)


if __name__ == '__main__':
    slu = TwoPointers()
    # slu.minimumLength("aabccabba")
    # ans = slu.threeSumMulti([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], 8)
    # nums = [1, 1, 1, 2, 2, 3]
    # ans = slu.removeElement(nums, 2)
    # ans = slu.removeDuplicates2(nums)
    # success = slu.canChange("_L__R__R_L", "L______RR_")
    slu.backspaceCompare("a#c", "b")
    pass
