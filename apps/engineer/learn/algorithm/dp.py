from typing import List


class Solution:

    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        dp1 = 1
        dp2 = 2
        ans = 0
        for num in range(2, n + 1):
            ans = dp1 + dp2
            dp1 = dp2
            dp2 = ans
        return ans

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0] * 3
        for i in range(2, len(cost) + 1):
            dp[2] = min(dp[0] + cost[i - 2], dp[1] + cost[i - 1])
            dp[0] = dp[1]
            dp[1] = dp[2]
        return  dp[2]
