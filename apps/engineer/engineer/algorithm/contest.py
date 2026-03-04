from typing import List


class Solution:
    # 第448场周赛
    # 3536.两个数字的最大乘积
    def maxProduct(n: int) -> int:
        digits = [int(c) for c in str(n)]
        digits.sort(reverse=True)
        return digits[0] * digits[1]

    # 3537.填充特殊网格
    def specialGrid(self, n: int) -> List[List[int]]:
        def build(n: int, base: int) -> List[List[int]]:
            if n == 0:
                return [[base]]

            quadrant_size = 4 ** (n - 1)  # 每个子象限的格子数

            # 递归构造四个象限
            A = build(n - 1, base + 3 * quadrant_size)  # 左上最大
            B = build(n - 1, base)  # 右上最小
            C = build(n - 1, base + quadrant_size)  # 右下第二小
            D = build(n - 1, base + 2 * quadrant_size)  # 左下第三小

            # 组合成一个 2^n x 2^n 的矩阵
            top = [a_row + b_row for a_row, b_row in zip(A, B)]
            bottom = [d_row + c_row for d_row, c_row in zip(D, C)]
            return top + bottom

        return build(n, 0)

if __name__ == '__main__':
    pass
