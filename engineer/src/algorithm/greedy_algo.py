from collections import defaultdict
from typing import List


class GreedyAlgo:
    # 781 Rabbits in Forest
    def numRabbits(self, answers: List[int]) -> int:
        count = defaultdict(int)
        ans = 0
        for num in answers:
            count[num] += 1

        for key, val in count.items():
            group_size = key + 1
            # 计算需要多少组，向上取整
            ans += (val + key) // group_size * group_size

        return ans


if __name__ == '__main__':
    pass
