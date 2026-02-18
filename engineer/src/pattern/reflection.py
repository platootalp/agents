"""
Reflection Agent 示例实现（职责分离版 + 提示词适配）
- Generator: 负责生成初始答案和精炼答案
- Reflector: 负责评估答案并提供改进建议
- ReflectionAgent: 协调反思循环
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from core import QwenLLM
# genAI_main_start
from pattern.base_agent import BaseAgent
# genAI_main_end

INITIAL_PROMPT = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {question}

请直接输出代码，不要包含任何额外的解释。
"""

REFINE_PROMPT = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{question}

# 你上一轮尝试的代码:
{answer}

评审员的反馈：
{reflection}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""
# =========================
# 1. Generator: 生成器
# =========================
class Generator:
    def __init__(self, agent: 'ReflectionAgent'):
        self.agent = agent

    def generate_initial(self, question: str) -> str:
        prompt = INITIAL_PROMPT.format(question=question)
        return self.agent.call_llm(prompt)

    def refine(self, question: str, current_answer: str, reflection: str) -> str:
        prompt = REFINE_PROMPT.format(
            question=question,
            answer=current_answer,
            reflection=reflection
        )
        return self.agent.call_llm(prompt)


REFLECTION_PROMPT = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

# 原始任务:
{question}

# 待审查的代码:
```python
{answer}
```

请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议。
如果代码在算法层面已经达到最优，回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""


# =========================
# 2. Reflector: 反思器
# =========================
class Reflector:
    def __init__(self, agent: 'ReflectionAgent'):
        self.agent = agent

    def reflect(self, question: str, answer: str) -> str:
        prompt = REFLECTION_PROMPT.format(
            question=question,
            answer=answer
        )
        return self.agent.call_llm(prompt)


# =========================
# 3. ReflectionAgent
# =========================
# genAI_main_start
class ReflectionAgent(BaseAgent):
    def __init__(self, llm_client: QwenLLM, max_reflections: int = 3):
        super().__init__(llm_client)
        self.max_reflections = max_reflections
        self.generator = Generator(self)
        self.reflector = Reflector(self)
# genAI_main_end

    def _should_continue_reflection(self, reflection: str) -> bool:
        """
        判断是否需要继续反思。
        如果反思中包含“无需改进”、“已最优”、“无需修改”等关键词，则终止。
        """
        reflection_lower = reflection.lower()
        stop_keywords = ["无需改进", "无需修改", "已最优", "perfect", "no improvement", "no need to"]
        return not any(kw in reflection_lower for kw in stop_keywords)

    def run(self, question: str) -> str:
        # 1. 生成初始答案
        current_answer = self.generator.generate_initial(question)
        self._print_stage("初始答案", current_answer)

        # 2. 反思-精炼循环（动态终止）
        for i in range(self.max_reflections):
            reflection = self.reflector.reflect(question, current_answer)
            self._print_stage(f"反思 #{i + 1}", reflection)

            # 检查是否需要继续
            if not self._should_continue_reflection(reflection):
                print("\n--- 反思终止：当前答案已足够好 ---")
                break

            current_answer = self.generator.refine(question, current_answer, reflection)
            self._print_stage(f"改进答案 #{i + 1}", current_answer)

        self._print_final_answer(current_answer)
        return current_answer


# =========================
# 示例运行
# =========================
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_API_URL")

    if not api_key:
        raise ValueError("请在 .env 中配置 DASHSCOPE_API_KEY")

    llm = QwenLLM(api_key=api_key, base_url=base_url)

    print("=" * 60)
    print("Reflection Agent（代码生成场景）")
    print("=" * 60)

    agent = ReflectionAgent(llm_client=llm, max_reflections=3)
    agent.run("编写一个Python函数，找出1到n之间所有的素数 (prime numbers)")
