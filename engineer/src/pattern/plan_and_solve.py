"""
Plan-Solve Agent 示例实现
- Planner: 生成计划
- ExecutionState: 封装执行状态
- Executor: 执行步骤（无状态）
- PlanSolveAgent: 协调流程
"""

import os
import re
import ast
from typing import List, Dict, Optional
from dotenv import load_dotenv
from core import QwenLLM
# genAI_main_start
from pattern.base_agent import BaseAgent
# genAI_main_end

# =========================
# 提示模板
# =========================
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。请将用户提出的复杂问题分解为一个清晰、有序、可执行的行动计划。
要求：
1. 每个步骤必须是独立、具体的子任务；
2. 步骤之间应有明确的逻辑顺序；
3. 不要包含解释、编号前缀或多余文本。

问题: {question}

请严格按以下格式输出（必须包含 ```python 和 ```）：
```python
["子任务1描述", "子任务2描述", "子任务3描述", ...]
```
"""

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。请严格按计划执行当前步骤。

# 原始问题:
{question}

# 完整计划:
{plan}

# 已完成的步骤与结果:
{history}

# 当前要执行的步骤:
{current_step}

注意：你可以自由使用"已完成的步骤与结果"中的信息。请仅输出当前步骤的最终答案，不要任何解释、前缀或后缀。
"""

FINAL_ANSWER_PROMPT_TEMPLATE = """
你是一个专业的AI助手。请根据以下任务执行过程，生成一个简洁、完整、面向用户的最终回答。

# 原始问题:
{question}

# 执行过程摘要:
{execution_summary}

请直接输出最终答案，语言自然流畅，不要提及"根据以上内容"或重复步骤编号。
"""


# =========================
# 1. ExecutionState: 状态管理
# =========================
class ExecutionState:
    def __init__(self, plan: List[str]):
        if not plan:
            raise ValueError("计划不能为空")
        self.plan = plan
        self._history: List[Dict[str, str]] = []  # 每项: {"step": str, "result": str}

    @property
    def history(self) -> List[Dict[str, str]]:
        return self._history.copy()  # 避免外部修改

    def is_completed(self) -> bool:
        return len(self._history) >= len(self.plan)

    def get_current_step(self) -> Optional[str]:
        if self.is_completed():
            return None
        return self.plan[len(self._history)]

    def get_step_index(self) -> int:
        return len(self._history)

    def add_result(self, result: str):
        if self.is_completed():
            raise RuntimeError("计划已执行完毕，无法添加新结果")
        step = self.plan[len(self._history)]
        self._history.append({"step": step, "result": result})

    def get_history_str(self) -> str:
        if not self._history:
            return "无"
        return "\n".join(
            f"步骤: {h['step']}\n结果: {h['result']}" for h in self._history
        )

    def get_execution_summary(self) -> str:
        return "\n".join(f"- {h['step']}: {h['result']}" for h in self._history)


# =========================
# 2. Planner: 规划器
# =========================
class Planner:
    def __init__(self, agent: 'PlanSolveAgent'):
        self.agent = agent

    def generate_plan(self, question: str) -> List[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        response = self.agent.call_llm(prompt, strip=False)
        plan = self._parse_plan(response)
        if not plan:
            raise ValueError("规划失败：未能生成有效计划。")
        return plan

    def _parse_plan(self, text: str) -> List[str]:
        match = re.search(r"```(?:python)?\s*(\[.*?])\s*```", text, re.DOTALL)
        if not match:
            match = re.search(r"(\[.*?])", text, re.DOTALL)
        if match:
            try:
                parsed = ast.literal_eval(match.group(1))
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except (SyntaxError, ValueError):
                pass
        return []


# =========================
# 3. Executor: 执行器
# =========================
class Executor:
    def __init__(self, agent: 'PlanSolveAgent'):
        self.agent = agent

    def execute_step(self, question: str, state: ExecutionState) -> str:
        current_step = state.get_current_step()
        if current_step is None:
            raise RuntimeError("无法执行：计划已完成")

        prompt = EXECUTOR_PROMPT_TEMPLATE.format(
            question=question,
            plan=str(state.plan),
            history=state.get_history_str(),
            current_step=current_step
        )

        result = self.agent.call_llm(prompt)
        return result if result else "[空结果]"

    def generate_final_answer(self, question: str, state: ExecutionState) -> str:
        prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
            question=question,
            execution_summary=state.get_execution_summary()
        )
        return self.agent.call_llm(prompt)


# =========================
# 4. PlanSolveAgent: 协调器
# =========================
# genAI_main_start
class PlanSolveAgent(BaseAgent):
    def __init__(self, llm_client: QwenLLM):
        super().__init__(llm_client)
        self.planner = Planner(self)
        self.executor = Executor(self)
# genAI_main_end

    def run(self, question: str) -> str:
        # 1. 规划
        plan = self.planner.generate_plan(question)
        self._print_plan(plan)

        # 2. 执行
        state = ExecutionState(plan)
        while not state.is_completed():
            result = self.executor.execute_step(question, state)
            self._print_step(state.get_step_index() + 1, len(plan), state.get_current_step(), result)
            state.add_result(result)

        # 3. 合成最终答案
        final_answer = self.executor.generate_final_answer(question, state)
        self._print_final_answer(final_answer)
        return final_answer

    def _print_plan(self, plan: List[str]):
        self._print_stage("生成的计划", "\n".join(f"{i}. {step}" for i, step in enumerate(plan, 1)))


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
    print("Plan-Solve Agent")
    print("=" * 60)

    agent = PlanSolveAgent(llm_client=llm)
    try:
        agent.run("""
           你正在把积木堆成金字塔。每个块都有一个颜色，用一个字母表示。每一行的块比它下面的行 少一个块 ，并且居中。
           为了使金字塔美观，只有特定的 三角形图案 是允许的。一个三角形的图案由 两个块 和叠在上面的 单个块 组成。模式是以三个字母字符串的列表形式 allowed 给出的，其中模式的前两个字符分别表示左右底部块，第三个字符表示顶部块。
           例如，"ABC" 表示一个三角形图案，其中一个 "C" 块堆叠在一个 'A' 块(左)和一个 'B' 块(右)之上。请注意，这与 "BAC" 不同，"B" 在左下角，"A" 在右下角。
           你从作为单个字符串给出的底部的一排积木 bottom 开始，必须 将其作为金字塔的底部。
           在给定 bottom 和 allowed 的情况下，如果你能一直构建到金字塔顶部，使金字塔中的 每个三角形图案 都是在 allowed 中的，则返回 true ，否则返回 false 。
           请用Java代码实现class Solution {
               public boolean pyramidTransition(String bottom, List<String> allowed) {

               }
           }
        """)
    except Exception as e:
        print(f"执行出错: {e}")
