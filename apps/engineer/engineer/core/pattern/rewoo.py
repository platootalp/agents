"""
ReWOO Agent
- Plan 格式: Plan: ... \n#E1 = Tool[参数]
- 占位符: #E1, #E2
- 支持 LLM[...] 内部推理
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from core import QwenLLM
from pattern.tools import ToolExecutor
# genAI_main_start
from pattern.base_agent import BaseAgent
# genAI_main_end

# =========================
# Planner Prompt（动态工具 + 提取示例）
# =========================
PLANNER_PROMPT_BASE = """
请为以下任务制定一个逐步解决问题的计划。对于每一步计划，请明确指出应调用哪个外部工具及输入参数，以获取证据。  
你可以将证据存入变量 `#E` 中，供后续步骤引用（格式如：Plan, #E1, Plan, #E2, ...）。

可用工具如下：
{tool_descriptions}

【示例：数学计算】
任务：一个长方体水箱长 2 米、宽 1.5 米、高 1 米。若每分钟注水 0.3 立方米，注满需要多少分钟？
Plan: 计算水箱的总体积（立方米）。
#E1 = LLM[计算 2 * 1.5 * 1]
Plan: 用总体积除以注水速率，得到所需时间（分钟）。
#E2 = LLM[计算 #E1 / 0.3]

开始！请为以下任务制定详细计划。每个 Plan 后只能跟一个 #E 变量。

任务：{task}
"""


# =========================
# Planner
# =========================
class ReWOOPlanner:
    def __init__(self, agent: 'ReWOOAgent'):
        self.agent = agent

    def generate_plan(self, task: str) -> List[Tuple[str, str, str, str]]:
        tool_desc = self.agent.get_tool_descriptions()
        prompt = PLANNER_PROMPT_BASE.format(
            tool_descriptions=tool_desc,
            task=task
        )
        response = self.agent.call_llm(prompt)

        # 匹配 Plan 和 #E（兼容 [...] 或 (...)）
        pattern = r"Plan:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*[\[\(]([^\]\)]*)[\]\)]"
        matches = re.findall(pattern, response, re.DOTALL)

        steps = []
        for plan_desc, var_name, tool_name, arg in matches:
            # 去除参数首尾引号
            arg = arg.strip()
            if arg.startswith('"') and arg.endswith('"'):
                arg = arg[1:-1]
            elif arg.startswith("'") and arg.endswith("'"):
                arg = arg[1:-1]
            steps.append((plan_desc.strip(), var_name, tool_name, arg))
        return steps


# =========================
# Worker（带 LLM 提取后处理）
# =========================
class ReWOOWorker:
    def __init__(self, agent: 'ReWOOAgent'):
        self.agent = agent

    def execute_plan(self, steps: List[Tuple[str, str, str, str]]) -> Dict[str, str]:
        evidence: Dict[str, str] = {}
        for plan_desc, var_name, tool_name, raw_arg in steps:
            resolved_arg = self._resolve_placeholders(raw_arg, evidence)

            if tool_name == "LLM":
                evidence[var_name] = self.agent.call_llm(resolved_arg)
            else:
                evidence[var_name] = self.agent.execute_tool(tool_name, resolved_arg)
        return evidence

    def _resolve_placeholders(self, text: str, evidence: Dict[str, str]) -> str:
        def replacer(match):
            var = match.group(0)  # #E1
            return evidence.get(var, var)

        return re.sub(r"#E\d+", replacer, text)


# =========================
# Solver
# =========================
SOLVER_PROMPT_TEMPLATE = """
    请根据以下任务、分步计划及每步的执行结果，直接给出最终答案。
    
    {full_plan_with_evidence}
    
    请基于上述证据回答问题，**不要包含任何解释、前缀或多余文字**，仅输出最终答案。
    
    任务：{task}
    回答：
"""


class ReWOOSolver:
    def __init__(self, agent: 'ReWOOAgent'):
        self.agent = agent

    def solve(self, task: str, steps: List[Tuple], evidence: Dict[str, str]) -> str:
        plan_lines = []
        for plan_desc, var_name, tool_name, raw_arg in steps:
            resolved_arg = self._resolve_placeholders(raw_arg, evidence)
            plan_lines.append(f"Plan: {plan_desc}")
            plan_lines.append(f"{var_name} = {tool_name}[{resolved_arg}] → {evidence.get(var_name, '')}")
        full_plan = "\n".join(plan_lines)

        prompt = SOLVER_PROMPT_TEMPLATE.format(
            full_plan_with_evidence=full_plan,
            task=task
        )
        return self.agent.call_llm(prompt)

    def _resolve_placeholders(self, text: str, evidence: Dict[str, str]) -> str:
        def replacer(match):
            var = match.group(0)
            return evidence.get(var, var)

        return re.sub(r"#E\d+", replacer, text)


# =========================
# ReWOO Agent
# =========================
# genAI_main_start
class ReWOOAgent(BaseAgent):
    def __init__(self, llm_client: QwenLLM, tool_executor: ToolExecutor):
        super().__init__(llm_client, tool_executor)
        self.planner = ReWOOPlanner(self)
        self.worker = ReWOOWorker(self)
        self.solver = ReWOOSolver(self)
# genAI_main_end

    def run(self, task: str) -> str:
        print("=== Step 1: Planner ===")
        steps = self.planner.generate_plan(task)
        if not steps:
            raise ValueError("Planner 未生成有效计划，请重试或调整任务。")
        for desc, var, tool, arg in steps:
            print(f"Plan: {desc}")
            print(f"{var} = {tool}[{arg}]")

        print("\n=== Step 2: Worker ===")
        evidence = self.worker.execute_plan(steps)
        for k, v in evidence.items():
            print(f"{k}: {v}")

        print("\n=== Step 3: Solver ===")
        final_answer = self.solver.solve(task, steps, evidence)
        self._print_final_answer(final_answer)
        return final_answer


# =========================
# Main
# =========================
if __name__ == "__main__":
    load_dotenv()
    llm = QwenLLM(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_API_URL")
    )
    tool_executor = ToolExecutor()

    print("=" * 60)
    print("ReWOO Agent")
    print("=" * 60)

    agent = ReWOOAgent(llm, tool_executor)
    agent.run("今天距离2026年春节还有多少天？")
