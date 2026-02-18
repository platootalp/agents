"""
ReAct Agent 示例实现
- 支持真实 Tavily 搜索
- 模块化设计，易于扩展
- 结构化历史记录
"""
import os
import re
from typing import Dict, Callable, List, Optional, Tuple
from dotenv import load_dotenv
from core import QwenLLM
from pattern.tools import ToolExecutor
# genAI_main_start
from pattern.base_agent import BaseAgent
# genAI_main_end


# =========================
# ReAct Agent
# =========================
# genAI_main_start
class ReActAgent(BaseAgent):
    def __init__(
            self,
            llm: QwenLLM,
            tool_executor: ToolExecutor,
            max_steps: int = 5
    ):
        super().__init__(llm, tool_executor)
        self.max_steps = max_steps

    def run(self, question: str) -> str:
# genAI_main_end
        history: List[Dict[str, str]] = []  # 结构化历史：[{"thought": ..., "action": ..., "observation": ...}]

        for step in range(1, self.max_steps + 1):
            # 构造历史字符串
            history_str = "\n".join(
                f"Thought: {h['thought']}\nAction: {h['action']}\nObservation: {h['observation']}"
                for h in history
            )

            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=self.get_tool_descriptions(),
                question=question,
                history=history_str
            )

            self._print_step(step, self.max_steps, f"Step {step}")
            response = self.call_llm(prompt, strip=False)
            if not response:
                print("LLM returned empty response.")
                break

            thought, action = self._parse_llm_output(response)
            if not action:
                print("Failed to parse LLM output.")
                break

            print(f"Thought: {thought}")
            print(f"Action: {action}")

            # 终止条件
            if action.startswith("Finish["):
                answer = re.match(r"Finish\[(.*)]", action, re.DOTALL)
                return answer.group(1).strip() if answer else "No final answer."

            # 执行工具
            tool_name, tool_input = self._parse_action(action)
            if not tool_name:
                observation = f"Invalid action format: {action}"
            else:
                observation = self.execute_tool(tool_name, tool_input)
            print(f"Observation: {observation}")

            history.append({
                "thought": thought,
                "action": action,
                "observation": observation
            })

        print("Maximum steps reached.")
        # genAI_main_start
        return "无法在最大步数内完成任务"
        # genAI_main_end

    def _parse_llm_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        thought_match = re.search(r"Thought:\s*(.*?)(?=\n)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.*?)(?=\n|$)", text, re.DOTALL)
        thought = (thought_match.group(1).strip() if thought_match else None) or ""
        action = (action_match.group(1).strip() if action_match else None) or ""
        return thought, action

    def _parse_action(self, action_str: str) -> Tuple[Optional[str], str]:
        match = re.match(r"(\w+)\[(.*)]", action_str, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, ""


# =========================
# 提示词模板
# =========================
REACT_PROMPT_TEMPLATE = """
你是一个能调用外部工具的智能助手。

可用工具:
{tools}

请严格按以下格式响应：
Thought: 分析当前情况，规划下一步。
Action: 执行以下之一：
  - ToolName[输入]
  - Finish[最终答案]

问题: {question}
历史记录:{history}
"""

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    load_dotenv()

    # 初始化 LLM
    llm = QwenLLM(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_API_URL")
    )

    # 初始化工具
    tool_executor = ToolExecutor()
    # 运行 Agent
    agent = ReActAgent(llm, tool_executor, max_steps=5)
    question = ("今天股票市场行情如何？")
    # question = ("今天距离2026年春节还有多少天？")
    print("=" * 60)
    print("ReAct Agent")
    print("=" * 60)

    print(f"\nQuestion: {question}")
    answer = agent.run(question)

    print("\n" + "=" * 50)
    print(f"Final Answer: {answer}")
