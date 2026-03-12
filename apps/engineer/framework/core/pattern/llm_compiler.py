"""
LLMCompiler - 通过 DAG 实现工具调用的并行编排
核心组件：
1. Function Calling Planner - 生成 DAG 执行计划
2. Task Fetching Unit - 并行调度就绪任务
3. Joining Unit - 结果聚合
"""
from __future__ import annotations

import os
import asyncio
import re
import json
from dotenv import load_dotenv
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from core import QwenLLM
from pattern.tools import ToolExecutor
# genAI_main_start
from pattern.base_agent import BaseAgent
# genAI_main_end


@dataclass
class TaskNode:
    """DAG 节点，表示一个工具调用任务"""
    task_id: str
    tool_name: str
    arguments: str
    dependencies: Set[str] = field(default_factory=set)
    result: Optional[str] = None
    status: str = "pending"


@dataclass
class DAG:
    """有向无环图，表示执行计划"""
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    edges: Dict[str, Set[str]] = field(default_factory=dict)

    def add_node(self, node: TaskNode):
        self.nodes[node.task_id] = node
        self.edges[node.task_id] = set()

    def add_edge(self, from_id: str, to_id: str):
        if from_id in self.edges and to_id in self.edges:
            self.edges[from_id].add(to_id)

    def get_ready_tasks(self) -> List[TaskNode]:
        """获取所有依赖已满足的待执行任务"""
        ready_tasks = []
        for task_id, node in self.nodes.items():
            if node.status == "pending":
                deps = node.dependencies
                if all(self.nodes[dep].result is not None for dep in deps):
                    ready_tasks.append(node)
        return ready_tasks

    def is_complete(self) -> bool:
        return all(node.status == "completed" for node in self.nodes.values())


class FunctionCallingPlanner:
    """Function Calling Planner - 生成 DAG 执行计划"""

    PLANNER_PROMPT = """
        请为以下任务制定一个执行计划，使用有向无环图（DAG）的形式表示。
        每个节点代表一个工具调用，边表示数据依赖关系。
        
        可用工具：
        {tool_descriptions}
        
        输出格式要求（JSON）：
        {{
          "tasks": [
            {{
              "task_id": "T1",
              "tool_name": "tool_name",
              "arguments": "参数内容",
              "dependencies": []
            }},
            {{
              "task_id": "T2",
              "tool_name": "tool_name",
              "arguments": "参数内容（可引用前置任务结果，如 #T1）",
              "dependencies": ["T1"]
            }}
          ]
        }}
        
        重要规则：
        1. task_id 必须唯一，格式为 T1, T2, T3...
        2. dependencies 列出该任务依赖的前置任务 ID
        3. arguments 中可以用 #T1, #T2 引用前置任务的结果
        4. 如果需要 LLM 进行推理，使用 "LLM" 作为 tool_name
        5. 确保生成的图是无环的（没有循环依赖）
        6. 对于需要多个参数的工具，使用 JSON 格式：{{"param1": "value1", "param2": "value2"}}
        
        示例：
        任务：一个长方体水箱长 2 米、宽 1.5 米、高 1 米。若每分钟注水 0.3 立方米，注满需要多少分钟？
        {{
          "tasks": [
            {{
              "task_id": "T1",
              "tool_name": "LLM",
              "arguments": "计算 2 * 1.5 * 1",
              "dependencies": []
            }},
            {{
              "task_id": "T2",
              "tool_name": "LLM",
              "arguments": "用 #T1 除以 0.3",
              "dependencies": ["T1"]
            }}
          ]
        }}
                
        任务：{task}
        
        请严格按照上述 JSON 格式输出，不要包含其他内容。
    """

    def __init__(self, agent: 'LLMCompilerAgent'):
        self.agent = agent

    def generate_plan(self, task: str) -> DAG:
        """生成 DAG 执行计划"""
        tool_desc = self.agent.get_tool_descriptions()
        prompt = self.PLANNER_PROMPT.format(
            tool_descriptions=tool_desc,
            task=task
        )

        response = self.agent.call_llm(prompt)

        try:
            plan_data = self._parse_response(response)
            return self._build_dag(plan_data)
        except Exception as e:
            raise Exception(f"生成计划失败: {e}\nLLM 响应: {response}")

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON"""
        json_match = re.search(r'\{[\s\S]*}', response)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        raise ValueError("无法从响应中提取有效的 JSON")

    def _build_dag(self, plan_data: Dict[str, Any]) -> DAG:
        """构建 DAG"""
        dag = DAG()

        for task_info in plan_data.get("tasks", []):
            task_id = task_info["task_id"]
            tool_name = task_info["tool_name"]
            arguments = task_info["arguments"]
            dependencies = set(task_info.get("dependencies", []))

            node = TaskNode(
                task_id=task_id,
                tool_name=tool_name,
                arguments=arguments,
                dependencies=dependencies
            )
            dag.add_node(node)

        for task_id, node in dag.nodes.items():
            for dep_id in node.dependencies:
                dag.add_edge(dep_id, task_id)

        return dag


class TaskFetchingUnit:
    """Task Fetching Unit - 并行调度就绪任务"""

    def __init__(self, agent: 'LLMCompilerAgent'):
        self.agent = agent

    async def execute_task(self, node: TaskNode, dag: DAG) -> str:
        """执行单个任务"""
        node.status = "running"

        resolved_args = self._resolve_arguments(node.arguments, dag)

        try:
            if node.tool_name == "LLM":
                result = self.agent.call_llm(resolved_args)
            else:
                result = self.agent.execute_tool(node.tool_name, resolved_args)

            node.result = result
            node.status = "completed"
            return result
        except Exception as e:
            node.status = "failed"
            node.result = f"Error: {str(e)}"
            raise

    async def execute_ready_tasks(self, dag: DAG) -> List[Tuple[str, str]]:
        """并行执行所有就绪任务"""
        ready_tasks = dag.get_ready_tasks()
        if not ready_tasks:
            return []

        tasks = [self.execute_task(node, dag) for node in ready_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        completed = []
        for node, result in zip(ready_tasks, results):
            if isinstance(result, Exception):
                completed.append((node.task_id, str(result)))
            else:
                completed.append((node.task_id, result))

        return completed

    def _resolve_arguments(self, arguments: str, dag: DAG) -> str:
        """解析参数中的依赖引用（如 #T1）"""
        resolved = arguments

        if resolved.strip().startswith("{") and resolved.strip().endswith("}"):
            try:
                json_str = resolved.strip()
                if "'" in json_str and '"' not in json_str:
                    json_str = json_str.replace("'", '"')
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    for key, value in parsed.items():
                        if isinstance(value, str) and value.startswith("#T"):
                            task_id = value[1:]
                            if task_id in dag.nodes and dag.nodes[task_id].result is not None:
                                parsed[key] = dag.nodes[task_id].result
                    resolved = json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass

        def replacer(match):
            ref = match.group(0)
            task_id = ref[1:]
            if task_id in dag.nodes and dag.nodes[task_id].result is not None:
                return dag.nodes[task_id].result
            return ref

        resolved = re.sub(r'#T\d+', replacer, resolved)

        return resolved


class JoiningUnit:
    """Joining Unit - 结果聚合"""

    SOLVER_PROMPT = """
        请根据以下任务和执行结果，给出最终答案。
        
        任务：{task}
        
        执行过程：
        {execution_summary}
        
        请基于上述执行结果回答问题，仅输出最终答案。
    """

    def __init__(self, agent: 'LLMCompilerAgent'):
        self.agent = agent

    def generate_final_answer(self, task: str, dag: DAG) -> str:
        """生成最终答案"""
        execution_summary = self._build_execution_summary(dag)

        prompt = self.SOLVER_PROMPT.format(
            task=task,
            execution_summary=execution_summary
        )

        return self.agent.call_llm(prompt)

    def _build_execution_summary(self, dag: DAG) -> str:
        """构建执行摘要"""
        lines = []
        for task_id in sorted(dag.nodes.keys()):
            node = dag.nodes[task_id]
            deps_str = ", ".join(node.dependencies) if node.dependencies else "无"
            lines.append(f"{task_id}: {node.tool_name}[{node.arguments}]")
            lines.append(f"  依赖: {deps_str}")
            lines.append(f"  结果: {node.result}")
            lines.append("")
        return "\n".join(lines)


# genAI_main_start
class LLMCompilerAgent(BaseAgent):
    """LLMCompilerAgent 主类 - 协调三个核心组件"""

    def __init__(self, llm_client: QwenLLM, tool_executor: ToolExecutor):
        super().__init__(llm_client, tool_executor)
        self.planner = FunctionCallingPlanner(self)
        self.task_fetcher = TaskFetchingUnit(self)
        self.joiner = JoiningUnit(self)
# genAI_main_end

    def run(self, task: str) -> str:
        """执行 LLMCompiler 流程"""
        print("=== Step 1: Function Calling Planner (生成 DAG) ===")
        dag = self.planner.generate_plan(task)
        self._print_dag(dag)

        print("\n=== Step 2: Task Fetching Unit (并行执行) ===")
        asyncio.run(self._execute_dag(dag))
        self._print_execution_results(dag)

        print("\n=== Step 3: Joining Unit (结果聚合) ===")
        final_answer = self.joiner.generate_final_answer(task, dag)
        self._print_final_answer(final_answer)

        return final_answer

    async def _execute_dag(self, dag: DAG):
        """执行 DAG 直到完成"""
        iteration = 1
        while not dag.is_complete():
            print(f"\n[迭代 {iteration}] 执行就绪任务...")
            completed = await self.task_fetcher.execute_ready_tasks(dag)

            if not completed:
                print("没有就绪任务，可能存在循环依赖或错误")
                break

            for task_id, result in completed:
                print(f"  {task_id} 完成: {result[:100]}...")

            iteration += 1

            if iteration > 100:
                print("执行超过 100 次迭代，停止")
                break

    def _print_dag(self, dag: DAG):
        """打印 DAG 结构"""
        print("执行计划 DAG:")
        for task_id in sorted(dag.nodes.keys()):
            node = dag.nodes[task_id]
            deps_str = ", ".join(node.dependencies) if node.dependencies else "无"
            print(f"  {task_id}: {node.tool_name}[{node.arguments}]")
            print(f"      依赖: {deps_str}")

    def _print_execution_results(self, dag: DAG):
        """打印执行结果"""
        print("\n执行结果:")
        for task_id in sorted(dag.nodes.keys()):
            node = dag.nodes[task_id]
            status_icon = "✓" if node.status == "completed" else "✗"
            print(f"  {status_icon} {task_id}: {node.result}")


if __name__ == "__main__":
    load_dotenv()
    llm = QwenLLM(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_API_URL")
    )
    tool_executor = ToolExecutor()

    print("=" * 60)
    print("LLMCompiler Agent")
    print("=" * 60)

    compiler = LLMCompilerAgent(llm, tool_executor)
    compiler.run("今天距离2026年春节还有多少天？")
