#!/usr/bin/env python3
"""
生产级Multi-Agent系统主程序
基于LangGraph框架，实现代码审查自动化流程
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import time

# 导入LangGraph相关
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from typing import Annotated
import operator

# 导入自定义模块
from message_queue import MessageQueue, MessageReliability
from capability_matrix import CapabilityMatrix, LoadBalancer
from coordinator import Coordinator, StateSynchronizer
from redis_store import RedisStateStore
# ================ 状态定义 ================


class CodeReviewState(TypedDict):
    """代码审查工作流状态"""

    # 输入
    code: str
    priority: int  # 任务优先级（1-5）
    task_id: str

    # 中间状态
    analysis_result: Optional[str]
    security_result: Optional[str]
    parallel_tasks: Annotated[List[str], operator.add]  # 并行任务列表

    # 最终输出
    final_report: Optional[str]
    execution_time: float  # 执行时间（毫秒）
    assigned_agent: str  # 分配的Agent标识
    workflow_status: str  # 状态：running/completed/failed


# ================ Agent定义 ================


class BaseAgent:
    """Agent基类"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        self.message_queue = MessageQueue()
        self.reliability = MessageReliability()

    async def process(self, state: CodeReviewState) -> Dict[str, Any]:
        """处理任务，子类实现"""
        raise NotImplementedError


class AnalyzerAgent(BaseAgent):
    """分析Agent：负责代码结构分析"""

    def __init__(self):
        super().__init__("analyzer")

    async def process(self, state: CodeReviewState) -> Dict[str, Any]:
        """分析代码结构"""
        print(f"[分析Agent] 开始分析任务 {state['task_id']}")

        code = state["coder"]

        # 构建分析提示
        prompt = f"""
        请分析以下代码：
        
        ```python
        {code}
        ```
        
        请提供：
        1. 代码结构分析
        2. 复杂度评估（圈复杂度、嵌套深度）
        3. 代码风格建议
        4. 潜在的改进点
        
        请用中文回复，结构清晰。
        """

        try:
            # 调用LLM
            response = self.llm.invoke(prompt)
            result = response.content

            # 模拟处理时间
            await asyncio.sleep(0.5)

            print(f"[分析Agent] 分析完成，结果长度: {len(result)} 字符")

            return {
                "analysis_result": result,
                "agent_id": self.agent_id,
                "timestamp": time.time(),
            }

        except Exception as e:
            print(f"[分析Agent] 分析失败: {e}")
            return {
                "analysis_result": f"分析失败: {str(e)}",
                "agent_id": self.agent_id,
                "error": True,
            }


class SecurityAgent(BaseAgent):
    """安全Agent：负责安全检查"""

    def __init__(self):
        super().__init__("security")

    async def process(self, state: CodeReviewState) -> Dict[str, Any]:
        """安全检查"""
        print(f"[安全Agent] 开始安全检查任务 {state['task_id']}")

        code = state["coder"]

        # 构建安全检查提示
        prompt = f"""
        请检查以下代码的安全问题：
        
        ```python
        {code}
        ```
        
        重点关注：
        1. SQL注入风险
        2. XSS攻击漏洞  
        3. 硬编码的密钥或密码
        4. 不安全的文件操作
        5. 命令注入风险
        
        请用中文回复，按风险等级分类。
        """

        try:
            # 调用LLM
            response = self.llm.invoke(prompt)
            result = response.content

            # 模拟处理时间
            await asyncio.sleep(0.7)

            print(f"[安全Agent] 安全检查完成，结果长度: {len(result)} 字符")

            return {
                "security_result": result,
                "agent_id": self.agent_id,
                "timestamp": time.time(),
            }

        except Exception as e:
            print(f"[安全Agent] 安全检查失败: {e}")
            return {
                "security_result": f"安全检查失败: {str(e)}",
                "agent_id": self.agent_id,
                "error": True,
            }


class ReporterAgent(BaseAgent):
    """报告Agent：负责生成综合报告"""

    def __init__(self):
        super().__init__("reporter")

    async def process(self, state: CodeReviewState) -> Dict[str, Any]:
        """生成报告"""
        print(f"[报告Agent] 开始生成报告任务 {state['task_id']}")

        analysis = state.get("analysis_result", "无分析结果")
        security = state.get("security_result", "无安全检查结果")
        code = state["coder"]

        # 构建报告提示
        prompt = f"""
        请基于以下信息生成代码审查报告：
        
        **代码内容：**
        ```python
        {code[:500]}...
        ```
        
        **代码分析结果：**
        {analysis}
        
        **安全检查结果：**
        {security}
        
        请生成一个专业的代码审查报告，包含：
        1. 执行摘要
        2. 代码质量评估
        3. 安全风险分析
        4. 改进建议
        5. 总体评分（1-5分）
        
        请用中文回复，使用Markdown格式。
        """

        try:
            # 调用LLM
            response = self.llm.invoke(prompt)
            result = response.content

            # 模拟处理时间
            await asyncio.sleep(0.3)

            print(f"[报告Agent] 报告生成完成，结果长度: {len(result)} 字符")

            return {
                "final_report": result,
                "agent_id": self.agent_id,
                "timestamp": time.time(),
            }

        except Exception as e:
            print(f"[报告Agent] 报告生成失败: {e}")
            return {
                "final_report": f"报告生成失败: {str(e)}",
                "agent_id": self.agent_id,
                "error": True,
            }


# ================ 工作流定义 ================


class MultiAgentWorkflow:
    """Multi-Agent工作流管理器"""

    def __init__(self):
        # 初始化Agent
        self.analyzer = AnalyzerAgent()
        self.security = SecurityAgent()
        self.reporter = ReporterAgent()

        # 初始化其他组件
        self.capability_matrix = CapabilityMatrix()
        self.load_balancer = LoadBalancer()
        self.redis_store = RedisStateStore()

        # 构建工作流
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流"""

        # 创建状态图
        workflow = StateGraph(CodeReviewState)

        # 添加节点
        workflow.add_node("task_dispatcher", self.task_dispatcher_agent)
        workflow.add_node("analyzer", self.analyzer_agent)
        workflow.add_node("security", self.security_agent)
        workflow.add_node("reporter", self.reporter_agent)
        workflow.add_node("parallel_executor", self.parallel_executor_agent)

        # 条件边定义
        def should_do_parallel(state):
            """判断是否需要并行执行"""
            return state.get("priority", 1) >= 3  # 高优先级任务并行

        # 边连接
        workflow.add_edge("task_dispatcher", "parallel_executor")

        workflow.add_conditional_edges(
            "parallel_executor",
            should_do_parallel,
            {
                "parallel": ["analyzer", "security"],  # 并行执行
                "sequential": "analyzer",  # 串行执行
            },
        )

        # 串行执行路径
        workflow.add_edge("analyzer", "security")
        workflow.add_edge("security", "reporter")

        # 并行执行路径（需要特殊处理）
        # 这里简化处理：并行执行后都转到报告Agent
        workflow.add_edge("analyzer", "reporter")
        workflow.add_edge("security", "reporter")

        workflow.add_edge("reporter", END)

        return workflow.compile()

    async def task_dispatcher_agent(self, state: CodeReviewState):
        """任务分配Agent"""
        print(f"[任务分配器] 处理任务 {state['task_id']}")

        # 选择最适合的Agent（简化实现）
        agents = ["analyzer", "security", "reporter"]
        selected = self.load_balancer.assign_task(
            {"id": state["task_id"], "coder": state["coder"]},
            agents,
            self.capability_matrix,
        )

        # 更新状态
        state["assigned_agent"] = selected

        # 保存状态到Redis
        await self.redis_store.save_workflow_state(state["task_id"], asdict(state))

        return state

    async def analyzer_agent(self, state: CodeReviewState):
        """分析Agent包装"""
        result = await self.analyzer.process(state)
        state["analysis_result"] = result.get("analysis_result")
        return state

    async def security_agent(self, state: CodeReviewState):
        """安全Agent包装"""
        result = await self.security.process(state)
        state["security_result"] = result.get("security_result")
        return state

    async def reporter_agent(self, state: CodeReviewState):
        """报告Agent包装"""
        result = await self.reporter.process(state)
        state["final_report"] = result.get("final_report")
        state["workflow_status"] = "completed"

        # 记录执行时间
        state["execution_time"] = (
            time.time() - float(state.get("start_time", time.time()))
        ) * 1000

        return state

    async def parallel_executor_agent(self, state: CodeReviewState):
        """并行执行协调器"""
        print(f"[并行执行器] 协调并行任务 {state['task_id']}")

        # 标记并行任务
        state["parallel_tasks"] = ["analysis", "security"]

        return state

    async def run(self, code: str, priority: int = 1) -> Dict[str, Any]:
        """运行工作流"""
        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 初始状态
        initial_state = {
            "coder": code,
            "priority": priority,
            "task_id": task_id,
            "analysis_result": None,
            "security_result": None,
            "parallel_tasks": [],
            "final_report": None,
            "execution_time": 0,
            "assigned_agent": "",
            "workflow_status": "running",
            "start_time": time.time(),
        }

        print(f"=== 开始执行代码审查任务 {task_id} ===")
        print(f"代码长度: {len(code)} 字符，优先级: {priority}")

        try:
            # 执行工作流
            start_time = time.time()
            result = await self.workflow.ainvoke(initial_state)
            end_time = time.time()

            execution_time = (end_time - start_time) * 1000

            print(f"=== 任务完成 ===")
            print(f"总执行时间: {execution_time:.2f} 毫秒")
            print(f"工作流状态: {result.get('workflow_status')}")

            # 返回结果
            return {
                "task_id": task_id,
                "success": True,
                "execution_time": execution_time,
                "final_report": result.get("final_report"),
                "analysis_result": result.get("analysis_result"),
                "security_result": result.get("security_result"),
                "assigned_agent": result.get("assigned_agent"),
                "workflow_status": result.get("workflow_status"),
            }

        except Exception as e:
            print(f"工作流执行失败: {e}")
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "workflow_status": "failed",
            }


# ================ 测试用例 ================


async def test_basic_workflow():
    """测试基本工作流"""
    print("\n" + "=" * 60)
    print("测试1: 基本工作流")
    print("=" * 60)

    # 创建工作流
    workflow = MultiAgentWorkflow()

    # 测试代码
    test_code = """
def process_user_input(user_input: str):
    import sqlite3
    conn = sqlite3.connect('database.db')
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    result = conn.execute(query)
    return result.fetchall()
    
def safe_process(user_input: str):
    import sqlite3
    conn = sqlite3.connect('database.db')
    query = "SELECT * FROM users WHERE name = ?"
    result = conn.execute(query, (user_input,))
    return result.fetchall()
    """

    # 运行工作流
    result = await workflow.run(test_code, priority=2)

    print(f"\n任务结果:")
    print(f"- 成功: {result['success']}")
    print(f"- 执行时间: {result['execution_time']:.2f}ms")

    if result["success"]:
        print(f"\n生成的报告摘要:")
        report = result.get("final_report", "")
        print(report[:500] + "..." if len(report) > 500 else report)

    return result


async def test_parallel_workflow():
    """测试并行工作流"""
    print("\n" + "=" * 60)
    print("测试2: 高优先级并行工作流")
    print("=" * 60)

    workflow = MultiAgentWorkflow()

    test_code = """
import os

def read_config():
    # 硬编码密码
    password = "admin123"
    
    # 直接拼接路径
    config_path = "/etc/config/" + os.getenv("ENV", "dev") + ".json"
    
    with open(config_path, 'r') as f:
        return json.load(f)
    """

    result = await workflow.run(test_code, priority=4)  # 高优先级触发并行

    print(f"\n并行任务结果:")
    print(f"- 成功: {result['success']}")
    print(f"- 执行时间: {result['execution_time']:.2f}ms")

    return result


async def test_load_balancing():
    """测试负载均衡"""
    print("\n" + "=" * 60)
    print("测试3: 负载均衡测试")
    print("=" * 60)

    capability_matrix = CapabilityMatrix()
    load_balancer = LoadBalancer()

    # 模拟10个任务
    tasks = []
    for i in range(10):
        task = {"id": f"task_{i}", "coder": f"def test_{i}(): pass"}
        tasks.append(task)

    # 分配任务
    assignments = []
    for task in tasks:
        agents = ["analyzer", "security", "reporter"]
        selected = load_balancer.assign_task(task, agents, capability_matrix)
        assignments.append(selected)

    # 统计分配情况
    from collections import Counter

    assignment_counts = Counter(assignments)

    print(f"任务分配统计:")
    for agent, count in assignment_counts.items():
        print(f"  {agent}: {count} 个任务")

    # 计算负载均衡度
    balance_degree = load_balancer.balance_degree
    print(f"\n负载均衡度: {balance_degree:.2%}")

    if balance_degree > 0.85:
        print("✅ 负载均衡测试通过")
    else:
        print("❌ 负载均衡测试失败")

    return balance_degree


async def test_reliability():
    """测试消息可靠性"""
    print("\n" + "=" * 60)
    print("测试4: 消息可靠性测试")
    print("=" * 60)

    reliability = MessageReliability(max_retries=3)

    # 模拟发送失败的消息
    async def failing_send(message):
        raise Exception("模拟网络故障")

    # 测试重试机制
    success = await reliability.send_with_retry({"test": "message"}, failing_send)

    print(f"发送结果: {'成功' if success else '失败'}")
    print(f"消息丢失率: {reliability.loss_rate:.2%}")

    if reliability.loss_rate < 0.001:  # < 0.1%
        print("✅ 消息可靠性测试通过")
    else:
        print("❌ 消息可靠性测试失败")

    return reliability.loss_rate


# ================ 主程序 ================


async def main():
    """主函数"""
    print("=" * 60)
    print("生产级Multi-Agent系统测试套件")
    print("=" * 60)

    # 运行测试
    test_results = {}

    # 测试1: 基本工作流
    test_results["basic_workflow"] = await test_basic_workflow()

    # 测试2: 并行工作流
    test_results["parallel_workflow"] = await test_parallel_workflow()

    # 测试3: 负载均衡
    test_results["load_balancing"] = await test_load_balancing()

    # 测试4: 消息可靠性
    test_results["reliability"] = await test_reliability()

    # 生成测试报告
    print("\n" + "=" * 60)
    print("测试报告摘要")
    print("=" * 60)

    success_count = 0
    total_count = len(test_results)

    for test_name, result in test_results.items():
        if test_name == "load_balancing":
            passed = result > 0.85
        elif test_name == "reliability":
            passed = result < 0.001
        elif test_name in ["basic_workflow", "parallel_workflow"]:
            passed = result.get("success", False)
        else:
            passed = False

        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20} {status}")

        if passed:
            success_count += 1

    print(
        f"\n总通过率: {success_count}/{total_count} ({success_count / total_count:.0%})"
    )

    # 验收标准检查
    print("\n" + "=" * 60)
    print("验收标准检查")
    print("=" * 60)

    # 1. 端到端任务执行成功
    e2e_success = test_results["basic_workflow"]["success"]
    print(f"1. 端到端任务执行: {'✅ 通过' if e2e_success else '❌ 失败'}")

    # 2. 消息丢失率 < 0.1%
    message_loss = test_results["reliability"]
    message_ok = message_loss < 0.001
    print(
        f"2. 消息丢失率 < 0.1%: {'✅ 通过' if message_ok else '❌ 失败'} ({message_loss:.2%})"
    )

    # 3. 负载均衡度 > 85%
    balance_degree = test_results["load_balancing"]
    balance_ok = balance_degree > 0.85
    print(
        f"3. 负载均衡度 > 85%: {'✅ 通过' if balance_ok else '❌ 失败'} ({balance_degree:.2%})"
    )

    # 4. 故障恢复测试（简化）
    print(f"4. 故障恢复机制: ⚠️ 模拟实现（需真实环境测试）")

    overall_pass = e2e_success and message_ok and balance_ok
    print(f"\n总体验收: {'✅ 通过' if overall_pass else '❌ 失败'}")

    return overall_pass


if __name__ == "__main__":
    # 运行主程序
    success = asyncio.run(main())

    if success:
        print("\n🎉 Multi-Agent系统生产级实现完成！")
    else:
        print("\n⚠️  部分测试未通过，需要优化改进。")

    exit(0 if success else 1)
