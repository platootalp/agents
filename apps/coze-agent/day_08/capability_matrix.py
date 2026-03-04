"""
能力矩阵与负载均衡模块
实现智能任务分配算法
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import time
import asyncio


@dataclass
class AgentCapability:
    """Agent能力定义"""
    agent_id: str
    capabilities: Dict[str, float]  # 能力类型 -> 权重分数（0-1）
    current_load: float = 0.0  # 当前负载（0-1）
    total_tasks: int = 0  # 处理的总任务数
    success_rate: float = 1.0  # 成功率
    avg_response_time: float = 0.0  # 平均响应时间（毫秒）
    
    def get_capability_score(self, required_capabilities: Dict[str, float]) -> float:
        """计算对特定任务的能力得分"""
        if not required_capabilities:
            return 0.0
        
        total_weight = sum(required_capabilities.values())
        if total_weight == 0:
            return 0.0
        
        score = 0.0
        for capability, weight in required_capabilities.items():
            if capability in self.capabilities:
                score += self.capabilities[capability] * weight
        
        # 归一化
        score /= total_weight
        
        # 负载惩罚
        load_penalty = 1.0 - self.current_load
        
        # 成功率奖励
        success_bonus = self.success_rate
        
        # 响应时间惩罚（越慢惩罚越大）
        time_penalty = 1.0 / (1.0 + self.avg_response_time / 1000.0)
        
        # 综合得分
        final_score = score * load_penalty * success_bonus * time_penalty
        
        return final_score
    
    def update_after_task(self, success: bool, response_time: float):
        """任务完成后更新Agent状态"""
        self.total_tasks += 1
        
        # 更新成功率
        if self.total_tasks == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            if success:
                self.success_rate = (self.success_rate * (self.total_tasks - 1) + 1) / self.total_tasks
            else:
                self.success_rate = (self.success_rate * (self.total_tasks - 1)) / self.total_tasks
        
        # 更新平均响应时间
        if self.total_tasks == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (
                self.avg_response_time * (self.total_tasks - 1) + response_time
            ) / self.total_tasks
        
        # 降低负载（任务完成后）
        self.current_load = max(0.0, self.current_load - 0.05)


class CapabilityMatrix:
    """能力矩阵管理器"""
    
    def __init__(self):
        self.agents: Dict[str, AgentCapability] = {}
        self.task_history: List[Dict[str, Any]] = []
        
        # 初始化默认Agent
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """初始化默认Agent"""
        # 分析Agent
        self.agents["analyzer"] = AgentCapability(
            agent_id="analyzer",
            capabilities={
                "structural_analysis": 0.9,
                "complexity_calculation": 0.8,
                "style_checking": 0.7,
                "code_optimization": 0.6
            }
        )
        
        # 安全Agent
        self.agents["security"] = AgentCapability(
            agent_id="security",
            capabilities={
                "sql_injection": 0.95,
                "xss_detection": 0.85,
                "hardcoded_secrets": 0.9,
                "command_injection": 0.8
            }
        )
        
        # 报告Agent
        self.agents["reporter"] = AgentCapability(
            agent_id="reporter",
            capabilities={
                "result_aggregation": 0.8,
                "report_generation": 0.9,
                "format_beautification": 0.7,
                "summary_extraction": 0.75
            }
        )
    
    def analyze_task_requirements(self, code: str) -> Dict[str, float]:
        """分析任务需求，返回各能力权重"""
        requirements = {}
        
        # 基于代码特征分析
        code_lower = code.lower()
        
        # 数据库相关特征
        db_keywords = ["select", "insert", "update", "delete", "from", "where"]
        has_db = any(keyword in code_lower for keyword in db_keywords)
        
        # 安全相关特征
        has_eval = "eval(" in code_lower
        has_exec = "exec(" in code_lower or "execfile" in code_lower
        has_shell = "shell=true" in code_lower or "subprocess" in code_lower
        
        # 复杂度特征
        lines = code.count('\n') + 1
        has_nested = code.count('    ') > 10  # 简单嵌套检测
        
        # 根据特征分配权重
        if has_db:
            requirements["sql_injection"] = 0.4
            requirements["structural_analysis"] = 0.3
            requirements["complexity_calculation"] = 0.3
            
        elif has_eval or has_exec or has_shell:
            requirements["command_injection"] = 0.5
            requirements["xss_detection"] = 0.3
            requirements["security"] = 0.2
            
        elif lines > 50 or has_nested:
            requirements["complexity_calculation"] = 0.4
            requirements["structural_analysis"] = 0.4
            requirements["style_checking"] = 0.2
            
        else:
            # 默认分配
            requirements["structural_analysis"] = 0.4
            requirements["style_checking"] = 0.3
            requirements["complexity_calculation"] = 0.3
        
        # 确保权重和为1
        total = sum(requirements.values())
        if total > 0:
            requirements = {k: v/total for k, v in requirements.items()}
        
        return requirements
    
    def select_best_agent(self, task_requirements: Dict[str, float], 
                         exclude_agents: List[str] = None) -> Tuple[str, float]:
        """选择最适合的Agent"""
        exclude_agents = exclude_agents or []
        
        best_agent = None
        best_score = -1.0
        
        for agent_id, agent in self.agents.items():
            if agent_id in exclude_agents:
                continue
            
            score = agent.get_capability_score(task_requirements)
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent, best_score
    
    def update_agent_load(self, agent_id: str, task_complexity: float = 0.1):
        """更新Agent负载"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.current_load = min(1.0, agent.current_load + task_complexity)
    
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """获取Agent统计信息"""
        if agent_id not in self.agents:
            return {}
        
        agent = self.agents[agent_id]
        
        return {
            "agent_id": agent.agent_id,
            "current_load": agent.current_load,
            "total_tasks": agent.total_tasks,
            "success_rate": agent.success_rate,
            "avg_response_time": agent.avg_response_time,
            "capabilities": agent.capabilities
        }
    
    def get_balance_degree(self) -> float:
        """计算负载均衡度"""
        if not self.agents:
            return 1.0
        
        loads = [agent.current_load for agent in self.agents.values()]
        
        # 计算标准差
        mean_load = np.mean(loads)
        variance = np.mean([(load - mean_load) ** 2 for load in loads])
        std_dev = np.sqrt(variance)
        
        # 最大可能标准差
        max_possible_std = np.sqrt(0.25)  # 负载在0-1范围内，最大方差为0.25
        
        if max_possible_std == 0:
            return 1.0
        
        # 均衡度 = 1 - 归一化的标准差
        normalized_std = std_dev / max_possible_std
        
        return 1.0 - normalized_std


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self.capability_matrix = CapabilityMatrix()
        self.assignment_history: List[Dict[str, Any]] = []
        self.task_queue: List[Dict[str, Any]] = []
        
        # 负载均衡策略
        self.strategy = "weighted_score"  # weighted_score, round_robin, least_load
        self.round_robin_index = 0
        
    def assign_task(self, task: Dict[str, Any], 
                   available_agents: List[str] = None) -> str:
        """分配任务给最合适的Agent"""
        available_agents = available_agents or list(self.capability_matrix.agents.keys())
        
        # 分析任务需求
        code = task.get("code", "")
        priority = task.get("priority", 1)
        
        requirements = self.capability_matrix.analyze_task_requirements(code)
        
        # 根据策略选择Agent
        if self.strategy == "round_robin":
            selected_agent = available_agents[self.round_robin_index % len(available_agents)]
            self.round_robin_index += 1
            
        elif self.strategy == "least_load":
            # 选择负载最低的Agent
            agent_loads = []
            for agent_id in available_agents:
                agent = self.capability_matrix.agents.get(agent_id)
                if agent:
                    agent_loads.append((agent_id, agent.current_load))
            
            if agent_loads:
                selected_agent = min(agent_loads, key=lambda x: x[1])[0]
            else:
                selected_agent = available_agents[0]
                
        else:  # weighted_score (默认)
            best_agent, best_score = self.capability_matrix.select_best_agent(
                requirements, 
                exclude_agents=[a for a in self.capability_matrix.agents.keys() 
                               if a not in available_agents]
            )
            
            if best_agent is None:
                selected_agent = available_agents[0]
            else:
                selected_agent = best_agent
        
        # 更新Agent负载（根据优先级调整复杂度）
        task_complexity = 0.05 + (priority - 1) * 0.02
        self.capability_matrix.update_agent_load(selected_agent, task_complexity)
        
        # 记录分配历史
        assignment_record = {
            "task_id": task.get("id", str(time.time())),
            "assigned_agent": selected_agent,
            "requirements": requirements,
            "priority": priority,
            "timestamp": time.time(),
            "strategy": self.strategy
        }
        
        self.assignment_history.append(assignment_record)
        
        print(f"[负载均衡器] 任务 {task.get('id', 'unknown')} 分配给 {selected_agent}, "
              f"策略: {self.strategy}, 优先级: {priority}")
        
        return selected_agent
    
    def update_task_result(self, task_id: str, agent_id: str, 
                          success: bool, response_time: float):
        """更新任务结果，用于调整Agent能力评估"""
        if agent_id in self.capability_matrix.agents:
            agent = self.capability_matrix.agents[agent_id]
            agent.update_after_task(success, response_time)
            
            # 更新历史记录
            for record in self.assignment_history:
                if record.get("task_id") == task_id:
                    record["completed"] = True
                    record["success"] = success
                    record["response_time"] = response_time
                    record["completion_time"] = time.time()
                    break
    
    def get_assignment_stats(self) -> Dict[str, Any]:
        """获取分配统计信息"""
        if not self.assignment_history:
            return {}
        
        # 按Agent统计
        agent_counts = Counter([h["assigned_agent"] for h in self.assignment_history])
        
        # 计算成功率
        completed_tasks = [h for h in self.assignment_history if h.get("completed")]
        success_rate = 0.0
        if completed_tasks:
            success_rate = sum(1 for t in completed_tasks if t.get("success")) / len(completed_tasks)
        
        # 平均响应时间
        avg_response_time = 0.0
        if completed_tasks:
            avg_response_time = np.mean([t.get("response_time", 0) for t in completed_tasks])
        
        return {
            "total_tasks": len(self.assignment_history),
            "completed_tasks": len(completed_tasks),
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "agent_distribution": dict(agent_counts),
            "balance_degree": self.capability_matrix.get_balance_degree()
        }
    
    def set_strategy(self, strategy: str):
        """设置负载均衡策略"""
        valid_strategies = ["weighted_score", "round_robin", "least_load"]
        if strategy in valid_strategies:
            self.strategy = strategy
            print(f"[负载均衡器] 策略切换为: {strategy}")
        else:
            print(f"[负载均衡器] 无效策略: {strategy}")


async def test_capability_matrix():
    """测试能力矩阵"""
    print("\n测试能力矩阵与负载均衡...")
    
    cm = CapabilityMatrix()
    lb = LoadBalancer()
    
    # 测试任务需求分析
    test_code = """
def vulnerable_query(user_input):
    import sqlite3
    conn = sqlite3.connect('db.sqlite')
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return conn.execute(query).fetchall()
    """
    
    requirements = cm.analyze_task_requirements(test_code)
    print(f"任务需求分析: {requirements}")
    
    # 测试Agent选择
    best_agent, best_score = cm.select_best_agent(requirements)
    print(f"最佳Agent: {best_agent}, 得分: {best_score:.3f}")
    
    # 测试负载均衡
    tasks = []
    for i in range(5):
        task = {
            "id": f"task_{i}",
            "code": test_code,
            "priority": (i % 3) + 1
        }
        tasks.append(task)
    
    print("\n负载均衡分配测试:")
    for task in tasks:
        agent = lb.assign_task(task)
        print(f"任务 {task['id']} (优先级 {task['priority']}) -> {agent}")
    
    # 模拟任务完成
    for i, task in enumerate(tasks):
        success = i != 2  # 第三个任务失败
        response_time = 100 + i * 50
        lb.update_task_result(task["id"], task["assigned_agent"], 
                             success, response_time)
    
    # 获取统计信息
    stats = lb.get_assignment_stats()
    print(f"\n负载均衡统计:")
    print(f"- 总任务数: {stats['total_tasks']}")
    print(f"- 完成率: {stats['completed_tasks'] / stats['total_tasks']:.1%}")
    print(f"- 成功率: {stats['success_rate']:.1%}")
    print(f"- 负载均衡度: {stats['balance_degree']:.2%}")
    print(f"- Agent分布: {stats['agent_distribution']}")
    
    balance_ok = stats["balance_degree"] > 0.85
    
    if balance_ok:
        print("✅ 负载均衡测试通过")
    else:
        print("❌ 负载均衡测试失败")
    
    return balance_ok


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_capability_matrix())