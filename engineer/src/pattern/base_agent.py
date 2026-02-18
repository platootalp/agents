"""
Base Agent - Agent 基类
定义所有 Agent 的通用接口和能力
"""
# genAI_main_start
from abc import ABC, abstractmethod
from typing import Optional
from core import QwenLLM
from pattern.tools import ToolExecutor


class BaseAgent(ABC):
    """
    Agent 基类，提供通用能力和接口定义
    
    所有具体的 Agent 实现都应该继承此类，并实现 run() 方法
    """
    
    def __init__(
        self,
        llm_client: QwenLLM,
        tool_executor: Optional[ToolExecutor] = None
    ):
        """
        初始化 BaseAgent
        
        Args:
            llm_client: LLM 客户端实例，所有 agent 都需要
            tool_executor: 工具执行器，可选，部分 agent 需要
        """
        self.llm = llm_client
        self.tool_executor = tool_executor
    
    @abstractmethod
    def run(self, question: str) -> str:
        """
        执行 agent 的主要逻辑
        
        Args:
            question: 用户问题或任务描述
            
        Returns:
            str: agent 的最终答案
        """
        pass
    
    def _print_stage(self, title: str, content: str):
        """
        打印阶段信息（通用工具方法）
        
        Args:
            title: 阶段标题
            content: 阶段内容
        """
        print(f"\n--- {title} ---")
        print(content)
    
    def _print_section(self, title: str):
        """
        打印章节标题（通用工具方法）
        
        Args:
            title: 章节标题
        """
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
    
    def has_tools(self) -> bool:
        """
        检查 agent 是否支持工具调用
        
        Returns:
            bool: 如果 tool_executor 不为 None，返回 True
        """
        return self.tool_executor is not None
    
    def call_llm(self, prompt: str, strip: bool = True) -> str:
        """
        调用 LLM 的通用方法
        
        Args:
            prompt: 提示词内容
            strip: 是否去除首尾空白字符
            
        Returns:
            str: LLM 的响应
        """
        response = self.llm.think([{"role": "user", "content": prompt}])
        return response.strip() if strip else response
    
    def execute_tool(self, tool_name: str, input_data: str) -> str:
        """
        执行工具的通用方法
        
        Args:
            tool_name: 工具名称
            input_data: 工具输入数据
            
        Returns:
            str: 工具执行结果
            
        Raises:
            ValueError: 如果 agent 不支持工具调用
        """
        if not self.has_tools():
            raise ValueError("此 Agent 不支持工具调用，请提供 tool_executor")
        return self.tool_executor.execute_tool(tool_name, input_data)
    
    def get_tool_descriptions(self) -> str:
        """
        获取工具描述的通用方法
        
        Returns:
            str: 工具描述字符串
            
        Raises:
            ValueError: 如果 agent 不支持工具调用
        """
        if not self.has_tools():
            raise ValueError("此 Agent 不支持工具调用，请提供 tool_executor")
        return self.tool_executor.get_tool_descriptions()
    
    def _print_step(self, step_num: int, total: int, step_desc: str, result: str = ""):
        """
        打印步骤信息（通用工具方法）
        
        Args:
            step_num: 当前步骤编号
            total: 总步骤数
            step_desc: 步骤描述
            result: 步骤结果（可选）
        """
        if result:
            print(f"\n--- 执行步骤 [{step_num}/{total}]: {step_desc} ---")
            print(f"结果: {result}")
        else:
            print(f"\n--- Step {step_num}/{total}: {step_desc} ---")
    
    def _print_final_answer(self, answer: str):
        """
        打印最终答案（通用工具方法）
        
        Args:
            answer: 最终答案
        """
        self._print_stage("最终答案", answer)
# genAI_main_end
