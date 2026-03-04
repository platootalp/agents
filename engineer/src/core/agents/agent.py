# 智能旅行助手 - Travel Assistant Agent
"""
一个功能完善的智能旅行助手，能够根据用户需求提供旅游建议和信息查询服务。
"""

import os
import logging
import re
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime

# 导入自定义工具和配置
# 支持作为包模块导入和独立脚本运行
try:
    from .tools import get_weather, get_attractions
    from .config import AgentConfig
except ImportError:
    # 作为独立脚本运行时，使用绝对导入
    import sys
    import os

    # 添加项目根目录到 Python 路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    from src.agent.tools import get_weather, get_attractions
    from src.agent.config import AgentConfig

# 配置日志
logging.basicConfig(level=getattr(logging, AgentConfig.LOG_LEVEL),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TravelAssistantAgent:
    """
    智能旅行助手类，提供完整的旅行规划和信息查询功能。
    
    功能特点：
    - 基于纯OpenAI API实现
    - 支持多轮对话和上下文记忆
    - 集成天气查询和景点推荐工具
    - 结构化的工具调用和响应处理
    - 可配置的系统提示词和模型参数
    
    Attributes:
        model_name: 使用的LLM模型名称
        thread_id: 对话线程ID，用于记忆管理
        api_key: OpenAI API密钥
        api_base: OpenAI API基础URL
        system_prompt: 系统提示词
        tools: 可用工具字典
        conversation_history: 对话历史记录
    """

    # 从配置类导入系统提示词
    SYSTEM_PROMPT = AgentConfig.SYSTEM_PROMPT

    def __init__(self, model_name: Optional[str] = None, thread_id: Optional[str] = None):
        """
        初始化旅行助手
        
        Args:
            model_name: 使用的LLM模型名称（默认使用配置中的DEFAULT_MODEL_NAME）
            thread_id: 对话线程ID，用于记忆管理（默认使用配置中的DEFAULT_THREAD_ID）
        """
        self.model_name = model_name or AgentConfig.DEFAULT_MODEL_NAME
        self.thread_id = thread_id or AgentConfig.DEFAULT_THREAD_ID
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        self.api_base = os.environ.get("DASHSCOPE_API_URL", "https://api.openai.com/v1")

        if not self.api_key:
            # 如果没有API密钥，设置为测试模式
            logger.warning("OPENAI_API_KEY environment variable not set, running in test mode")
            self._test_mode = True
        else:
            self._test_mode = False

        # 初始化工具和对话历史
        self.tools = {
            "get_weather": get_weather,
            "get_attractions": get_attractions
        }
        self.conversation_history = []

        logger.info("Travel assistant initialized successfully")

    def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        """
        调用OpenAI API获取响应
        
        Args:
            messages: 消息列表，包含系统提示和对话历史
            
        Returns:
            str: OpenAI API的响应内容或测试模式下的模拟响应
        """
        # 测试模式下返回模拟响应
        if self._test_mode:
            logger.info("Running in test mode, returning mock response")
            user_query = messages[-1]["content"]

            # 模拟工具调用响应
            if "天气" in user_query and "上海" in user_query:
                return "Thought: 用户想知道上海的天气，我需要调用get_weather工具查询。\nAction: get_weather(city=\"上海\")"
            elif "景点" in user_query and "上海" in user_query:
                return "Thought: 用户想了解上海的景点，我需要调用get_attractions工具获取推荐。\nAction: get_attractions(city=\"上海\")"
            # 模拟工具调用后返回最终结果的情况
            elif "工具调用结果" in user_query:
                # 解析工具结果并返回最终回答
                if "上海" in user_query and "天气" in user_query:
                    return "根据查询结果，上海今天的天气是晴天，气温在15-25°C之间，空气质量良好。非常适合户外活动！"
                elif "上海" in user_query and "景点" in user_query:
                    return "根据推荐，上海的热门景点包括外滩、东方明珠、豫园和迪士尼乐园等。外滩适合欣赏夜景，东方明珠可以俯瞰上海全景，豫园展现了传统园林文化，迪士尼乐园则是家庭游的好去处。"
            else:
                return "这是一个测试响应，实际使用时会调用OpenAI API生成真实回答。"

        # 正常模式下调用OpenAI API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}", exc_info=True)
            raise RuntimeError(f"OpenAI API call failed: {str(e)}") from e

    def _parse_tool_call(self, content: str) -> Dict[str, Any]:
        """
        解析模型输出中的工具调用
        
        Args:
            content: 模型输出内容
            
        Returns:
            Dict[str, Any]: 包含工具名称、参数和思考过程的字典
        """
        # 提取Thought和Action部分
        thought_match = re.search(r"Thought: (.*)", content)
        action_match = re.search(r"Action: (\w+)\((.*)\)", content)

        if thought_match and action_match:
            thought = thought_match.group(1)
            tool_name = action_match.group(1)
            tool_params_str = action_match.group(2)

            # 解析参数
            params = {}
            param_matches = re.findall(r"(\w+)=(\"([^\"]*)\"|'([^\']*)')", tool_params_str)
            for match in param_matches:
                key = match[0]
                value = match[2] if match[2] else match[3]
                params[key] = value

            return {
                "thought": thought,
                "tool_name": tool_name,
                "params": params
            }

        return None

    def process_request(self, user_query: str) -> List[Dict[str, Any]]:
        """
        处理用户请求并返回响应
        
        Args:
            user_query: 用户的旅行相关问题
            
        Returns:
            List[Dict[str, Any]]: 包含代理响应的列表，每个响应包含以下字段：
                - type: 消息类型（"human", "ai", "tool", "error"）
                - content: 响应内容
                - timestamp: 消息标识（使用时间戳）
                
        Examples:
            >>> agent = TravelAssistantAgent()
            >>> response = agent.process_request("北京天气怎么样？")
            >>> print(response[0]["content"])
            北京当前天气：晴，气温 20°C，湿度 50%，风速 10 km/h
        """
        if not user_query:
            logger.warning("Received empty user query")
            return [{"type": "error", "content": "请输入有效的旅行问题"}]

        logger.info(f"Processing user request: {user_query}")

        # 处理响应
        results = []
        timestamp = datetime.now().isoformat()

        try:
            # 添加用户查询到对话历史
            self.conversation_history.append({"role": "user", "content": user_query})
            results.append({
                "type": "human",
                "content": user_query,
                "timestamp": timestamp
            })

            # 输出用户问题
            print(f"\n{'=' * 60}")
            print(f"User Question:")
            print(f"{'=' * 60}")
            print(f"{user_query}")

            # ReAct循环：持续调用API直到不需要工具调用
            while True:
                # 构建消息列表
                messages = [
                               {"role": "system", "content": self.SYSTEM_PROMPT}
                           ] + self.conversation_history

                # 调用OpenAI API
                response_content = self._call_openai(messages)
                logger.debug(f"OpenAI response: {response_content[:100]}...")

                # 添加AI响应到对话历史
                self.conversation_history.append({"role": "assistant", "content": response_content})

                # 检查是否需要调用工具
                tool_call = self._parse_tool_call(response_content)
                if tool_call:
                    # 输出思考过程
                    print(f"\n{'=' * 60}")
                    print(f"Thought: {tool_call['thought']}")
                    print(f"{'=' * 60}")

                    # 添加思考过程到结果
                    results.append({
                        "type": "ai",
                        "content": f"Thought: {tool_call['thought']}",
                        "timestamp": timestamp
                    })

                    # 调用工具
                    tool_name = tool_call["tool_name"]
                    if tool_name in self.tools:
                        try:
                            # 输出工具调用信息
                            params_str = ', '.join([f'{k}={v!r}' for k, v in tool_call['params'].items()])
                            print(f"\n{'=' * 60}")
                            print(f"Action: {tool_name}({params_str})")
                            print(f"{'=' * 60}")

                            tool_result = self.tools[tool_name](**tool_call["params"])
                            logger.info(f"Tool {tool_name} executed successfully")

                            # 输出工具调用结果
                            print(f"\n{'=' * 60}")
                            print(f"Tool Result:")
                            print(f"{'=' * 60}")
                            print(f"{tool_result}")

                            # 添加工具调用和结果到结果列表
                            results.append({
                                "type": "ai",
                                "content": f"Action: {tool_name}({', '.join([f'{k}={v!r}' for k, v in tool_call['params'].items()])})",
                                "timestamp": timestamp
                            })

                            results.append({
                                "type": "tool",
                                "content": tool_result,
                                "timestamp": timestamp
                            })

                            # 将工具结果添加到对话历史
                            self.conversation_history.append({"role": "user", "content": f"工具调用结果：{tool_result}"})
                        except Exception as e:
                            logger.error(f"Tool {tool_name} execution failed: {e}", exc_info=True)
                            error_msg = f"工具调用失败: {str(e)}"
                            print(f"\n{'=' * 60}")
                            print(f"Error: {error_msg}")
                            print(f"{'=' * 60}")
                            results.append({
                                "type": "error",
                                "content": error_msg,
                                "timestamp": timestamp
                            })
                            break
                    else:
                        error_msg = f"不支持的工具: {tool_name}"
                        logger.error(error_msg)
                        print(f"\n{'=' * 60}")
                        print(f"Error: {error_msg}")
                        print(f"{'=' * 60}")
                        results.append({
                            "type": "error",
                            "content": error_msg,
                            "timestamp": timestamp
                        })
                        break
                else:
                    # 不需要工具调用，直接返回AI响应
                    print(f"\n{'=' * 60}")
                    print(f"Final Answer:")
                    print(f"{'=' * 60}")
                    print(f"{response_content}")
                    results.append({
                        "type": "ai",
                        "content": response_content,
                        "timestamp": timestamp
                    })
                    break

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            error_msg = f"处理请求时发生错误: {str(e)}"

            # 提供更具体的错误信息
            if isinstance(e, KeyError):
                error_msg += "（可能是数据格式错误）"
            elif isinstance(e, RuntimeError):
                error_msg += "（运行时错误）"
            elif isinstance(e, ImportError):
                error_msg += "（缺少依赖包）"

            results.append({
                "type": "error",
                "content": error_msg,
                "timestamp": timestamp
            })

        return results

    def get_response_content(self, results: List[Dict[str, Any]]) -> str:
        """
        从处理结果中提取最终响应内容
        
        Args:
            results: 处理请求的结果列表，由process_request方法返回
            
        Returns:
            str: 最终的文本响应内容，优先返回最后一个AI响应（排除工具调用的思考过程）
            
        Examples:
            >>> agent = TravelAssistantAgent()
            >>> results = agent.process_request("上海有什么好玩的？")
            >>> response = agent.get_response_content(results)
            >>> print(response)
            上海有很多值得一去的景点，推荐您去外滩、东方明珠和豫园等。
        """
        # 优先返回最后一个真正的AI响应（排除工具调用的思考过程）
        for result in reversed(results):
            if result["type"] == "ai":
                content = result["content"]
                # 排除工具调用的思考过程（包含Thought和Action）
                if not ("Thought:" in content and "Action:" in content):
                    return content

        # 如果没有真正的AI响应，返回最后一个工具响应
        for result in reversed(results):
            if result["type"] == "tool":
                return result["content"]

        # 如果没有工具响应，返回最后一个AI响应（包括思考过程）作为最后的手段
        for result in reversed(results):
            if result["type"] == "ai":
                return result["content"]

        return "抱歉，未能生成有效响应。"

    def plan_trip(self, destination: str, travel_days: int, interests: Optional[List[str]] = None) -> List[
        Dict[str, Any]]:
        """
        规划完整的旅行行程
        
        Args:
            destination: 目的地城市
            travel_days: 旅行天数
            interests: 用户感兴趣的活动类型列表（可选）
            
        Returns:
            List[Dict[str, Any]]: 包含规划结果的响应列表，格式与process_request方法返回结果一致
            
        Examples:
            >>> agent = TravelAssistantAgent()
            >>> results = agent.plan_trip("杭州", 3, ["自然风光", "美食"])
            >>> print(agent.get_response_content(results))
            3天杭州之旅行程规划：
            第一天：西湖游览...
            第二天：灵隐寺...
            第三天：杭州美食之旅...
        """
        logger.info(f"Planning trip to {destination} for {travel_days} days, interests: {interests}")

        # 构造旅行规划请求
        interest_str = ", ".join(interests) if interests else "自然风光和文化景点"
        query = f"帮我规划一次{travel_days}天的{destination}之旅，我对{interest_str}感兴趣。请提供详细的每日行程安排，包括推荐的景点、美食和交通建议。"

        # 使用现有的process_request方法处理
        return self.process_request(query)


def plan_agent(model_name: Optional[str] = None) -> TravelAssistantAgent:
    """
    创建一个专门用于旅行规划的智能代理
    
    Args:
        model_name: 使用的LLM模型名称，默认为配置中的DEFAULT_MODEL_NAME
        
    Returns:
        TravelAssistantAgent: 创建的旅行助手实例
        
    Raises:
        RuntimeError: 代理创建过程中发生错误时抛出
        
    Examples:
        >>> agent = plan_agent()
        >>> results = agent.plan_trip("上海", 3)
        >>> print(agent.get_response_content(results))
        上海3日游行程规划：...
    """
    try:
        # 使用配置中的默认模型名称
        actual_model_name = model_name or AgentConfig.DEFAULT_MODEL_NAME
        logger.info(f"Creating travel planning agent with model: {actual_model_name}")

        # 创建并返回TravelAssistantAgent实例
        agent = TravelAssistantAgent(model_name=actual_model_name)
        logger.info("Travel planning agent created successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to create travel planning agent: {e}", exc_info=True)
        raise RuntimeError(f"Travel planning agent creation failed: {str(e)}") from e


# 命令行测试入口
def main() -> None:
    """
    命令行测试旅行助手功能
    
    该函数提供一个交互式命令行界面，用户可以输入旅行问题，获取智能助手的回答。
    支持的命令：
    - 旅行相关问题（如："北京天气怎么样？", "上海有什么好玩的？"）
    - "退出"、"quit"或"exit"：结束对话
    
    Examples:
        $ python agent.py
        === 智能旅行助手 ===
        输入您的旅行问题，输入 '退出' 结束对话
        ==============================
        
        您的问题: 北京天气怎么样？
        助手回答: 北京当前天气：晴，气温 20°C，湿度 50%，风速 10 km/h
        
        您的问题: 退出
        感谢使用智能旅行助手，再见！
    """
    print("=== 智能旅行助手 ===")
    print("输入您的旅行问题，输入 '退出' 结束对话")
    print("=" * 30)

    # 初始化助手
    agent = TravelAssistantAgent(thread_id="test_thread")

    while True:
        try:
            user_input = input("\n您的问题: ").strip()
            if user_input.lower() in ["退出", "quit", "exit"]:
                print("感谢使用智能旅行助手，再见！")
                break

            if not user_input:
                continue

            # 处理请求
            results = agent.process_request(user_input)
            response = agent.get_response_content(results)

        except KeyboardInterrupt:
            print("\n感谢使用智能旅行助手，再见！")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
