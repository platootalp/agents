# Agent配置管理模块
"""
统一管理智能旅行助手的配置参数
"""

import os
import logging
from typing import Optional

# 配置日志
logger = logging.getLogger(__name__)

# 尝试加载.env文件
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv module not found, skipping .env file loading")
except Exception as e:
    logger.warning(f"Failed to load .env file: {e}")


class AgentConfig:
    """
    智能旅行助手配置类，统一管理所有配置参数
    """
    
    # 从环境变量加载配置或使用默认值
    # 系统提示词
    SYSTEM_PROMPT = os.environ.get(
        "AGENT_SYSTEM_PROMPT",
        """
        你是一个专业的智能旅行助手，能够帮助用户规划旅行行程并提供相关信息。
        
        ## 核心能力：
        - 理解用户的旅行需求和问题
        - 合理使用可用工具获取必要信息
        - 基于获取的信息提供专业、实用的旅行建议
        
        ## 可用工具：
        - `get_weather(city: str)`: 查询指定城市的实时天气
        - `get_attractions(city: str, weather: Optional[str] = None)`: 根据城市和天气推荐景点
        
        ## 工作流程：
        1. 分析用户请求，确定需要的信息
        2. 使用适当的工具获取信息
        3. 整合信息并提供最终答案
        4. 保持对话友好、专业，使用中文回答
        
        ## 输出格式：
        当需要调用工具时，请严格使用以下格式：
        ```
        Thought: [你的思考过程]
        Action: tool_name(parameter="value")
        ```
        
        当任务完成，不再需要调用工具时，请直接返回最终答案，不需要使用任何工具。
        """
    )
    
    # 默认模型名称
    DEFAULT_MODEL_NAME = os.environ.get("AGENT_DEFAULT_MODEL", "qwen-flash")
    
    # 默认线程ID
    DEFAULT_THREAD_ID = os.environ.get("AGENT_DEFAULT_THREAD_ID", "default")
    
    # 日志级别
    LOG_LEVEL = os.environ.get("AGENT_LOG_LEVEL", "INFO").upper()
    
    # 流模式
    STREAM_MODE = os.environ.get("AGENT_STREAM_MODE", "values")
    
    @classmethod
    def validate(cls) -> bool:
        """
        验证配置的有效性
        
        Returns:
            bool: 配置是否有效
        """
        try:
            # 验证日志级别
            if cls.LOG_LEVEL not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                logger.warning(f"Invalid log level: {cls.LOG_LEVEL}, using INFO instead")
                cls.LOG_LEVEL = "INFO"
            
            # 验证流模式
            if cls.STREAM_MODE not in ["values", "updates"]:
                logger.warning(f"Invalid stream mode: {cls.STREAM_MODE}, using 'values' instead")
                cls.STREAM_MODE = "values"
            
            logger.info("Agent configuration validation passed")
            return True
        except Exception as e:
            logger.error(f"Agent configuration validation failed: {e}")
            return False
