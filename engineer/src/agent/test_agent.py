#!/usr/bin/env python3
"""
测试智能旅行助手代理的功能
"""

import os
import sys
# 将当前脚本所在目录的父目录（即项目根目录）添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
from src.agent.agent import TravelAssistantAgent

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_agent():
    """
    测试智能旅行助手代理的基本功能
    """
    logger.info("开始测试智能旅行助手代理")
    
    try:
        # 创建代理实例（测试模式下不需要API密钥）
        agent = TravelAssistantAgent()
        logger.info("成功创建旅行助手代理实例")
        
        # 测试天气查询功能
        logger.info("\n测试天气查询功能：")
        results = agent.process_request("上海天气怎么样？")
        response = agent.get_response_content(results)
        logger.info(f"天气查询响应：{response}")
        
        # 测试景点推荐功能
        logger.info("\n测试景点推荐功能：")
        results = agent.process_request("上海有什么好玩的？")
        response = agent.get_response_content(results)
        logger.info(f"景点推荐响应：{response}")
        
        # 测试旅行规划功能
        logger.info("\n测试旅行规划功能：")
        results = agent.plan_trip("北京", 3, ["历史文化", "美食"])
        response = agent.get_response_content(results)
        logger.info(f"旅行规划响应：{response}")
        
        logger.info("\n所有测试完成！")
        return True
    except Exception as e:
        logger.error(f"测试失败：{e}", exc_info=True)
        return False

if __name__ == "__main__":
    test_agent()