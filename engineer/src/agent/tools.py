import os
import logging
import requests
from typing import Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_weather(city: str) -> str:
    """
    通过调用wttr.in API查询指定城市的实时天气信息。
    
    Args:
        city: 城市名称（中文或英文）
        
    Returns:
        格式化的天气信息
    """
    logger.info(f"Querying weather for city: {city}")
    
    # API端点，请求JSON格式数据
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 发起网络请求
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 检查HTTP错误
        
        # 解析JSON数据
        data = response.json()

        # 提取天气信息
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        humidity = current_condition['humidity']
        wind_speed = current_condition['windspeedKmph']

        # 格式化返回结果
        return f"{city}当前天气：{weather_desc}，气温 {temp_c}°C，湿度 {humidity}%，风速 {wind_speed} km/h"

    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API request failed: {e}")
        return f"错误：查询天气时遇到网络问题 - {str(e)}"
    except (KeyError, IndexError) as e:
        logger.error(f"Weather data parsing failed: {e}")
        return f"错误：解析天气数据失败，可能是城市名称无效 - {str(e)}"

def get_attractions(city: str, weather: Optional[str] = None) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    
    Args:
        city: 城市名称
        weather: 天气状况（可选，用于提供更精准的推荐）
        
    Returns:
        格式化的景点推荐信息
    """
    logger.info(f"Searching attractions for city: {city}, weather: {weather}")
    
    # 获取API密钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置TAVILY_API_KEY环境变量"

    try:
        # 延迟导入，避免不必要的依赖
        from tavily import TavilyClient
        
        # 初始化客户端
        tavily = TavilyClient(api_key=api_key)

        # 构造查询
        if weather:
            query = f"{city}在{weather}天气下最值得去的旅游景点推荐及理由"
        else:
            query = f"{city}最值得去的旅游景点推荐及理由"

        # 调用API
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # 处理响应
        if response.get("answer"):
            return response["answer"]

        # 格式化原始结果
        formatted_results = []
        for result in response.get("results", [])[:3]:  # 最多返回3个结果
            formatted_results.append(f"- **{result['title']}**: {result['content'][:150]}...")

        if not formatted_results:
            return f"抱歉，未找到{city}的相关景点推荐"

        return f"根据搜索，为您推荐{city}的以下景点：\n" + "\n".join(formatted_results)

    except ImportError:
        logger.error("Tavily module not found")
        return "错误：未安装tavily模块，请运行 'pip install tavily-python'"
    except Exception as e:
        logger.error(f"Tavily search failed: {e}", exc_info=True)
        return f"错误：搜索景点时出现问题 - {str(e)}"
