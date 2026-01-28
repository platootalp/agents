from dataclasses import dataclass

from langchain.agents import create_react_agent


# @dataclass
class AttractionSearchAgent():
    ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。

   **工具调用格式:**
   `[TOOL_CALL:amap_maps_text_search:keywords=景点,city=城市名]`

   **示例:**
   - `[TOOL_CALL:amap_maps_text_search:keywords=景点,city=北京]`
   - `[TOOL_CALL:amap_maps_text_search:keywords=博物馆,city=上海]`

   **重要:**
   - 必须使用工具搜索,不要编造信息
   - 根据用户偏好({preferences})搜索{city}的景点
   """

    def __init__(self):
        pass


class WeatherQueryAgent():
    WEATHER_AGENT_PROMPT = """你是天气查询专家。

    **工具调用格式:**
    `[TOOL_CALL:amap_maps_weather:city=城市名]`

    请查询{city}的天气信息。
    """

    def __init__(self):
        pass


class HotelAgent():
    HOTEL_AGENT_PROMPT = """你是酒店推荐专家。

    **工具调用格式:**
    `[TOOL_CALL:amap_maps_text_search:keywords=酒店,city=城市名]`

    请搜索{city}的{accommodation}酒店。
    """

    def __init__(self):
        pass


class Planner():
    PLANNER_AGENT_PROMPT = """你是行程规划专家。

    **输出格式:**
    严格按照以下JSON格式返回:
    {
      "city": "城市名称",
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD",
      "days": [...],
      "weather_info": [...],
      "overall_suggestions": "总体建议",
      "budget": {...}
    }

    **规划要求:**
    1. weather_info必须包含每天的天气
    2. 温度为纯数字(不带°C)
    3. 每天安排2-3个景点
    4. 考虑景点距离和游览时间
    5. 包含早中晚三餐
    6. 提供实用建议
    7. 包含预算信息
    """

    def __init__(self):
        pass


if __name__ == '__main__':
    agent = create_react_agent()
