import os
import requests
import json
from dotenv import load_dotenv

def search(query, max_results=5):
    """
    使用Tavily API进行网络搜索
    
    参数:
    - query: 搜索关键词
    - max_results: 最大返回结果数，默认为5
    
    返回:
    - 搜索结果列表，每个结果包含title, url, content等字段
    """
    # Load environment variables from .env file
    load_dotenv()
    # 获取API密钥
    api_key = os.getenv('TAVILY_API_KEY')
    if not api_key:
        raise ValueError("请设置TAVILY_API_KEY环境变量")
    
    # 构建请求
    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": max_results
    }
    
    # 发送请求
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        results = response.json()
        return results.get('results', [])
    else:
        raise Exception(f"搜索失败: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # 测试搜索功能
    try:
        results = search("人工智能最新进展", max_results=3)
        for i, result in enumerate(results, 1):
            print(f"结果 {i}:")
            print(f"标题: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"摘要: {result['content'][:150]}...")
            print("-" * 50)
    except Exception as e:
        print(f"错误: {e}")