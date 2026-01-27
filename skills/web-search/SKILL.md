---
name: web-search
description: 使用Tavily客户端进行网络搜索的技能。适用于需要从互联网获取最新信息、研究特定主题或查找相关资料的场景。
---

# Web Search Skill

这个技能允许使用Tavily API进行网络搜索，获取最新的互联网信息。

## 使用方法

1. 确保已设置TAVILY_API_KEY环境变量
2. 调用scripts/search.py脚本执行搜索
3. 处理返回的搜索结果

## 脚本说明

主要脚本位于`scripts/search.py`，提供以下功能：
- 执行关键词搜索
- 限制返回结果数量
- 获取相关网站链接和摘要

## 示例

```python
from scripts.search import search

results = search("人工智能最新进展", max_results=5)
for result in results:
    print(f"标题: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"摘要: {result['content'][:100]}...")
```