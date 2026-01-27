from typing import Literal
from deepagents import create_deep_agent
from util import tavily_client, model

def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

if __name__ == '__main__':
    agent = create_deep_agent(
        model=model,
        tools=[internet_search]
    )
    result = agent.stream({"messages": [{"role": "user", "content": "What is langgraph?"}]})
    for e in result:
        print(e)