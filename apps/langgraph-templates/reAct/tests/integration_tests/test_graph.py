import os

import pytest
from langsmith import unit

from dotenv import load_dotenv

from reAct.src.react_agent import graph


@pytest.mark.asyncio
# @unit
async def test_react_agent_simple_passthrough() -> None:
    load_dotenv()
    print(f"QWEN_API_KEY: {os.environ.get('HF_API_KEY')}")
    print(f"LANGCHAIN_API_KEY:{os.environ.get('LANGCHAIN_API_KEY')}")
    # llm = HuggingFaceEndpoint(
    #     repo_id="deepseek-ai/DeepSeek-V3-0324",
    #     task="text-generation",
    #     max_new_tokens=1024,
    #     do_sample=False,
    #     repetition_penalty=1.03,
    #     huggingfacehub_api_token=os.environ.get("HF_API_KEY")
    # )

    res = await graph.ainvoke(
        {"messages": [("user", "Who is the founder of LangChain?")]},
        {"configurable": {"system_prompt": "You are a helpful AI assistant.",
                          "model": "qwen-turbo",
                          "api_key": os.environ.get("QWEN_API_KEY")}},
    )

    assert "harrison" in str(res["messages"][-1].content).lower()
