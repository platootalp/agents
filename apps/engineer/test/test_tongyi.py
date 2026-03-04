import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

def test_tongyi_chat():
    load_dotenv()
    chatLLM = ChatTongyi(model="qwen-turbo", api_key=os.environ.get("DASHSCOPE_API_KEY"))

    messages = [
        SystemMessage(content="你是一个从中文到英文的翻译专家"),
        HumanMessage(content="请将这句话翻译成英文：今天路上的雪很大，我差点没看见前面的车"),
    ]

    print(chatLLM.invoke(messages))

    # 输出
    AIMessage(content="The snow on the road was very heavy today, and I almost didn't see the car in front of me.",
              additional_kwargs={}, response_metadata={'model_name': 'qwen-turbo', 'finish_reason': 'stop',
                                                       'request_id': '54bbbc9a-a655-9b97-a1fd-cff09d1a149e',
                                                       'token_usage': {'input_tokens': 42, 'output_tokens': 23,
                                                                       'prompt_tokens_details': {'cached_tokens': 0},
                                                                       'total_tokens': 65}},
              id='run-5491a259-9d2d-419f-981e-da3987038264-0')
