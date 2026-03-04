import os

from dotenv import load_dotenv
from react_agent.configuration import Configuration


def test_google_models()-> None:
    load_dotenv()
    print(f"QWEN_API_KEY: {os.environ.get('HF_API_KEY')}")
    print(f"LANGCHAIN_API_KEY:{os.environ.get('LANGCHAIN_API_KEY')}")


def test_configuration_empty() -> None:
    Configuration.from_runnable_config({})
