from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.filesystem import FilesystemBackend
from util import model
from langgraph.types import Command

"""
技能创建器代理

该代理使用 `/Users/lijunyi/road/llm/agents/skills/` 目录中的技能，
专门用于创建和管理其他技能。
"""


def create_skill_creator_agent():
    """
    创建技能创建器代理
    
    Returns:
        agent: 配置好的技能创建器代理
    """
    # 定义系统提示
    system_prompt = """
    你是一个专业的技能创造者，负责创建、管理和优化 Agent 技能。
    
    你的职责包括：
    1. 分析用户需求并设计合适的技能
    2. 创建符合标准格式的技能文件
    3. 验证技能的有效性和功能性
    4. 优化技能的性能和可靠性
    
    请根据用户的请求，利用可用的工具和资源，提供专业的技能创建服务。
    """

    # 创建技能创建器代理
    checkpointer = MemorySaver()

    agent = create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir="/Users/lijunyi/road/llm/agents"),
        skills=["/Users/lijunyi/road/llm/agents/skills/"],
        # interrupt_on={
        #     "write_file": True,  # Default: approve, edit, reject
        #     "read_file": False,  # No interrupts needed
        #     "edit_file": True  # Default: approve, edit, reject
        # },
        checkpointer=checkpointer,  # Required!
    )
    return agent


if __name__ == "__main__":
    # 创建技能创建器代理
    skill_creator_agent = create_skill_creator_agent()
    config = {"configurable": {"thread_id": "12345"}}
    # 测试代理
    print("=== 测试技能创建器代理 ===")

    # 测试创建简单技能的请求
    test_request = "请创建一个简单的web搜索技能，能够使用Tavily 客户端进行网络搜索。"

    print(f"发送请求: {test_request}")

    try:
        # 流式获取结果
        result = skill_creator_agent.stream(
            {"messages": [{"role": "user", "content": test_request}]},
            config=config
        )

        print("\n代理响应:")
        for chunk in result:
            print(chunk, end="\n", flush=True)
        print()
        #
        # # Check if execution was interrupted
        # if result.get("__interrupt__"):
        #     # Extract interrupt information
        #     interrupts = result["__interrupt__"][0].value
        #     action_requests = interrupts["action_requests"]
        #     review_configs = interrupts["review_configs"]
        #
        #     # Create a lookup map from tool name to review config
        #     config_map = {cfg["action_name"]: cfg for cfg in review_configs}
        #
        #     # Display the pending actions to the user
        #     for action in action_requests:
        #         review_config = config_map[action["name"]]
        #         print(f"Tool: {action['name']}")
        #         print(f"Arguments: {action['args']}")
        #         print(f"Allowed decisions: {review_config['allowed_decisions']}")
        #
        #     # Get user decisions (one per action_request, in order)
        #     decisions = [
        #         {"type": "approve"}  # User approved the deletion
        #     ]
        #
        #     # Resume execution with decisions
        #     result = skill_creator_agent.invoke(
        #         Command(resume={"decisions": decisions}),
        #         config=config  # Must use the same config!
        #     )
        #
        # # Process final result
        # print(result["messages"][-1].content)

    except Exception as e:
        print(f"测试失败: {e}")
