from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.filesystem import FilesystemBackend
from util import model
from langgraph.types import Command
import uuid

"""
技能创建器代理

该代理使用 `/Users/lijunyi/road/llm/agents/skills/skill-creator` 目录中的技能，
专门用于创建和管理其他技能。

这个版本实现了完整的 Human-in-the-loop 功能，允许在执行敏感操作前
（如写入文件、编辑文件）征求用户批准。
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
        system_prompt=system_prompt,
        backend=FilesystemBackend(root_dir="/Users/lijunyi/road/llm/agents"),
        skills=["/Users/lijunyi/road/llm/agents/skills/skill-creator"],
        # 配置需要人工审批的工具
        interrupt_on={
            "write_file": {
                "allowed_decisions": ["approve", "edit", "reject"]
            },  # 写入文件：允许批准、编辑或拒绝
            "edit_file": {
                "allowed_decisions": ["approve", "edit", "reject"]
            },  # 编辑文件：允许批准、编辑或拒绝
            "read_file": False,  # 读取文件：不需要审批
            "ls": False,  # 列出文件：不需要审批
        },
        checkpointer=checkpointer,  # Human-in-the-loop 必需！
    )
    return agent


def get_user_decision(action, review_config):
    """
    获取用户对工具调用的决策
    
    Args:
        action: 工具调用信息，包含 name 和 args
        review_config: 审批配置，包含 allowed_decisions
        
    Returns:
        decision: 用户决策字典
    """
    print("\n" + "=" * 60)
    print("🔔 需要您的审批")
    print("=" * 60)
    print(f"工具名称: {action['name']}")
    print(f"工具参数:")
    for key, value in action['args'].items():
        # 如果值太长，截断显示
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}... (共 {len(value)} 字符)")
        else:
            print(f"  {key}: {value}")
    print(f"允许的操作: {', '.join(review_config['allowed_decisions'])}")
    print("=" * 60)

    allowed = review_config['allowed_decisions']

    while True:
        print("\n请选择操作:")
        if "approve" in allowed:
            print("  [a] 批准 (approve) - 执行此操作")
        if "edit" in allowed:
            print("  [e] 编辑 (edit) - 修改参数后执行")
        if "reject" in allowed:
            print("  [r] 拒绝 (reject) - 跳过此操作")

        choice = input("\n您的选择: ").lower().strip()

        if choice == 'a' and "approve" in allowed:
            return {"type": "approve"}

        elif choice == 'e' and "edit" in allowed:
            print("\n请编辑参数 (JSON 格式，或输入 'cancel' 取消编辑):")
            print(f"原始参数: {action['args']}")

            # 简化版：让用户确认是否要编辑每个参数
            edited_args = {}
            for key, value in action['args'].items():
                edit_choice = input(f"\n是否编辑 '{key}' (当前值: {value})? [y/n]: ").lower()
                if edit_choice == 'y':
                    new_value = input(f"请输入新值: ")
                    # 尝试保持原始类型
                    if isinstance(value, bool):
                        edited_args[key] = new_value.lower() in ['true', 'yes', '1']
                    elif isinstance(value, int):
                        try:
                            edited_args[key] = int(new_value)
                        except:
                            edited_args[key] = new_value
                    else:
                        edited_args[key] = new_value
                else:
                    edited_args[key] = value

            return {
                "type": "edit",
                "edited_action": {
                    "name": action["name"],
                    "args": edited_args
                }
            }

        elif choice == 'r' and "reject" in allowed:
            confirm = input("确认拒绝此操作? [y/n]: ").lower()
            if confirm == 'y':
                return {"type": "reject"}

        else:
            print("❌ 无效的选择，请重试")


def _has_interrupt(state):
    """
    检查状态是否包含中断
    
    Args:
        state: 代理状态（可以是 dict 或 StateSnapshot）
        
    Returns:
        bool: 是否存在中断
    """
    # 处理 StateSnapshot 对象
    if hasattr(state, 'next'):
        # 检查 next 是否非空（存在待执行的节点通常意味着有中断）
        if state.next:
            return True

    if hasattr(state, 'tasks'):
        # 检查 tasks 中是否有中断
        if state.tasks:
            for task in state.tasks:
                if hasattr(task, 'interrupts') and task.interrupts:
                    return True

    if hasattr(state, 'values'):
        if "__interrupt__" in state.values and state.values["__interrupt__"]:
            return True

    # 处理普通字典
    if isinstance(state, dict):
        if "__interrupt__" in state and state["__interrupt__"]:
            return True

    return False


def _extract_interrupt_info(state):
    """
    从状态中提取中断信息
    
    Args:
        state: 代理状态（可以是 dict 或 StateSnapshot）
        
    Returns:
        tuple: (action_requests, review_configs) 或 (None, None)
    """
    interrupt_data = None

    # 方法1：从 StateSnapshot.tasks 中提取
    if hasattr(state, 'tasks') and state.tasks:
        for task in state.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                # tasks[].interrupts 是一个列表
                for interrupt in task.interrupts:
                    if hasattr(interrupt, 'value'):
                        interrupt_value = interrupt.value
                        if isinstance(interrupt_value, dict):
                            action_requests = interrupt_value.get("action_requests")
                            review_configs = interrupt_value.get("review_configs")
                            if action_requests and review_configs:
                                return action_requests, review_configs

    # 方法2：从 StateSnapshot.values 中提取
    if hasattr(state, 'values'):
        interrupt_data = state.values.get("__interrupt__")
    # 方法3：从普通字典中提取
    elif isinstance(state, dict):
        interrupt_data = state.get("__interrupt__")

    if interrupt_data and len(interrupt_data) > 0:
        interrupts = interrupt_data[0].value
        return interrupts["action_requests"], interrupts["review_configs"]

    return None, None


def run_agent_with_hitl(agent, user_message, config=None, stream_mode="updates"):
    """
    运行代理并处理 Human-in-the-loop 交互（使用 stream 模式实时输出）
    
    Args:
        agent: 代理实例
        user_message: 用户消息
        config: 配置字典（包含 thread_id）
        stream_mode: stream 模式，可选 "values", "updates", "messages" 等
        
    Returns:
        最终结果
    """
    if config is None:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print(f"\n💬 用户请求: {user_message}\n")
    print("🔄 开始流式处理...\n")

    # 使用 stream 模式首次调用代理
    has_interrupt_in_stream = False
    for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
            stream_mode=stream_mode
    ):
        # 显示流式输出
        _display_stream_chunk(chunk, stream_mode)

        # 在 stream 过程中检测中断
        if stream_mode == "updates" and isinstance(chunk, dict):
            if "__interrupt__" in chunk:
                has_interrupt_in_stream = True
                print("\n⚠️  检测到中断信号（在stream中）...\n")

    # 获取当前完整状态（更可靠的方式）
    current_state = agent.get_state(config)

    # 调试信息：显示状态的完整信息
    print(f"\n🔍 [调试] has_interrupt_in_stream: {has_interrupt_in_stream}")
    print(f"🔍 [调试] current_state type: {type(current_state)}")

    if hasattr(current_state, 'next'):
        print(f"🔍 [调试] current_state.next: {current_state.next}")

    if hasattr(current_state, 'tasks'):
        print(f"🔍 [调试] current_state.tasks 数量: {len(current_state.tasks) if current_state.tasks else 0}")
        if current_state.tasks:
            for idx, task in enumerate(current_state.tasks):
                print(f"🔍 [调试] task[{idx}] id: {task.id if hasattr(task, 'id') else 'N/A'}")
                if hasattr(task, 'interrupts'):
                    print(f"🔍 [调试] task[{idx}] interrupts: {len(task.interrupts) if task.interrupts else 0} 个")

    if hasattr(current_state, 'values'):
        print(f"🔍 [调试] 状态 values 键: {list(current_state.values.keys())}")
    elif isinstance(current_state, dict):
        print(f"🔍 [调试] 状态 dict 键: {list(current_state.keys())}")

    print(f"🔍 [调试] _has_interrupt(current_state): {_has_interrupt(current_state)}")

    # 循环处理中断，直到没有中断为止
    iteration = 0
    while _has_interrupt(current_state) or has_interrupt_in_stream:
        iteration += 1
        if iteration > 10:  # 防止无限循环
            print("⚠️  警告：中断处理循环次数过多，退出")
            break

        print("\n⏸️  代理执行被暂停，等待人工审批...\n")

        # 提取中断信息（使用辅助函数，避免硬编码）
        action_requests, review_configs = _extract_interrupt_info(current_state)

        if not action_requests or not review_configs:
            print("⚠️  警告：检测到中断但无法提取中断信息")
            print(f"    action_requests: {action_requests}")
            print(f"    review_configs: {review_configs}")
            break

        # 创建工具名称到审批配置的映射
        config_map = {cfg["action_name"]: cfg for cfg in review_configs}

        # 收集所有工具调用的用户决策
        decisions = []

        if len(action_requests) > 1:
            print(f"\n📋 检测到 {len(action_requests)} 个需要审批的操作\n")

        for idx, action in enumerate(action_requests, 1):
            if len(action_requests) > 1:
                print(f"\n--- 操作 {idx}/{len(action_requests)} ---")

            review_config = config_map[action["name"]]
            decision = get_user_decision(action, review_config)
            decisions.append(decision)

            # 显示用户的选择
            if decision["type"] == "approve":
                print("✅ 已批准")
            elif decision["type"] == "edit":
                print("✏️  已编辑")
            elif decision["type"] == "reject":
                print("❌ 已拒绝")

        # 使用决策恢复执行（也使用 stream 模式）
        print("\n▶️  恢复代理执行...\n")
        has_interrupt_in_stream = False
        for chunk in agent.stream(
                Command(resume={"decisions": decisions}),
                config=config,  # 必须使用相同的 config！
                stream_mode=stream_mode
        ):
            # 显示流式输出
            _display_stream_chunk(chunk, stream_mode)

            # 在 stream 过程中检测中断
            if stream_mode == "updates" and isinstance(chunk, dict):
                if "__interrupt__" in chunk:
                    has_interrupt_in_stream = True
                    print("\n⚠️  检测到中断信号...\n")

        # 重新获取当前状态以检查是否还有中断
        current_state = agent.get_state(config)

    # 返回最终结果
    print("\n" + "=" * 60)
    print("✅ 任务完成")
    print("=" * 60)

    # 获取最终状态以显示消息
    final_state = agent.get_state(config)
    if hasattr(final_state, 'values'):
        messages = final_state.values.get("messages", [])
    else:
        messages = final_state.get("messages", [])

    if messages and len(messages) > 0:
        final_message = messages[-1].content
        print(f"\n📝 代理最终响应:\n{final_message}\n")

    return current_state


def _display_stream_chunk(chunk, stream_mode):
    """
    显示 stream 输出的 chunk
    
    Args:
        chunk: stream 输出的数据块
        stream_mode: stream 模式
    """
    if stream_mode == "updates":
        # updates 模式：显示每个节点的更新
        for node_name, node_update in chunk.items():
            print(f"📍 节点: {node_name}")

            # 检查 node_update 是否为 None
            if node_update is None:
                print("  (无更新)")
                print()
                continue

            # 检查 node_update 是否是字典
            if not isinstance(node_update, dict):
                print(f"  更新: {node_update}")
                print()
                continue

            # 如果有消息更新
            if "messages" in node_update:
                messages = node_update["messages"]
                if not isinstance(messages, list):
                    messages = [messages]

                for msg in messages:
                    if hasattr(msg, 'content') and msg.content:
                        # 显示消息内容（限制长度）
                        content = str(msg.content)
                        if len(content) > 200:
                            print(f"  💭 {content[:200]}...")
                        else:
                            print(f"  💭 {content}")

                    # 如果有工具调用
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            print(f"  🔧 调用工具: {tool_call.get('name', 'unknown')}")
                            if 'args' in tool_call:
                                print(f"     参数: {tool_call['args']}")

            # 如果有其他更新
            for key, value in node_update.items():
                if key != "messages":
                    # 限制值的显示长度
                    value_str = str(value)
                    if len(value_str) > 200:
                        print(f"  {key}: {value_str[:200]}...")
                    else:
                        print(f"  {key}: {value}")

            print()  # 空行分隔

    elif stream_mode == "values":
        # values 模式：显示完整状态
        if isinstance(chunk, dict):
            print(f"📊 状态更新: {list(chunk.keys())}")
        else:
            print(f"📊 状态更新: {chunk}")
        print()

    elif stream_mode == "messages":
        # messages 模式：只显示消息
        if isinstance(chunk, tuple):
            message, metadata = chunk
            if hasattr(message, 'content') and message.content:
                print(f"💬 {message.content}")
                print()


if __name__ == "__main__":
    # 创建技能创建器代理
    print("🚀 初始化技能创建器代理...")
    skill_creator_agent = create_skill_creator_agent()

    print("\n" + "=" * 60)
    print("技能创建器代理 - Human-in-the-loop 演示")
    print("=" * 60)
    print("\n此代理将在执行敏感操作（如写入/编辑文件）前请求您的批准。")
    print("您可以选择批准、编辑参数或拒绝操作。\n")

    # 测试请求
    test_request = "请创建一个web搜索技能，能够使用Tavily客户端进行网络搜索。"

    try:
        result = run_agent_with_hitl(
            agent=skill_creator_agent,
            user_message=test_request
        )

        print("\n✨ 演示完成！")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
