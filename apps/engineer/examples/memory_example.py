"""
记忆系统使用示例
演示如何使用短期记忆和长期记忆
"""

# genAI_main_start
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engineer.core.memory import (
    # 基础类
    Message, MemoryVariables,
    # 聊天历史
    InMemoryChatMessageHistory, FileChatMessageHistory, SessionChatMessageHistory,
    get_chat_history,
    # 短期记忆
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory,
    # 长期记忆
    VectorStoreMemory,
    ConversationEntityMemory,
    # 管理器
    CombinedMemory,
    MemoryManager,
    create_memory,
    list_memory_types
)


def example_1_buffer_memory():
    """示例1: 基础缓冲区记忆"""
    print("\n" + "="*60)
    print("示例1: 基础缓冲区记忆 (ConversationBufferMemory)")
    print("="*60)
    
    # 创建缓冲区记忆
    memory = ConversationBufferMemory(verbose=True)
    
    # 模拟对话
    conversations = [
        ("你好，我是小明", "你好小明！很高兴认识你。"),
        ("我今年25岁", "25岁是一个很好的年龄！"),
        ("我喜欢编程", "编程是一个很有趣的技能！你主要用什么语言？"),
    ]
    
    for user_input, ai_output in conversations:
        memory.save_context(
            {"input": user_input},
            {"output": ai_output}
        )
    
    # 加载记忆变量
    mem_vars = memory.load_memory_variables({})
    print(f"\n📝 对话历史:\n{mem_vars.history}")
    print(f"\n📊 消息数量: {len(memory.buffer)}")


def example_2_window_memory():
    """示例2: 滑动窗口记忆"""
    print("\n" + "="*60)
    print("示例2: 滑动窗口记忆 (ConversationBufferWindowMemory)")
    print("="*60)
    
    # 创建滑动窗口记忆（只保留最近2轮对话）
    memory = ConversationBufferWindowMemory(k=2, verbose=True)
    
    # 模拟多轮对话
    conversations = [
        ("第一个问题", "第一个回答"),
        ("第二个问题", "第二个回答"),
        ("第三个问题", "第三个回答"),
        ("第四个问题", "第四个回答"),
    ]
    
    for i, (user_input, ai_output) in enumerate(conversations, 1):
        print(f"\n--- 第 {i} 轮对话 ---")
        memory.save_context(
            {"input": user_input},
            {"output": ai_output}
        )
        
        # 查看当前窗口内的消息
        mem_vars = memory.load_memory_variables({})
        print(f"窗口内消息数: {len(memory.buffer)}")
    
    print(f"\n📝 最终窗口内容:\n{mem_vars.history}")


def example_3_token_buffer_memory():
    """示例3: Token限制记忆"""
    print("\n" + "="*60)
    print("示例3: Token限制记忆 (ConversationTokenBufferMemory)")
    print("="*60)
    
    # 创建Token限制记忆（限制100个token）
    memory = ConversationTokenBufferMemory(max_token_limit=100, verbose=True)
    
    # 模拟对话
    conversations = [
        ("请介绍一下Python语言", "Python是一种高级编程语言，具有简洁易读的语法。"),
        ("Python有哪些应用场景？", "Python广泛应用于Web开发、数据科学、机器学习等领域。"),
        ("如何学习Python？", "建议从基础语法开始，然后通过项目实践提升技能。"),
    ]
    
    for user_input, ai_output in conversations:
        memory.save_context(
            {"input": user_input},
            {"output": ai_output}
        )
    
    print(f"\n📊 当前Token使用: {memory.current_token_count}/{memory.max_token_limit}")
    print(f"📝 缓冲区内容:\n{memory.buffer_as_str}")


def example_4_summary_memory():
    """示例4: 摘要记忆"""
    print("\n" + "="*60)
    print("示例4: 摘要记忆 (ConversationSummaryMemory)")
    print("="*60)
    
    # 创建摘要记忆（不使用LLM，使用简单截断）
    memory = ConversationSummaryMemory(
        max_buffer_size=4,  # 4条消息后生成摘要
        verbose=True
    )
    
    # 模拟对话
    conversations = [
        ("你好", "你好！"),
        ("今天天气怎么样？", "今天阳光明媚，非常适合外出。"),
        ("有什么推荐的活动吗？", "可以去公园散步或者骑自行车。"),
        ("好的，谢谢建议", "不客气！祝你有愉快的一天。"),
    ]
    
    for user_input, ai_output in conversations:
        memory.save_context(
            {"input": user_input},
            {"output": ai_output}
        )
    
    mem_vars = memory.load_memory_variables({})
    print(f"\n📝 摘要: {memory.summary[:200] if memory.summary else '无'}...")
    print(f"📝 待处理消息: {len(memory.buffer)}")


def example_5_vector_memory():
    """示例5: 向量存储记忆（长期记忆）"""
    print("\n" + "="*60)
    print("示例5: 向量存储记忆 (VectorStoreMemory)")
    print("="*60)
    
    # 创建向量存储记忆
    memory = VectorStoreMemory(
        retrieval_k=2,
        verbose=True
    )
    
    # 添加一些记忆
    memory.add_memory("小明喜欢编程，尤其是Python和JavaScript")
    memory.add_memory("小红擅长设计，主要使用Figma工具")
    memory.add_memory("项目A的截止日期是下周五")
    memory.add_memory("团队每周三下午有例会")
    
    print(f"\n📊 存储的记忆数量: {len(memory.vector_store)}")
    
    # 检索相关记忆
    print("\n🔍 查询: '谁会编程？'")
    mem_vars = memory.load_memory_variables({"input": "谁会编程？"})
    print(f"相关上下文:\n{mem_vars.context}")
    
    print("\n🔍 查询: '什么时候开会？'")
    mem_vars = memory.load_memory_variables({"input": "什么时候开会？"})
    print(f"相关上下文:\n{mem_vars.context}")


def example_6_entity_memory():
    """示例6: 实体记忆（长期记忆）"""
    print("\n" + "="*60)
    print("示例6: 实体记忆 (ConversationEntityMemory)")
    print("="*60)
    
    # 创建实体记忆
    memory = ConversationEntityMemory(verbose=True)
    
    # 手动添加一些实体
    memory.add_entity("小明", "25岁的程序员，擅长Python")
    memory.add_entity("小红", "UI设计师，喜欢用Figma")
    memory.add_entity("Python", "一种流行的编程语言")
    
    # 模拟对话
    memory.save_context(
        {"input": "小明最近在学什么？"},
        {"output": "小明最近在学习机器学习和深度学习。"}
    )
    
    # 加载记忆变量
    mem_vars = memory.load_memory_variables({"input": "小明"})
    print(f"\n📝 实体上下文:\n{mem_vars.context}")
    print(f"\n📊 已知实体: {list(memory.entity_store.entities.keys())}")


def example_7_combined_memory():
    """示例7: 组合记忆"""
    print("\n" + "="*60)
    print("示例7: 组合记忆 (CombinedMemory)")
    print("="*60)
    
    # 创建多个记忆
    buffer_memory = ConversationBufferWindowMemory(k=3)
    vector_memory = VectorStoreMemory(retrieval_k=2)
    entity_memory = ConversationEntityMemory()
    
    # 组合记忆
    combined = CombinedMemory(
        memories=[buffer_memory, vector_memory, entity_memory],
        verbose=True
    )
    
    # 添加一些实体
    entity_memory.add_entity("项目Alpha", "一个重要的客户项目")
    
    # 添加一些向量记忆
    vector_memory.add_memory("项目Alpha的预算是100万")
    vector_memory.add_memory("项目Beta已经完成")
    
    # 模拟对话
    combined.save_context(
        {"input": "项目Alpha进展如何？"},
        {"output": "项目Alpha正在按计划进行。"}
    )
    
    # 加载组合记忆
    mem_vars = combined.load_memory_variables({"input": "项目Alpha"})
    print(f"\n📝 组合历史:\n{mem_vars.history}")
    print(f"\n📝 相关上下文:\n{mem_vars.context}")


def example_8_memory_manager():
    """示例8: 记忆管理器"""
    print("\n" + "="*60)
    print("示例8: 记忆管理器 (MemoryManager)")
    print("="*60)
    
    # 创建记忆管理器
    manager = MemoryManager(verbose=True)
    
    # 使用管理器创建不同类型的记忆
    buffer = manager.create_memory("buffer", name="main_buffer")
    window = manager.create_memory("buffer_window", name="recent", k=3)
    vector = manager.create_memory("vector", name="long_term", retrieval_k=2)
    
    print(f"\n📊 已创建的记忆: {manager.list_memories()}")
    
    # 使用快捷函数
    print("\n🔧 使用 create_memory() 快捷函数:")
    quick_memory = create_memory("buffer_window", k=5)
    print(f"创建的记忆: {quick_memory}")
    
    # 打印可用的记忆类型
    print(list_memory_types())


def example_9_chat_history():
    """示例9: 聊天消息历史"""
    print("\n" + "="*60)
    print("示例9: 聊天消息历史")
    print("="*60)
    
    # 内存历史
    print("\n📦 InMemoryChatMessageHistory:")
    memory_history = InMemoryChatMessageHistory()
    memory_history.add_user_message("你好")
    memory_history.add_ai_message("你好！有什么可以帮助你的？")
    print(f"消息数量: {len(memory_history)}")
    for msg in memory_history:
        print(f"  {msg.role}: {msg.content}")
    
    # 会话历史
    print("\n📦 SessionChatMessageHistory:")
    session1 = SessionChatMessageHistory("user_001")
    session1.add_user_message("我是用户1")
    
    session2 = SessionChatMessageHistory("user_002")
    session2.add_user_message("我是用户2")
    
    print(f"所有会话: {SessionChatMessageHistory.get_all_sessions()}")
    
    # 使用工厂函数
    print("\n📦 使用 get_chat_history() 工厂函数:")
    history = get_chat_history("memory")
    history.add_user_message("通过工厂函数创建")
    print(f"创建的历史: {history}")


def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("记忆系统使用示例")
    print("="*70)
    
    example_1_buffer_memory()
    example_2_window_memory()
    example_3_token_buffer_memory()
    example_4_summary_memory()
    example_5_vector_memory()
    example_6_entity_memory()
    example_7_combined_memory()
    example_8_memory_manager()
    example_9_chat_history()
    
    print("\n" + "="*70)
    print("所有示例运行完成！")
    print("="*70)


if __name__ == "__main__":
    main()
# genAI_main_end
