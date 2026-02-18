# LLM 核心模块重构文档

## 📋 概述

本文档记录了 `src/core` 模块的重构过程，该重构将项目的大模型基础能力与 LangChain 的命名规范和架构设计完全对齐，提升代码的标准化程度和可维护性。

## 🎯 重构目标

1. **命名规范对齐**：文件名、类名、方法名与 LangChain 保持一致
2. **架构设计对齐**：采用 LangChain 的设计模式和接口风格
3. **代码现代化**：使用 Pydantic、异步支持等现代特性
4. **功能增强**：添加异步方法、批量处理等高级功能

## 📁 文件结构变化

### 重构前

```
src/core/
├── base_llm.py              # 旧：基础 LLM 类
├── llm_factory.py           # 旧：LLM 工厂函数
├── model_info.py            # 模型信息
├── providers/               # 提供商实现
└── __init__.py              # 模块导出
```

### 重构后

```
src/core/
├── language_models.py       # 新：语言模型基础类 ✨
├── chat_model_factory.py    # 新：聊天模型工厂函数 ✨
├── model_info.py            # 保持不变
├── providers/               # 保持不变
└── __init__.py              # 完全更新，只导出新接口
```

## 🔄 主要变更

### 1. 文件名变更

| 旧文件名 | 新文件名 | 说明 |
|---------|---------|------|
| `base_llm.py` | `language_models.py` | 对齐 LangChain 命名 |
| `llm_factory.py` | `chat_model_factory.py` | 更明确的命名 |

**重要**：旧文件已完全移除，只保留新架构。

### 2. 类名变更

| 旧类名 | 新类名 | 说明 |
|--------|--------|------|
| `BaseLLM` | `BaseChatModel` | 更准确地描述聊天模型 |
| `ModelConfig` | `ChatModelConfig` | 使用 Pydantic BaseModel |
| `ModelResponse` | `ChatResult` | 对齐 LangChain 命名 |
| - | `LLMResult` | 新增批量调用结果类 |
| - | `FunctionMessage` | 新增函数消息类 |

**重要**：旧类名已完全移除，请使用新类名。

### 3. 方法名变更

#### BaseChatModel（原 BaseLLM）

| 旧方法名 | 新方法名 | 说明 |
|---------|---------|------|
| `chat()` | `invoke()` | 对齐 LangChain 标准接口 |
| `complete()` | `predict()` | 向后兼容方法 |
| `stream_chat()` | `stream()` | 简化名称 |
| - | `batch()` | 新增批量调用 |
| - | `ainvoke()` | 新增异步调用 |
| - | `astream()` | 新增异步流式调用 |
| - | `abatch()` | 新增异步批量调用 |

### 4. 工厂函数变更

| 旧函数名 | 新函数名 | 说明 |
|---------|---------|------|
| `create_llm()` | `create_chat_model()` | 更明确的命名 |
| `create_model()` | `init_chat_model()` | 对齐 LangChain |
| - | `ChatOpenAI()` | 新增类型特定工厂 |
| - | `ChatAnthropic()` | 新增类型特定工厂 |
| - | `ChatOllama()` | 新增类型特定工厂 |
| `quick_chat()` | `quick_invoke()` | 对齐命名规范 |
| - | `quick_batch()` | 新增批量调用 |

## 🌟 新增功能

### 1. 异步支持

```python
# 异步调用
response = await model.ainvoke("你好")

# 异步流式调用
async for chunk in model.astream("讲个笑话"):
    print(chunk, end='', flush=True)

# 异步批量调用
responses = await model.abatch(["你好", "再见", "谢谢"])
```

### 2. 批量处理

```python
# 同步批量调用
responses = model.batch(["问题1", "问题2", "问题3"])

# 便捷批量函数
responses = quick_batch(
    ["问题1", "问题2"],
    model="gpt-4"
)
```

### 3. 统一的 invoke 接口

```python
# 支持字符串
response = model.invoke("你好")

# 支持消息列表
response = model.invoke([
    SystemMessage(content="你是助手"),
    HumanMessage(content="你好")
])
```

### 4. LangChain 风格的工厂函数

```python
# 方式 1: init_chat_model（推荐）
model = init_chat_model("gpt-4")

# 方式 2: 类型特定工厂
model = ChatOpenAI(model_name="gpt-4")
model = ChatAnthropic(model_name="claude-3-opus")
model = ChatOllama(model_name="llama2")

# 方式 3: 快速调用
response = quick_invoke("你好", model="gpt-4")
```

## 📝 迁移指南

### 基本使用迁移

#### 旧代码（已废弃，不再支持）

```python
from src.core import create_model, BaseLLM  # ❌ 这些导入已不存在

# 创建模型
model = create_model("gpt-4")  # ❌ 此函数已移除

# 调用模型
response = model.chat([Message(role="user", content="你好")])  # ❌ chat() 方法已移除
print(response.content)
```

#### 新代码（必须使用）

```python
from src.core import init_chat_model, HumanMessage

# 创建模型
model = init_chat_model("gpt-4")

# 调用模型
response = model.invoke("你好")
# 或者
response = model.invoke([HumanMessage(content="你好")])
print(response.content)
```

### 流式调用迁移

#### 旧代码（已废弃）

```python
model = create_model("gpt-4")  # ❌ 已移除
for chunk in model.stream_chat([Message(role="user", content="讲个笑话")]):  # ❌ 已移除
    print(chunk, end='', flush=True)
```

#### 新代码（必须使用）

```python
model = init_chat_model("gpt-4")
for chunk in model.stream("讲个笑话"):
    print(chunk, end='', flush=True)
```

### 配置对象迁移

#### 旧代码（已废弃）

```python
from src.core import ModelConfig, ModelType, create_llm  # ❌ 这些已移除

config = ModelConfig(  # ❌ ModelConfig 已移除
    model_name="gpt-4",
    model_type=ModelType.OPENAI,
    temperature=0.7
)
model = create_llm(config)  # ❌ create_llm() 已移除
```

#### 新代码（必须使用）

```python
from src.core import ChatModelConfig, ModelType

# 使用配置对象
config = ChatModelConfig(
    model_name="gpt-4",
    model_type=ModelType.OPENAI,
    temperature=0.7
)

# 然后手动创建实例，或者更简单（推荐）：
from src.core import init_chat_model

model = init_chat_model(
    model="gpt-4",
    temperature=0.7
)
```

### 类型特定工厂迁移

#### 旧代码（已废弃）

```python
from src.core import create_model, ModelType  # ❌ create_model 已移除

model = create_model(  # ❌ 此函数已移除
    model_name="gpt-4",
    model_type=ModelType.OPENAI,
    api_key="sk-..."
)
```

#### 新代码（必须使用）

```python
from src.core import ChatOpenAI

# 更简洁、更符合 LangChain 风格
model = ChatOpenAI(
    model_name="gpt-4",
    api_key="sk-..."
)
```

## 🔧 完整示例

### 示例 1: 基础聊天

```python
from src.core import init_chat_model, HumanMessage, SystemMessage

# 初始化模型
model = init_chat_model(
    model="gpt-4",
    temperature=0.7,
    max_tokens=2000
)

# 简单调用
response = model.invoke("你好，请介绍一下你自己")
print(response.content)

# 多轮对话
messages = [
    SystemMessage(content="你是一个友好的助手"),
    HumanMessage(content="你好"),
]
response = model.invoke(messages)
print(response.content)
```

### 示例 2: 流式输出

```python
from src.core import ChatOpenAI

model = ChatOpenAI(model_name="gpt-4")

print("AI: ", end='', flush=True)
for chunk in model.stream("请写一首关于春天的诗"):
    print(chunk, end='', flush=True)
print()
```

### 示例 3: 批量处理

```python
from src.core import init_chat_model

model = init_chat_model("gpt-3.5-turbo")

questions = [
    "什么是人工智能？",
    "什么是机器学习？",
    "什么是深度学习？"
]

# 批量调用
responses = model.batch(questions)

for question, response in zip(questions, responses):
    print(f"Q: {question}")
    print(f"A: {response.content}\n")
```

### 示例 4: 异步调用

```python
import asyncio
from src.core import init_chat_model

async def main():
    model = init_chat_model("gpt-4")
    
    # 异步调用
    response = await model.ainvoke("你好")
    print(response.content)
    
    # 异步流式
    print("AI: ", end='', flush=True)
    async for chunk in model.astream("讲个笑话"):
        print(chunk, end='', flush=True)
    print()
    
    # 异步批量
    responses = await model.abatch([
        "你好",
        "再见",
        "谢谢"
    ])
    for resp in responses:
        print(resp.content)

asyncio.run(main())
```

### 示例 5: 使用类型特定工厂

```python
from src.core import ChatOpenAI, ChatAnthropic, ChatOllama

# OpenAI
openai_model = ChatOpenAI(
    model_name="gpt-4",
    api_key="sk-..."
)

# Anthropic
anthropic_model = ChatAnthropic(
    model_name="claude-3-opus-20240229",
    api_key="sk-ant-..."
)

# Ollama (本地)
ollama_model = ChatOllama(
    model_name="llama2",
    base_url="http://localhost:11434"
)

# 使用相同的接口
for model in [openai_model, anthropic_model, ollama_model]:
    response = model.invoke("你好")
    print(f"{model}: {response.content}")
```

## ⚠️ 重大变更（Breaking Changes）

**v3.0.0 版本已完全移除以下旧接口，无向后兼容：**

| 旧接口 | 新接口 | 状态 |
|--------|--------|------|
| `BaseLLM` | `BaseChatModel` | ❌ 已移除 |
| `ModelConfig` | `ChatModelConfig` | ❌ 已移除 |
| `ModelResponse` | `ChatResult` | ❌ 已移除 |
| `Message` | `BaseMessage` 及其子类 | ❌ 已移除 |
| `create_llm()` | `init_chat_model()` | ❌ 已移除 |
| `create_model()` | `init_chat_model()` | ❌ 已移除 |
| `quick_chat()` | `quick_invoke()` | ❌ 已移除 |
| `quick_chat_stream()` | `model.stream()` | ❌ 已移除 |
| `chat()` 方法 | `invoke()` 方法 | ❌ 已移除 |
| `stream_chat()` 方法 | `stream()` 方法 | ❌ 已移除 |

**必须更新代码以使用新接口。**

## 📊 对比总结

| 方面 | 重构前 | 重构后 |
|------|--------|--------|
| 文件命名 | `base_llm.py` | `language_models.py` |
| 基类名称 | `BaseLLM` | `BaseChatModel` |
| 主要方法 | `chat()` | `invoke()` |
| 流式方法 | `stream_chat()` | `stream()` |
| 异步支持 | ❌ 无 | ✅ `ainvoke()`, `astream()`, `abatch()` |
| 批量处理 | ❌ 无 | ✅ `batch()` |
| 工厂函数 | `create_model()` | `init_chat_model()` |
| 类型工厂 | ❌ 无 | ✅ `ChatOpenAI()`, `ChatAnthropic()` 等 |
| 配置类 | `ModelConfig` (dataclass) | `ChatModelConfig` (Pydantic) |
| 响应类 | `ModelResponse` | `ChatResult` |
| 向后兼容 | N/A | ✅ 保留所有旧接口 |

## 🎯 最佳实践

### 1. 推荐使用新接口

```python
# ✅ 推荐
from src.core import init_chat_model
model = init_chat_model("gpt-4")
response = model.invoke("你好")

# ❌ 不推荐（虽然仍可用）
from src.core import create_model
model = create_model("gpt-4")
response = model.chat([Message(role="user", content="你好")])
```

### 2. 使用类型特定工厂

```python
# ✅ 更清晰
from src.core import ChatOpenAI
model = ChatOpenAI(model_name="gpt-4")

# ✅ 也可以
from src.core import init_chat_model
model = init_chat_model("gpt-4", model_provider="openai")
```

### 3. 利用新功能

```python
# 批量处理
responses = model.batch(["问题1", "问题2", "问题3"])

# 异步调用
response = await model.ainvoke("你好")

# 流式输出
for chunk in model.stream("讲个笑话"):
    print(chunk, end='')
```

## 🔍 常见问题

### Q1: 旧代码还能用吗？

**A**: 不能！v3.0.0 已完全移除所有旧接口，必须更新代码以使用新接口。

### Q2: 如何快速迁移代码？

**A**: 按照以下映射表更新：
- `create_model()` → `init_chat_model()`
- `model.chat()` → `model.invoke()`
- `model.stream_chat()` → `model.stream()`
- `BaseLLM` → `BaseChatModel`
- `ModelConfig` → `ChatModelConfig`
- `ModelResponse` → `ChatResult`

### Q3: 为什么不保留向后兼容？

**A**: 为了：
1. 完全对齐 LangChain 标准
2. 避免接口混淆
3. 保持代码库简洁
4. 强制使用最佳实践

### Q4: 性能有变化吗？

**A**: 没有。新架构性能与旧代码一致，同时增加了异步和批量处理等高级功能。

### Q5: 迁移很复杂吗？

**A**: 不复杂！大多数情况只需：
1. 替换导入语句
2. 替换函数名
3. 替换方法名
查看上面的迁移示例即可快速完成。

### Q6: 如何报告问题？

**A**: 如果遇到问题，请：
1. 检查本文档的迁移指南
2. 查看 QUICK_START.md 的示例代码
3. 提交 Issue 并附上详细信息

## 📅 时间线

- **2026-01-13**: v2.0.0 发布，引入新架构（保留向后兼容）
- **2026-01-13**: v3.0.0 发布，移除所有旧接口，完全切换到新架构

## 📚 相关文档

- [快速上手指南](./QUICK_START.md)
- [完整 API 文档](./README.md)
- [提供商实现指南](./PROVIDERS.md)

---

**版本**: 3.0.0  
**更新日期**: 2026-01-13  
**状态**: ✅ 完成（已移除所有向后兼容代码）  
**重大变更**: 旧接口已完全移除
