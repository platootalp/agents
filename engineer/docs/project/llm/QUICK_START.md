# LLM 核心模块快速上手指南

## 🚀 5分钟快速开始

### 1. 最简单的使用方式

```python
from src.core import init_chat_model

# 初始化模型（自动推断类型）
model = init_chat_model("gpt-4")

# 调用模型
response = model.invoke("你好，请介绍一下你自己")
print(response.content)
```

### 2. 使用类型特定工厂

```python
from src.core import ChatOpenAI, ChatAnthropic, ChatOllama

# OpenAI
model = ChatOpenAI(model_name="gpt-4")

# Anthropic
model = ChatAnthropic(model_name="claude-3-opus-20240229")

# Ollama (本地)
model = ChatOllama(model_name="llama2")

# 统一的接口
response = model.invoke("你好")
```

### 3. 流式输出

```python
from src.core import init_chat_model

model = init_chat_model("gpt-4")

# 流式输出
for chunk in model.stream("讲个笑话"):
    print(chunk, end='', flush=True)
```

## 📚 核心概念

### 消息类型

```python
from src.core import HumanMessage, AIMessage, SystemMessage

# 系统消息：设置助手行为
system_msg = SystemMessage(content="你是一个友好的助手")

# 人类消息：用户输入
human_msg = HumanMessage(content="你好")

# AI 消息：助手回复
ai_msg = AIMessage(content="你好！我能帮你什么？")
```

### 多轮对话

```python
from src.core import init_chat_model, HumanMessage, SystemMessage

model = init_chat_model("gpt-4")

messages = [
    SystemMessage(content="你是一个Python专家"),
    HumanMessage(content="什么是列表推导式？"),
]

response = model.invoke(messages)
print(response.content)

# 继续对话
messages.append(AIMessage(content=response.content))
messages.append(HumanMessage(content="请给个例子"))

response = model.invoke(messages)
print(response.content)
```

## 🎯 常用场景

### 场景 1: 简单问答

```python
from src.core import quick_invoke

# 快速调用
answer = quick_invoke(
    text="什么是人工智能？",
    model="gpt-3.5-turbo"
)
print(answer)
```

### 场景 2: 批量处理

```python
from src.core import init_chat_model

model = init_chat_model("gpt-3.5-turbo")

questions = [
    "什么是Python？",
    "什么是JavaScript？",
    "什么是Java？"
]

# 批量调用
answers = model.batch(questions)

for q, a in zip(questions, answers):
    print(f"Q: {q}")
    print(f"A: {a.content}\n")
```

### 场景 3: 异步调用

```python
import asyncio
from src.core import init_chat_model

async def main():
    model = init_chat_model("gpt-4")
    
    # 异步调用
    response = await model.ainvoke("你好")
    print(response.content)
    
    # 异步批量
    responses = await model.abatch([
        "问题1",
        "问题2",
        "问题3"
    ])
    
    for resp in responses:
        print(resp.content)

asyncio.run(main())
```

### 场景 4: 配置参数

```python
from src.core import ChatOpenAI

model = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,        # 控制随机性
    max_tokens=2000,        # 最大输出长度
    api_key="sk-...",      # API 密钥
)

response = model.invoke("写一首诗")
```

## 🔧 高级用法

### 使用配置对象

```python
from src.core import ChatModelConfig, ModelType, create_chat_model

config = ChatModelConfig(
    model_name="gpt-4",
    model_type=ModelType.OPENAI,
    temperature=0.8,
    max_tokens=3000,
    api_key="sk-..."
)

model = create_chat_model(config)
```

### 性能监控

```python
from src.core import init_chat_model

model = init_chat_model("gpt-4")

# 调用模型
response = model.invoke("你好")

# 查看性能指标
metrics = model.get_metrics()
print(f"平均延迟: {metrics.avg_latency:.2f}秒")
print(f"成功率: {metrics.success_rate * 100:.1f}%")
print(f"总调用次数: {metrics.total_calls}")
```

## 📖 支持的模型

### OpenAI

```python
from src.core import ChatOpenAI

model = ChatOpenAI(model_name="gpt-4")
model = ChatOpenAI(model_name="gpt-3.5-turbo")
```

### Anthropic

```python
from src.core import ChatAnthropic

model = ChatAnthropic(model_name="claude-3-opus-20240229")
model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
```

### 通义千问

```python
from src.core import init_chat_model

model = init_chat_model(
    model="qwen-turbo",
    model_provider="qwen"
)
```

### Ollama (本地)

```python
from src.core import ChatOllama

model = ChatOllama(
    model_name="llama2",
    base_url="http://localhost:11434"
)
```

## 💡 最佳实践

### 1. 使用环境变量管理密钥

```bash
# .env 文件
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DASHSCOPE_API_KEY=sk-...
```

```python
# 自动从环境变量读取
from src.core import ChatOpenAI

model = ChatOpenAI(model_name="gpt-4")  # 自动读取 OPENAI_API_KEY
```

### 2. 错误处理

```python
from src.core import init_chat_model

model = init_chat_model("gpt-4")

try:
    response = model.invoke("你好")
    print(response.content)
except Exception as e:
    print(f"调用失败: {e}")
```

### 3. 流式输出优化用户体验

```python
from src.core import init_chat_model

model = init_chat_model("gpt-4")

print("AI: ", end='', flush=True)
for chunk in model.stream("请详细介绍人工智能"):
    print(chunk, end='', flush=True)
print()
```

## 🆚 新旧接口对比

### 旧接口（仍可用）

```python
from src.core import create_model, Message

model = create_model("gpt-4")
response = model.chat([Message(role="user", content="你好")])
print(response.content)
```

### 新接口（推荐）

```python
from src.core import init_chat_model

model = init_chat_model("gpt-4")
response = model.invoke("你好")
print(response.content)
```

## 🐛 故障排除

### 问题 1: API 密钥错误

```python
# 确保设置了环境变量
import os
print(os.getenv("OPENAI_API_KEY"))

# 或者显式传递
model = ChatOpenAI(model_name="gpt-4", api_key="sk-...")
```

### 问题 2: 模型不存在

```python
# 使用正确的模型名称
model = ChatOpenAI(model_name="gpt-4")  # ✅
model = ChatOpenAI(model_name="gpt4")   # ❌ 错误
```

### 问题 3: 超时错误

```python
# 增加超时时间
model = init_chat_model(
    model="gpt-4",
    timeout=120  # 秒
)
```

## 📚 更多资源

- [完整 API 文档](./README.md)
- [重构指南](./REFACTORING.md)
- [提供商配置](./PROVIDERS.md)

## 🎉 开始使用

现在你已经掌握了基础知识，可以开始使用了！

```python
from src.core import init_chat_model

# 初始化你的第一个模型
model = init_chat_model("gpt-4")

# 开始对话
response = model.invoke("你好！让我们开始吧")
print(response.content)
```

祝使用愉快！ 🚀
