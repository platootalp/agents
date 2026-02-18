# LLM 核心模块文档索引

## 📚 文档导航

本目录包含 LLM 核心模块的完整文档，该模块已重构以对齐 LangChain 的命名规范和架构设计。

### 📖 文档列表

| 文档 | 说明 | 适合人群 |
|------|------|----------|
| [QUICK_START.md](./QUICK_START.md) | 快速上手指南 | 新用户、快速开始 |
| [REFACTORING.md](./REFACTORING.md) | 重构文档和迁移指南 | 所有用户、迁移者 |
| ~~README.md~~ | 完整 API 文档（待创建） | 深度使用者 |

### 🚀 推荐阅读顺序

1. **新用户**
   - 📘 [QUICK_START.md](./QUICK_START.md) - 5分钟快速上手
   - 📗 [REFACTORING.md](./REFACTORING.md) - 了解新架构

2. **现有用户**
   - 📙 [REFACTORING.md](./REFACTORING.md) - 迁移指南
   - 📘 [QUICK_START.md](./QUICK_START.md) - 新接口使用

3. **维护者**
   - 📙 [REFACTORING.md](./REFACTORING.md) - 了解重构细节
   - 📕 代码实现（`src/core/language_models.py`）

## 🎯 核心变更概述

### 文件命名对齐 LangChain

| 旧名称 | 新名称 |
|--------|--------|
| `base_llm.py` | `language_models.py` |
| `llm_factory.py` | `chat_model_factory.py` |

### 类名对齐 LangChain

| 旧类名 | 新类名 |
|--------|--------|
| `BaseLLM` | `BaseChatModel` |
| `ModelConfig` | `ChatModelConfig` |
| `ModelResponse` | `ChatResult` |

### 方法名对齐 LangChain

| 旧方法 | 新方法 |
|--------|--------|
| `chat()` | `invoke()` |
| `stream_chat()` | `stream()` |
| - | `batch()` ⭐ 新增 |
| - | `ainvoke()` ⭐ 新增 |
| - | `astream()` ⭐ 新增 |
| - | `abatch()` ⭐ 新增 |

## 📋 快速参考

### 创建模型

```python
# 推荐方式（对齐 LangChain）
from src.core import init_chat_model
model = init_chat_model("gpt-4")

# 类型特定工厂（对齐 LangChain）
from src.core import ChatOpenAI
model = ChatOpenAI(model_name="gpt-4")

# 旧方式（仍可用，但不推荐）
from src.core import create_model
model = create_model("gpt-4")
```

### 调用模型

```python
# 推荐方式
response = model.invoke("你好")          # 同步
response = await model.ainvoke("你好")   # 异步

# 旧方式（仍可用）
response = model.chat([Message(role="user", content="你好")])
```

### 流式输出

```python
# 推荐方式
for chunk in model.stream("你好"):       # 同步
    print(chunk, end='')

async for chunk in model.astream("你好"): # 异步
    print(chunk, end='')

# 旧方式（仍可用）
for chunk in model.stream_chat([Message(role="user", content="你好")]):
    print(chunk, end='')
```

## 🌟 新功能亮点

### 1. 异步支持 ⭐

```python
# 异步调用
response = await model.ainvoke("你好")

# 异步流式
async for chunk in model.astream("你好"):
    print(chunk, end='')

# 异步批量
responses = await model.abatch(["问题1", "问题2"])
```

### 2. 批量处理 ⭐

```python
# 同步批量
responses = model.batch(["问题1", "问题2", "问题3"])

# 异步批量
responses = await model.abatch(["问题1", "问题2", "问题3"])
```

### 3. 统一的 invoke 接口 ⭐

```python
# 支持字符串
response = model.invoke("你好")

# 支持消息列表
response = model.invoke([
    SystemMessage(content="你是助手"),
    HumanMessage(content="你好")
])
```

## 📊 版本信息

- **当前版本**: 3.0.0
- **发布日期**: 2026-01-13
- **重构状态**: ✅ 完成
- **重大变更**: ❌ 已移除所有旧接口，无向后兼容

## ⚠️ 重大变更（v3.0.0）

以下旧接口已**完全移除**，无向后兼容：

- ❌ `BaseLLM` → 必须使用 `BaseChatModel`
- ❌ `ModelConfig` → 必须使用 `ChatModelConfig`
- ❌ `ModelResponse` → 必须使用 `ChatResult`
- ❌ `create_model()` → 必须使用 `init_chat_model()`
- ❌ `chat()` 方法 → 必须使用 `invoke()` 方法
- ❌ 旧文件 `base_llm.py` 和 `llm_factory.py` 已删除

## 🔗 快速链接

### 文档
- [快速上手](./QUICK_START.md)
- [重构指南](./REFACTORING.md)

### 代码位置
- 语言模型：`src/core/language_models.py`
- 工厂函数：`src/core/chat_model_factory.py`
- 模型信息：`src/core/model_info.py`

### 示例代码
- 待补充：`examples/llm_*.py`

## 💡 使用建议

1. **所有项目**：必须使用新接口（旧接口已移除）
2. **迁移必需**：查看 REFACTORING.md 获取迁移指南
3. **学习 LangChain**：新接口完全对齐，便于学习和迁移

## 📞 获取帮助

- 查看 [QUICK_START.md](./QUICK_START.md) 的故障排除部分
- 阅读 [REFACTORING.md](./REFACTORING.md) 的常见问题
- 查看代码示例和注释

## 🎉 开始使用

选择你的学习路径：

- 👉 [我是新用户，想快速开始](./QUICK_START.md)
- 👉 [我需要迁移现有代码](./REFACTORING.md#📝-迁移指南)
- 👉 [我想了解重构细节](./REFACTORING.md#🔄-主要变更)

---

**更新日期**: 2026-01-13  
**版本**: 3.0.0  
**状态**: ✅ 已完成  
**重大变更**: Breaking Changes - 旧接口已完全移除
