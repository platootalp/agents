# GitHub Agent 使用说明

## 🎯 功能概述

`github_agent.py` 是一个基于 LangGraph 的智能 GitHub Stars 索引生成器，具有以下特点：

- ✅ **自动获取数据** - 直接从 GitHub API 获取 starred 仓库
- ✅ **智能分类** - 使用 Qwen 模型智能分析和分类
- ✅ **学习路径** - 自动生成学习建议和推荐
- ✅ **无参数运行** - 支持从环境变量读取配置
- ✅ **灵活使用** - 可命令行运行或代码调用

## 📋 依赖安装

```bash
pip install requests langchain-openai langgraph langchain-core python-dotenv
```

## 🔑 环境配置

在项目根目录创建 `.env` 文件（或设置环境变量）：

```bash
# GitHub Token（必需）
GITHUB_TOKEN=ghp_xxxxxxxxxxxxx

# DashScope API Key（必需，用于 Qwen 模型）
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxx

# Qwen 模型配置（可选）
DASHSCOPE_API_MODEL=qwen3-max-preview
DASHSCOPE_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 获取 GitHub Token

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 勾选 `user:read` 权限
4. 生成并复制 token

### 获取 DashScope API Key

1. 访问 https://dashscope.console.aliyun.com/
2. 登录并获取 API Key

---

## 🚀 使用方式

### 方式 1：无参数运行（推荐）

设置好环境变量后，直接运行：

```bash
cd src/agent
python github_agent.py
```

默认生成 `stars_index_ai.md` 文件。

### 方式 2：命令行参数运行

```bash
# 指定输出文件
python github_agent.py --output my_stars.md

# 过滤高质量项目（100+ stars）
python github_agent.py --min-stars 100

# 获取其他用户的公开 stars
python github_agent.py --username other-user

# 组合使用
python github_agent.py \
  --min-stars 50 \
  --output quality_stars.md
```

### 方式 3：代码调用

```python
from src.agent.github_agent import run_agent

# 基本用法（从环境变量读取 token）
result = run_agent()

# 指定参数
result = run_agent(
    github_token="ghp_xxxxx",  # 可选，默认从环境变量读取
    username=None,              # 可选，默认使用认证用户
    min_stars=50,              # 可选，默认 0
    output="my_stars.md",      # 可选，默认 stars_index_ai.md
)

# 访问结果
print(f"获取了 {len(result['repositories'])} 个仓库")
print(f"分为 {len(result['categories'])} 个分类")
print(result['markdown_output'])  # Markdown 内容
```

---

## 🔄 工作流程

Agent 采用 LangGraph 工作流，包含 4 个步骤：

```
START
  ↓
1. fetch (获取 GitHub Stars 数据)
  ├─ 调用 GitHub API
  ├─ 提取仓库元数据
  └─ 可选过滤（min_stars）
  ↓
2. analyze (智能分析和分类)
  ├─ Qwen 模型分析技术栈
  ├─ 跨语言智能分类
  └─ 生成分类描述
  ↓
3. recommend (生成推荐)
  ├─ 学习路径规划
  ├─ 关键项目推荐
  └─ 技术栈组合建议
  ↓
4. generate (生成 Markdown)
  ├─ 格式化文档
  ├─ 添加统计信息
  └─ 输出完整索引
  ↓
END
```

---

## 📊 输出示例

生成的 Markdown 文档包含：

```markdown
# 🌟 GitHub Stars 智能索引

## 📖 关于
- **总收藏**: 120 个项目
- **智能分类**: 8 个领域
- **AI 分析**: 由 LangGraph Agent (Qwen) 智能整理

## 📋 目录
- [AI/机器学习 → LLM工具链](#...) (15个)
- [Web开发 → 前端框架](#...) (12个)
...

## 💡 AI 推荐

### 学习路径
- **LLM 应用开发**: LangChain → LlamaIndex → AutoGen
- **深度学习**: PyTorch → Transformers → DeepSpeed

### 关键项目
- ⭐ [langchain-ai/langchain](URL) - LLM 应用开发必备

## AI/机器学习 → LLM工具链
*构建大语言模型应用的核心框架和工具集*

收录 15 个项目

| 名称 | 简介 | Stars | 语言 | 链接 |
|------|------|-------|------|------|
| **langchain** | Build LLM apps | ⭐ 85.2k | Python | [🔗](...) |
...
```

---

## 💡 使用场景

### 场景 1：个人技术栈整理

```python
# 获取所有 stars，自动分类
result = run_agent(output="my_tech_stack.md")
```

### 场景 2：高质量项目筛选

```python
# 只保留 100+ stars 的项目
result = run_agent(
    min_stars=100,
    output="top_projects.md"
)
```

### 场景 3：定期更新

```python
# 定时任务脚本
from datetime import datetime
from src.agent.github_agent import run_agent

date_str = datetime.now().strftime('%Y%m%d')
result = run_agent(
    output=f"stars_index_{date_str}.md"
)
```

### 场景 4：团队资源收集

```python
# 收集团队成员的 stars
members = ['alice', 'bob', 'charlie']

for member in members:
    result = run_agent(
        username=member,
        output=f"stars_{member}.md"
    )
```

---

## 🔧 高级配置

### 自定义 LLM 模型

```python
from src.util import get_qwen_model
from src.agent.github_agent import run_agent

# 使用自定义模型配置
llm = get_qwen_model()  # 自动从环境变量读取配置

result = run_agent(
    llm=llm,
    output="custom_index.md"
)
```

### 错误处理

```python
from src.agent.github_agent import run_agent
import requests

try:
    result = run_agent()
except ValueError as e:
    print(f"配置错误: {e}")
except requests.exceptions.HTTPError as e:
    print(f"GitHub API 错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

---

## ❓ 常见问题

### Q: 如何无参数运行？

**A:** 设置好 `.env` 文件中的环境变量后，直接运行：

```bash
python github_agent.py
```

### Q: 使用什么模型？

**A:** 使用 Qwen 模型（通过 `src/util.py` 的 `get_qwen_model()`），从 `.env` 文件读取配置。

### Q: 如何指定其他用户？

**A:** 使用 `--username` 参数：

```bash
python github_agent.py --username other-user
```

### Q: 生成很慢怎么办？

**A:** 可以：
1. 使用 `--min-stars` 过滤，减少分析数量
2. Agent 只分析前 50 个仓库（代码中限制）
3. 等待 3-5 分钟是正常的

### Q: Token 无效？

**A:** 
1. 检查 `.env` 文件中的 `GITHUB_TOKEN`
2. 确认 token 有 `user:read` 权限
3. Token 可能已过期，重新生成

### Q: 如何在 Jupyter Notebook 中使用？

**A:**

```python
# 在 Notebook 中
from src.agent.github_agent import run_agent

result = run_agent(
    min_stars=100,
    output="notebook_stars.md"
)

# 查看结果
print(result['markdown_output'])
```

---

## 📝 完整示例

```python
#!/usr/bin/env python3
"""
完整使用示例
"""
import os
from dotenv import load_dotenv
from src.agent.github_agent import run_agent

# 加载环境变量
load_dotenv()

def main():
    # 检查环境变量
    if not os.getenv("GITHUB_TOKEN"):
        print("错误: 未设置 GITHUB_TOKEN")
        return
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("错误: 未设置 DASHSCOPE_API_KEY")
        return
    
    # 运行 Agent
    print("开始生成 GitHub Stars 智能索引...\n")
    
    result = run_agent(
        min_stars=50,           # 只保留 50+ stars
        output="my_stars.md"   # 输出文件
    )
    
    # 打印统计
    print("\n统计信息:")
    print(f"  总仓库: {len(result['repositories'])}")
    print(f"  分类数: {len(result['categories'])}")
    
    # 打印分类
    print("\n分类详情:")
    for cat_name, cat_repos in result['categories'].items():
        print(f"  - {cat_name}: {len(cat_repos)} 个")

if __name__ == "__main__":
    main()
```

---

## 🎉 总结

**GitHub Agent** 提供了三种使用方式：

1. **命令行** - 简单快捷，适合日常使用
2. **代码调用** - 灵活强大，适合集成和自动化
3. **无参数运行** - 最便捷，环境配置好后直接运行

**推荐工作流**：
1. 配置 `.env` 文件
2. 无参数运行快速生成
3. 根据需要调整参数

享受 AI 驱动的智能索引生成！🚀
