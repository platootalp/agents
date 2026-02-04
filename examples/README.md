# GitHub Agent 使用示例

## 📋 示例列表

### github_agent_demo.py

演示 GitHub Agent 的三种使用方式：
1. 无参数运行
2. 带参数运行
3. 代码调用

**运行**：
```bash
cd examples
python github_agent_demo.py
```

---

## 🚀 快速开始

### 1. 配置环境

在项目根目录创建 `.env` 文件：

```bash
GITHUB_TOKEN=ghp_xxxxxxxxxxxxx
DASHSCOPE_API_KEY=sk_xxxxxxxxxxxxx
```

### 2. 安装依赖

```bash
pip install requests python-dotenv langchain-openai langgraph langchain-core
```

### 3. 运行演示

```bash
cd examples
python github_agent_demo.py
```

---

## 📝 使用示例

### 最简单的方式

```python
from src.agent.github_stars_agent import run_agent

# 一行代码，完成所有操作
result = run_agent()
```

### 指定参数

```python
result = run_agent(
    min_stars=100,          # 只获取 100+ stars
    output="my_stars.md"   # 指定输出文件
)
```

### 完整示例

```python
from src.agent.github_stars_agent import run_agent
import os

# 确保环境变量已设置
if not os.getenv("GITHUB_TOKEN"):
    print("请设置 GITHUB_TOKEN")
    exit(1)

# 运行 Agent
result = run_agent(
    min_stars=50,
    output="output/my_github_stars.md"
)

# 查看结果
print(f"✓ 获取了 {len(result['repositories'])} 个仓库")
print(f"✓ 分为 {len(result['categories'])} 个分类")

# 打印分类
print("\n分类详情:")
for cat_name, repos in result['categories'].items():
    print(f"  - {cat_name}: {len(repos)} 个")
```

---

## 📚 更多文档

- **详细使用指南**: `src/agent/github_agent_usage.md`
- **测试脚本**: `src/agent/test_github_agent.py`
- **工作流说明**: `skills/github-stars-indexer/scripts/WORKFLOW.md`
