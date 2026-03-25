# Coder Agent V1 - MVP 设计方案

> 最小可用版本：能跑起来，解决核心问题

**版本**: V1.0 MVP  
**目标**: 30 分钟内部署，支持基础代码编辑  
**代码量**: ~500 行核心代码  

---

## 🎯 设计哲学

**能跑 > 完美**

MVP 版本专注于：
1. 一个文件解决问题
2. 最基础的工具集
3. 最简化的配置
4. 立即可用

---

## 📦 核心能力

```
✅ 读取文件（read_file）
✅ 写入文件（write_file）
✅ 编辑文件（edit_file）
✅ 执行 Shell（bash，基础版）
✅ 多轮对话
❌ 无沙箱
❌ 无记忆
❌ 无 MCP
```

---

## 🏗️ 极简架构

```
┌─────────────────────────────────────┐
│         SimpleCoderAgent            │
│  ┌──────────┐  ┌────────────────┐  │
│  │  Model   │  │  ToolManager   │  │
│  └──────────┘  └────────────────┘  │
└─────────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│ read  │ │ write │ │ bash  │
└───────┘ └───────┘ └───────┘
```

---

## 💻 核心代码（单文件实现）

```python
# simple_coder.py - 单文件 MVP 实现
"""
极简 Coder Agent - MVP 版本

特点：
- 单文件实现
- 零配置启动
- 基础工具集
- 对话式交互
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ToolResult:
    output: str = ""
    success: bool = True
    error: str = ""


class SimpleCoderAgent:
    """极简代码助手"""
    
    SYSTEM_PROMPT = """你是一个代码助手。你可以：
1. 读取文件：read_file(path)
2. 写入文件：write_file(path, content)
3. 编辑文件：edit_file(path, old_string, new_string)
4. 执行命令：bash(command)

重要提示：
- 编辑前必须先读取文件
- 使用精确的字符串匹配进行编辑
- 命令执行有风险，谨慎操作
"""

    def __init__(self, model_client=None, workspace: str = "."):
        self.workspace = Path(workspace).resolve()
        self.model = model_client
        self.history: List[Dict] = []
        self._register_tools()
    
    def _register_tools(self):
        """注册基础工具"""
        self.tools = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "edit_file": self._edit_file,
            "bash": self._bash,
        }
    
    # ============ 工具实现 ============
    
    def _read_file(self, path: str) -> ToolResult:
        """读取文件"""
        try:
            file_path = self.workspace / path
            if not file_path.exists():
                return ToolResult(error=f"文件不存在: {path}")
            content = file_path.read_text(encoding="utf-8")
            return ToolResult(output=content)
        except Exception as e:
            return ToolResult(error=str(e))
    
    def _write_file(self, path: str, content: str) -> ToolResult:
        """写入文件"""
        try:
            file_path = self.workspace / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return ToolResult(output=f"已写入: {path}")
        except Exception as e:
            return ToolResult(error=str(e))
    
    def _edit_file(self, path: str, old_string: str, new_string: str) -> ToolResult:
        """编辑文件（字符串替换）"""
        try:
            file_path = self.workspace / path
            if not file_path.exists():
                return ToolResult(error=f"文件不存在: {path}")
            
            content = file_path.read_text(encoding="utf-8")
            if old_string not in content:
                return ToolResult(error=f"未找到匹配字符串")
            
            new_content = content.replace(old_string, new_string, 1)
            file_path.write_text(new_content, encoding="utf-8")
            return ToolResult(output=f"已编辑: {path}")
        except Exception as e:
            return ToolResult(error=str(e))
    
    def _bash(self, command: str) -> ToolResult:
        """执行 shell 命令（基础版，无沙箱）"""
        try:
            # 基础安全检查
            dangerous = ["rm -rf /", "mkfs", "dd if=/dev/zero"]
            for d in dangerous:
                if d in command:
                    return ToolResult(error=f"危险命令被拒绝: {d}")
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.workspace
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr] {result.stderr}"
            return ToolResult(output=output, success=result.returncode == 0)
        except Exception as e:
            return ToolResult(error=str(e))
    
    # ============ 对话循环 ============
    
    def chat(self, message: str) -> str:
        """处理用户消息"""
        # 添加用户消息
        self.history.append({"role": "user", "content": message})
        
        # 构建系统提示
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            *self.history
        ]
        
        # 调用模型（简化版，实际需接入真实模型）
        # response = self.model.generate(messages, tools=self._get_tool_definitions())
        
        # 模拟回复
        response = self._simulate_response(message)
        
        # 处理工具调用
        if response.get("tool_calls"):
            for tool_call in response["tool_calls"]:
                result = self._execute_tool(tool_call)
                # 添加工具结果到历史
                self.history.append({
                    "role": "assistant",
                    "content": f"工具结果: {result.output}"
                })
        
        reply = response.get("content", "完成")
        self.history.append({"role": "assistant", "content": reply})
        return reply
    
    def _execute_tool(self, tool_call: Dict) -> ToolResult:
        """执行工具调用"""
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        
        if tool_name in self.tools:
            return self.tools[tool_name](**tool_args)
        return ToolResult(error=f"未知工具: {tool_name}")
    
    def _simulate_response(self, message: str) -> Dict:
        """模拟模型响应（实际项目需接入真实模型）"""
        # 这里只是一个占位符，实际应该调用 OpenAI/Anthropic 等 API
        return {"content": f"收到: {message}"}
    
    def _get_tool_definitions(self) -> List[Dict]:
        """获取工具定义（OpenAI 格式）"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取文件内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        }
                    }
                }
            },
            # ... 其他工具定义
        ]


# ============ 使用示例 ============

if __name__ == "__main__":
    # 创建 Agent
    coder = SimpleCoderAgent(workspace="./my_project")
    
    # 交互式对话
    print("=" * 50)
    print("Simple Coder - MVP 版本")
    print("=" * 50)
    
    while True:
        user_input = input("\n👤 你: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        response = coder.chat(user_input)
        print(f"\n🤖 Agent: {response}")
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 只需要标准库！
# Python 3.9+
```

### 2. 配置模型

```python
# 接入 OpenAI（示例）
import openai

class OpenAIModel:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, messages, tools=None):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools
        )
        return response

# 使用
coder = SimpleCoderAgent(
    model_client=OpenAIModel(api_key="sk-..."),
    workspace="./my_project"
)
```

### 3. 运行

```bash
python simple_coder.py
```

---

## 📋 功能清单

| 功能 | 状态 | 说明 |
|------|------|------|
| 读取文件 | ✅ | 基础文件读取 |
| 写入文件 | ✅ | 创建新文件 |
| 编辑文件 | ✅ | 字符串替换 |
| 执行命令 | ✅ | 基础 shell |
| 多轮对话 | ✅ | 上下文保持 |
| 沙箱安全 | ❌ | V2 添加 |
| 记忆系统 | ❌ | V2 添加 |
| MCP 支持 | ❌ | V3 添加 |

---

## ⚠️ 安全警告

MVP 版本**不提供完整安全保障**：
- 只有基础危险命令拦截
- 无沙箱隔离
- 无路径限制
- 建议仅在本地开发环境使用

**生产环境请使用 V2+ 版本**

---

## 🔄 演进路径

```
V1 (MVP)       →    V2 (进阶)      →    V3 (完整)
单文件实现           模块化架构           企业级功能
基础工具            沙箱 + 记忆          MCP + Subagent
无状态              SQLite 存储          分布式支持
30 分钟部署         2 小时部署           1 天部署
```

---

## 📝 代码统计

- **总代码行数**: ~200 行
- **文件数量**: 1 个
- **依赖**: 仅 Python 标准库
- **配置**: 零配置

---

**下一步**: 当 MVP 验证通过后，升级到 [V2 进阶版本](./design-v2-advanced.md)
