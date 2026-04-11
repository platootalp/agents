import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv

try:
    from apps.engineer.coder.core.model import Model
    from apps.engineer.coder.core.message import (
        Message,
        UserMessage,
        AssistantMessage,
        SystemMessage,
        ToolMessage,
    )
except ImportError:
    from apps.engineer.coder.core.model import Model
    from apps.engineer.coder.core.message import (
        Message,
        UserMessage,
        AssistantMessage,
        SystemMessage,
        ToolMessage,
    )


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
        self._register_tools()
        self.max_steps = 50

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
                command, shell=True, capture_output=True, text=True, timeout=30, cwd=self.workspace
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr] {result.stderr}"
            return ToolResult(output=output, success=result.returncode == 0)
        except Exception as e:
            return ToolResult(error=str(e))

    # ============ 对话循环 ============

    def chat(self, message: str) -> str:
        """处理用户消息 - 支持多轮执行"""
        # 初始化消息列表（系统提示 + 用户输入）
        messages: List[Message] = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            UserMessage(content=message),
        ]

        for step in range(self.max_steps):
            # 调用模型
            response = self.model.generate(messages, tools=self._get_tool_definitions())

            # 处理模型返回的Message对象
            content = response.content or ""
            tool_calls = response.tool_calls or []

            # 添加助手消息到历史
            assistant_msg = AssistantMessage(content=content, tool_calls=tool_calls)
            messages.append(assistant_msg)

            # 如果没有工具调用，返回最终答案
            if not tool_calls:
                return content if content else "完成"

            # 执行工具调用
            for tool_call in tool_calls:
                result = self._execute_tool_call(tool_call)

                # 添加工具响应到消息列表
                tool_call_id = tool_call.get("id", "unknown")
                tool_result_content = result.output if result.success else result.error
                tool_msg = ToolMessage(content=tool_result_content, tool_call_id=tool_call_id)
                messages.append(tool_msg)

        # 达到最大步数限制
        return "达到最大执行步数限制，无法完成请求。"

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> ToolResult:
        """执行单个工具调用"""
        tool_id = tool_call.get("id", "unknown")
        function_info = tool_call.get("function", {})
        tool_name = function_info.get("name", "")
        tool_args_str = function_info.get("arguments", "{}")

        # 解析参数
        try:
            tool_args = json.loads(tool_args_str) if tool_args_str else {}
        except json.JSONDecodeError:
            return ToolResult(error=f"无法解析工具参数: {tool_args_str}")

        # 执行工具
        return self._execute_tool(tool_name, tool_args)

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> ToolResult:
        """根据工具名称执行对应的工具函数"""
        if tool_name in self.tools:
            try:
                return self.tools[tool_name](**tool_args)
            except Exception as e:
                return ToolResult(error=f"工具执行错误: {e}")
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
                    "description": "读取指定路径的文件内容。编辑前必须先读取文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "文件路径，相对于workspace"}
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "写入内容到指定路径的文件。如果目录不存在会自动创建。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "文件路径，相对于workspace"},
                            "content": {"type": "string", "description": "要写入的文件内容"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "编辑文件内容，使用字符串替换。old_string必须精确匹配文件中的内容。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "文件路径，相对于workspace"},
                            "old_string": {"type": "string", "description": "要替换的原始字符串"},
                            "new_string": {"type": "string", "description": "替换后的新字符串"},
                        },
                        "required": ["path", "old_string", "new_string"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "执行shell命令。谨慎使用，避免执行危险命令。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "要执行的shell命令"}
                        },
                        "required": ["command"],
                    },
                },
            },
        ]


# ============ 使用示例 ============

if __name__ == "__main__":
    load_dotenv()
    # 创建 Agent
    coder = SimpleCoderAgent(model_client=Model.from_env("openai") , workspace="./my_project")

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
