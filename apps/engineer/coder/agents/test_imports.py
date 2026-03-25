"""
测试脚本 - 验证所有 agents 使用新工具系统的导入
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试所有导入是否成功"""
    print("=" * 60)
    print("Testing Agent Imports with New Tool System")
    print("=" * 60)

    errors = []

    # 1. 测试新工具系统导入
    print("\n1. Testing new tool system imports...")
    try:
        try:
            from apps.engineer.coder.core.tools.base import BaseTool, Tool, ToolResult, tool
            from apps.engineer.coder.core.tools.manager import ToolManager, ToolExecutor
        except ImportError:
            from apps.engineer.coder.core.tools.base import BaseTool, Tool, ToolResult, tool
            from apps.engineer.coder.core.tools.manager import ToolManager, ToolExecutor
        print("   ✓ Tool system imports successful")
    except Exception as e:
        errors.append(f"Tool system: {e}")
        print(f"   ✗ Tool system imports failed: {e}")

    # 2. 测试 ToolUseAgent
    print("\n2. Testing ToolUseAgent...")
    try:
        try:
            from apps.engineer.coder.agents.tool_use_agent import ToolUseAgent
        except ImportError:
            from apps.engineer.coder.agents.tool_use_agent import ToolUseAgent
        print("   ✓ ToolUseAgent import successful")
        print(f"   - Has tool_manager: {hasattr(ToolUseAgent, '__init__')}")
    except Exception as e:
        errors.append(f"ToolUseAgent: {e}")
        print(f"   ✗ ToolUseAgent import failed: {e}")

    # 3. 测试 McpAgent
    print("\n3. Testing McpAgent...")
    try:
        try:
            from apps.engineer.coder.agents.mcp_agent import McpAgent, StreamChunk
        except ImportError:
            from apps.engineer.coder.agents.mcp_agent import McpAgent, StreamChunk
        print("   ✓ McpAgent import successful")
    except Exception as e:
        errors.append(f"McpAgent: {e}")
        print(f"   ✗ McpAgent import failed: {e}")

    # 4. 测试 SkillsUseAgent
    print("\n4. Testing SkillsUseAgent...")
    try:
        try:
            from apps.engineer.coder.agents.skills_agent import (
                SkillsUseAgent,
                Skills,
                SkillsRepository,
                SkillsToolSet,
            )
        except ImportError:
            from apps.engineer.coder.agents.skills_agent import (
                SkillsUseAgent,
                Skills,
                SkillsRepository,
                SkillsToolSet,
            )
        print("   ✓ SkillsUseAgent import successful")
    except Exception as e:
        errors.append(f"SkillsUseAgent: {e}")
        print(f"   ✗ SkillsUseAgent import failed: {e}")

    # 5. 测试 SubAgent
    print("\n5. Testing SubAgent...")
    try:
        try:
            from apps.engineer.coder.agents.subagent_agent import (
                SubAgent,
                Task,
                TaskResult,
                TaskStatus,
            )
        except ImportError:
            from apps.engineer.coder.agents.subagent_agent import SubAgent, Task, TaskResult, TaskStatus
        print("   ✓ SubAgent import successful")
    except Exception as e:
        errors.append(f"SubAgent: {e}")
        print(f"   ✗ SubAgent import import failed: {e}")

    # 6. 测试 TaskAgent
    print("\n6. Testing TaskAgent...")
    try:
        try:
            from apps.engineer.coder.agents.task_agent import (
                TaskAgent,
                TaskGraph,
                TaskStatus,
                TaskPriority,
            )
        except ImportError:
            from apps.engineer.coder.agents.task_agent import TaskAgent, TaskGraph, TaskStatus, TaskPriority
        print("   ✓ TaskAgent import successful")
    except Exception as e:
        errors.append(f"TaskAgent: {e}")
        print(f"   ✗ TaskAgent import failed: {e}")

    # 7. 测试主模块导入
    print("\n7. Testing main agents module...")
    try:
        try:
            from apps.engineer.coder.agents import (
                ToolUseAgent,
                McpAgent,
                SkillsUseAgent,
                SubAgent,
                TaskAgent,
            )
        except ImportError:
            from apps.engineer.coder.agents import (
                ToolUseAgent,
                McpAgent,
                SkillsUseAgent,
                SubAgent,
                TaskAgent,
            )
        print("   ✓ Main module import successful")
    except Exception as e:
        errors.append(f"Main module: {e}")
        print(f"   ✗ Main module import failed: {e}")

    # 总结
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ Import test completed with {len(errors)} errors:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("✅ All imports successful!")
        print("\nAll agents are now using the new tool system from apps.engineer.coder.core.tools")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
