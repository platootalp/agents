#!/usr/bin/env python3
"""
流式输出演示

展示 github_stars_agent.py 的流式输出效果，可以实时看到 AI 的思考过程
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def main():
    """运行流式输出演示"""
    
    # 检查环境变量
    if not os.getenv("GITHUB_TOKEN"):
        print("❌ 错误: 未设置 GITHUB_TOKEN")
        print("\n请在 .env 文件中添加:")
        print("  GITHUB_TOKEN=ghp_xxxxxxxxxxxxx")
        return 1
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 未设置 DASHSCOPE_API_KEY")
        print("\n请在 .env 文件中添加:")
        print("  DASHSCOPE_API_KEY=sk_xxxxxxxxxxxxx")
        return 1
    
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "流式输出演示" + " " * 38 + "║")
    print("║" + " " * 10 + "实时查看 AI 的思考和执行过程" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    print("📌 说明:")
    print("   - 每个步骤都会显示详细进度")
    print("   - AI 分析和推荐会实时流式输出")
    print("   - 可以看到 AI 的完整思考过程")
    print()
    
    input("按 Enter 键开始...")
    
    # 导入并运行 Agent
    from src.agent.github_stars_agent import run_agent
    
    try:
        result = run_agent(
            min_stars=100,  # 只获取高质量项目，加快演示
            output="examples/output/stream_demo.md"
        )
        
        print("\n" + "╔" + "=" * 68 + "╗")
        print("║  🎉 演示完成！" + " " * 52 + "║")
        print("╚" + "=" * 68 + "╝")
        
        print(f"\n📊 统计信息:")
        print(f"   - 获取仓库: {len(result['repositories'])} 个")
        print(f"   - 分类数量: {len(result['categories'])} 个")
        print(f"   - 输出文件: examples/output/stream_demo.md")
        
        print("\n💡 提示:")
        print("   查看输出: cat examples/output/stream_demo.md")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
