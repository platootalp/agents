#!/usr/bin/env python3
"""
GitHub Agent 使用演示

演示三种使用方式：
1. 无参数运行（最简单）
2. 带参数运行
3. 代码调用
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


def demo_1_no_params():
    """演示 1: 无参数运行"""
    print("=" * 70)
    print("演示 1: 无参数运行（从环境变量读取所有配置）")
    print("=" * 70)
    print()
    
    print("环境变量检查:")
    print(f"  GITHUB_TOKEN: {'✓ 已设置' if os.getenv('GITHUB_TOKEN') else '✗ 未设置'}")
    print(f"  DASHSCOPE_API_KEY: {'✓ 已设置' if os.getenv('DASHSCOPE_API_KEY') else '✗ 未设置'}")
    print()
    
    if not os.getenv('GITHUB_TOKEN') or not os.getenv('DASHSCOPE_API_KEY'):
        print("⚠️  请先设置环境变量！")
        print("\n在 .env 文件中添加:")
        print("  GITHUB_TOKEN=ghp_xxxxxxxxxxxxx")
        print("  DASHSCOPE_API_KEY=sk_xxxxxxxxxxxxx")
        return
    
    from src.agent.github_agent import run_agent
    
    print("运行命令（等效）:")
    print("  python github_agent.py")
    print()
    
    print("开始执行...\n")
    
    try:
        result = run_agent(
            min_stars=100,  # 为了演示快速，只获取高质量项目
            output="examples/output/demo_no_params.md"
        )
        
        print("\n✅ 演示 1 完成!")
        print(f"   输出文件: examples/output/demo_no_params.md")
        print(f"   仓库数量: {len(result['repositories'])}")
        print(f"   分类数量: {len(result['categories'])}")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


def demo_2_with_params():
    """演示 2: 带参数运行"""
    print("\n" + "=" * 70)
    print("演示 2: 带参数运行（指定过滤和输出）")
    print("=" * 70)
    print()
    
    if not os.getenv('GITHUB_TOKEN') or not os.getenv('DASHSCOPE_API_KEY'):
        print("⚠️  跳过（环境变量未设置）")
        return
    
    from src.agent.github_agent import run_agent
    
    print("运行命令（等效）:")
    print("  python github_agent.py --min-stars 200 --output high_quality.md")
    print()
    
    print("开始执行...\n")
    
    try:
        result = run_agent(
            min_stars=200,  # 只获取高质量项目
            output="examples/output/demo_with_params.md"
        )
        
        print("\n✅ 演示 2 完成!")
        print(f"   输出文件: examples/output/demo_with_params.md")
        print(f"   仓库数量: {len(result['repositories'])}")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


def demo_3_code_usage():
    """演示 3: 在代码中调用"""
    print("\n" + "=" * 70)
    print("演示 3: 在代码中调用（编程接口）")
    print("=" * 70)
    print()
    
    if not os.getenv('GITHUB_TOKEN') or not os.getenv('DASHSCOPE_API_KEY'):
        print("⚠️  跳过（环境变量未设置）")
        return
    
    print("示例代码:")
    print("""
    from src.agent.github_agent import run_agent
    
    # 调用 Agent
    result = run_agent(
        min_stars=150,
        output="my_custom_index.md"
    )
    
    # 访问结果
    print(f"分类: {list(result['categories'].keys())}")
    """)
    print()
    
    from src.agent.github_agent import run_agent
    
    print("开始执行...\n")
    
    try:
        result = run_agent(
            min_stars=150,
            output="examples/output/demo_code_usage.md"
        )
        
        print("\n✅ 演示 3 完成!")
        print(f"   输出文件: examples/output/demo_code_usage.md")
        print(f"   仓库数量: {len(result['repositories'])}")
        
        # 显示分类
        print("\n   分类列表:")
        for cat_name, cat_repos in list(result['categories'].items())[:5]:
            print(f"     - {cat_name}: {len(cat_repos)} 个")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


def main():
    """运行所有演示"""
    print("\n🎬 GitHub Agent 使用演示\n")
    
    # 创建输出目录
    os.makedirs("examples/output", exist_ok=True)
    
    # 运行演示
    demo_1_no_params()
    
    # 其他演示可选（避免重复调用 API）
    # demo_2_with_params()
    # demo_3_code_usage()
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)
    print()
    print("生成的文件:")
    print("  examples/output/demo_no_params.md")
    # print("  examples/output/demo_with_params.md")
    # print("  examples/output/demo_code_usage.md")
    print()
    print("查看文档:")
    print("  cat src/agent/github_agent_usage.md")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
