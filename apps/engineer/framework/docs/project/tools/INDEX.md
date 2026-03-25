# 工具系统文档索引

## 📚 文档导航

本目录包含工具系统的完整文档，帮助您快速了解和使用工具集成功能。

### 📖 文档列表

| 文档 | 说明 | 适合人群 |
|------|------|----------|
| [QUICK_START.md](./QUICK_START.md) | 快速上手指南 | 初次使用者 |
| [README.md](./README.md) | 完整使用文档 | 所有用户 |
| [IMPLEMENTATION.md](./IMPLEMENTATION.md) | 实现报告和技术细节 | 开发者、维护者 |

### 🚀 推荐阅读顺序

1. **初学者路径**
   - 先读 `QUICK_START.md` - 5分钟快速上手
   - 再读 `README.md` - 深入了解所有功能
   - 最后读 `IMPLEMENTATION.md` - 了解实现细节

2. **开发者路径**
   - 先读 `README.md` - 了解整体架构
   - 再读 `IMPLEMENTATION.md` - 掌握技术细节
   - 参考 `QUICK_START.md` - 快速验证功能

3. **维护者路径**
   - 先读 `IMPLEMENTATION.md` - 了解代码结构
   - 再读 `README.md` - 理解设计理念
   - 参考 `QUICK_START.md` - 进行功能测试

### 📋 文档概要

#### QUICK_START.md
- 5分钟快速开始
- 基础示例代码
- 常见使用场景
- 内置工具列表
- 故障排除

#### README.md
- 功能特性介绍
- 核心组件详解
- 使用指南（装饰器、函数式、类式）
- 内置工具文档（11个工具）
- LLM 集成示例
- 最佳实践
- 扩展开发指南

#### IMPLEMENTATION.md
- 实现概述
- 已完成功能模块
- 代码统计
- 架构设计
- 技术栈
- 设计特点
- 测试说明

### 🔗 相关资源

#### 代码位置
```
src/core/tools/
├── base_tool.py       # 基础抽象类
├── tool.py            # 工具实现类
├── decorators.py      # 装饰器系统
├── builtin_tools.py   # 内置工具集
├── tool_manager.py    # 工具管理器
└── __init__.py        # 模块导出
```

#### 示例代码
```
examples/
├── tools_example.py       # 完整示例（8个）
└── tools_simple_demo.py   # 简化演示（6个）
```

#### 测试代码
```
tests/
└── test_tools.py          # 单元测试（30+个）
```

### 💡 快速链接

- **快速开始**: [QUICK_START.md](./QUICK_START.md)
- **完整文档**: [README.md](./README.md)
- **实现报告**: [IMPLEMENTATION.md](./IMPLEMENTATION.md)

### 🎯 使用场景

1. **我想快速上手** → 阅读 `QUICK_START.md`
2. **我想了解所有功能** → 阅读 `README.md`
3. **我想扩展工具系统** → 阅读 `README.md` + `IMPLEMENTATION.md`
4. **我想了解设计思路** → 阅读 `IMPLEMENTATION.md`
5. **我遇到了问题** → 查看 `QUICK_START.md` 故障排除部分

### 📊 文档统计

| 项目 | 数量 |
|------|------|
| 文档总数 | 3个 |
| 总字数 | ~24000字 |
| 代码示例 | 100+个 |
| 章节数 | 30+个 |

### 🔄 文档更新

- **最后更新**: 2026-01-13
- **版本**: 1.0.0
- **状态**: ✅ 已完成

### 📞 支持

如有问题，请参考：
1. 故障排除部分（在 `QUICK_START.md` 中）
2. 最佳实践部分（在 `README.md` 中）
3. 单元测试示例（在 `tests/test_tools.py` 中）

---

**开始阅读**: [快速上手指南 →](./QUICK_START.md)

