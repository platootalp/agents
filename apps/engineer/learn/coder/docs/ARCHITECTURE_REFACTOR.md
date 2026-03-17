# Skill系统架构重构说明

## 原架构问题分析

### 1. 类职责混合（SRP违反）

| 类 | 原职责 | 问题 |
|---|-------|------|
| `SkillRegistry` | 注册 + 加载 + 工具管理 + Prompt构建 | 4个职责，任何变化都可能影响它 |
| `SkillExplorer` | 文件操作 + 路径解析 + 安全校验 | 工具集合但被命名为"Explorer" |
| `SkillLoader` | 静态工具方法 | 只被Registry使用，却独立成类 |
| `SkillUseAgent` | LLM交互 + 工具映射 + 参数解析 + 提示构建 | 太厚重，逻辑分散 |

### 2. 代码重复

```python
# 原代码中JSON参数解析出现3次：
# 1. _call_explorer()
# 2. _handle_load_skill_main()
# 3. _handle_unknown_tool()

# 统一为：
args = json.loads(args_str) if args_str.startswith("{") else {"skill_name": args_str}
```

### 3. 依赖关系混乱

```
原架构：
SkillUseAgent → SkillRegistry → SkillExplorer → SkillLoader
     ↓                ↓
   直接依赖      创建并持有

问题：Agent依赖Registry的具体实现，无法替换为其他存储方式
```

## 新架构设计

### 核心原则

1. **单一职责**：每个类只有一个改变的理由
2. **显式依赖**：通过构造函数注入依赖，不使用全局状态
3. **工具集合**：所有skill相关操作封装在ToolSet中

### 新类结构

```
┌─────────────────────────────────────────┐
│         SkillUseAgent                   │  ◄── 薄代理层（仅LLM交互）
│    - 构建system_prompt                  │
│    - 协调调用流程                        │
│    - 维护对话历史                        │
├─────────────────────────────────────────┤
│         SkillToolSet                    │  ◄── 工具集合（所有操作）
│    - 5个工具的定义和实现                  │
│    - 统一参数解析                        │
│    - 协调Repository和文件系统            │
├─────────────────────────────────────────┤
│         SkillRepository                 │  ◄── 纯数据存储（CRUD）
│    - 注册/查询 Skill 对象               │
│    - 从目录加载                          │
└─────────────────────────────────────────┘
```

### 职责划分

| 类 | 新职责 | 说明 |
|---|-------|------|
| `SkillRepository` | Skill对象的CRUD | 纯内存存储，可替换为数据库 |
| `SkillToolSet` | 所有工具操作的集合 | 包含5个工具的定义和执行逻辑 |
| `SkillUseAgent` | LLM交互代理 | 薄层，所有业务逻辑委托给ToolSet |

## 核心流程（带注释）

```python
# 1. 初始化阶段
agent = SkillUseAgent(name="Agent", model=Model())
agent.setup_skills("./skills")  # 加载skill到repository，创建toolset

# 2. 用户请求阶段
agent.invoke("生成PPT")
#   ├── 构建system_prompt（包含可用skill列表）
#   └── 初始化message_history

# 3. LLM交互循环
for step in range(max_steps):
    response = model.generate(messages, tools)
    #   └── LLM决定：调用工具 或 生成回答

    if has_tool_calls:
        for tool_call in tool_calls:
            # 通过toolset执行工具
            result = toolset.execute_tool(tool_name, args)
            messages.append(tool_result)
    else:
        return response.content  # 完成！

# 4. 工具执行流程
toolset.execute_tool("read_skill_file", args)
#   ├── 解析参数（统一入口）
#   ├── 调用具体处理函数
#   │   └── _handle_read_file()
#   │       ├── 从repository获取skill路径
#   │       ├── 安全检查（防止路径遍历）
#   │       ├── 读取文件内容
#   │       └── 返回格式化结果
#   └── 返回结果给LLM
```

## 关键改进点

### 1. 参数解析统一化

**原代码（3处重复）：**
```python
def _call_explorer(self, func, args_str, arg_names):
    try:
        args = json.loads(args_str) if args_str.startswith("{") else {"skill_name": args_str}
        kwargs = {name: args.get(name) for name in arg_names if name in args}
        return func(**kwargs)

# 还有2个类似的解析逻辑...
```

**新代码（1处统一）：**
```python
@staticmethod
def _parse_args(args_json: str) -> Dict[str, Any]:
    if not args_json:
        return {}
    try:
        return json.loads(args_json) if args_json.startswith("{") else {"skill_name": args_json}
    except json.JSONDecodeError:
        return {"skill_name": args_json}
```

### 2. 工具定义集中化

**原代码：**
```python
# 散落在各个地方，硬编码在TOOL_DEFINITIONS列表中
# 每个工具单独创建Tool对象，重复设置parameters
```

**新代码：**
```python
TOOL_DEFINITIONS = [
    {"name": "load_skill_main", "description": "...", "params": ["skill_name"]},
    # ... 统一元数据
]

def get_tool_definitions(self) -> List[Tool]:
    return [Tool(name=d["name"], ...) for d in self.TOOL_DEFINITIONS]
```

### 3. 显式生命周期管理

**原代码（隐式创建）：**
```python
registry = SkillRegistry()  # 在__init__中创建explorer
agent = SkillUseAgent(registry=registry)  # 从registry获取explorer
# 依赖关系隐藏，难以测试
```

**新代码（显式注入）：**
```python
agent = SkillUseAgent()
agent.setup_skills("./skills")  # 显式设置，创建repository和toolset
# 依赖清晰，易于mock测试
```

## 代码量对比

| 指标 | 原版本 | 重构版 | 变化 |
|------|-------|-------|------|
| 总行数 | 519 | 586 | +67 |
| 类数量 | 4 | 3 | -1 |
| 注释行数 | ~30 | ~120 | +90 |
| 职责清晰度 | 低 | 高 | 显著提升 |

**说明**：行数增加主要是因为添加了详细的架构注释，实际逻辑代码更简洁。

## 扩展性对比

### 添加新工具

**原版本：**
1. 在`TOOL_DEFINITIONS`添加定义
2. 在`_build_tool_map`添加lambda
3. 在`invoke`的tool_map添加处理
4. 修改3个地方，容易遗漏

**新版本：**
1. 在`TOOL_DEFINITIONS`添加定义
2. 添加`_handle_xxx`方法
3. 在`execute_tool`的handlers字典中添加映射
4. 仅修改2个地方，且都在ToolSet内

### 更换存储方式

**原版本：** 需要修改`SkillRegistry`的实现

**新版本：** 只需实现新的`SkillRepository`子类，通过构造函数注入

```python
class RedisSkillRepository(SkillRepository):
    def get(self, name: str) -> Optional[Skill]:
        # 从Redis读取
        ...

# 使用
agent = SkillUseAgent(repository=RedisSkillRepository())
```

## 测试对比

**原版本测试：**
```python
def test_registry():
    registry = SkillRegistry()
    registry.load_from_directory("./skills")
    # 测试时实际读取文件系统，难以mock
```

**新版本测试：**
```python
def test_toolset():
    repo = SkillRepository()
    repo.register(Skill(name="test", description="...", base_path="/tmp"))

    toolset = SkillToolSet(repo, Path("/tmp"))
    result = toolset.execute_tool("load_skill_main", '{"skill_name": "test"}')
    # 可以轻松mock repository，单元测试更纯粹
```

## 迁移建议

### 向后兼容

重构版API与原版基本相同：

```python
# 原版
registry = SkillRegistry()
count = registry.load_from_directory("./skills")
agent = SkillUseAgent(registry=registry)
result = agent.invoke("...")

# 新版（几乎相同）
agent = SkillUseAgent()
count = agent.setup_skills("./skills")
result = agent.invoke("...")
```

### 推荐迁移步骤

1. **Phase 1**: 将`skills_use_agent_refactored.py`作为新模块引入
2. **Phase 2**: 逐步迁移现有代码使用新API
3. **Phase 3**: 测试通过后，替换原文件
4. **Phase 4**: 删除旧代码

## 总结

### 解决的问题

✅ **SRP违反**: 每个类现在只有一个明确的职责
✅ **代码重复**: 参数解析等逻辑统一到一个方法
✅ **依赖混乱**: 依赖关系显式化，通过构造函数注入
✅ **测试困难**: 可以独立测试各层，易于mock

### 付出的代价

- 代码行数略有增加（主要是注释）
- 需要理解新的类关系（但有详细注释）

### 总体评价

重构后的架构更清晰，符合SOLID原则，虽然代码量略有增加，但可维护性和可测试性显著提升，是值得的改进。
