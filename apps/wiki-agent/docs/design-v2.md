# Wiki Agent Team - 设计文档 v2

## 1. 核心问题与解决思路

基于 Oracle 审阅，确定以下关键改进：

### 1.1 过度工程化问题
**原方案**：4 个 Agent（Coordinator/Scanner/Generator/Syncer）  
**问题**：职责重叠、复杂度过高  
**解决**：简化为 3 个核心 Agent，明确边界

### 1.2 工具抽象层次问题
**原方案**：通用工具直接操作 Wiki  
**问题**：缺乏业务语义、与 wiki-auto.sh 重复  
**解决**：封装 wiki-auto.sh 为 Python 模块，提供高级抽象

### 1.3 一致性维护缺失
**原方案**：被动检测  
**问题**：实时性差、容易遗漏  
**解决**：主动监控（Git hooks + watchdog）+ 代码-文档映射配置

---

## 2. 简化后的架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Team（3个核心角色）                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  Monitor Agent   │───▶│ Generator Agent  │                  │
│  │    （监控者）     │    │    （生成者）     │                  │
│  │                  │    │                  │                  │
│  │ • Git hooks监听  │    │ • 文档内容生成    │                  │
│  │ • 文件系统监控   │    │ • 智能更新建议    │                  │
│  │ • 变更影响分析   │    │ • 模板渲染       │                  │
│  │ • 触发工作流    │    │ • 多格式输出     │                  │
│  └────────┬─────────┘    └────────┬─────────┘                  │
│           │                       │                            │
│           └───────────┬───────────┘                            │
│                       ▼                                        │
│              ┌──────────────────┐                             │
│              │ Coordinator Agent│                             │
│              │   （协调者）      │                             │
│              │                  │                             │
│              │ • 任务调度       │                             │
│              │ • 冲突解决       │                             │
│              │ • 人工确认       │                             │
│              │ • 状态管理       │                             │
│              └──────────────────┘                             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      能力层（Capabilities）                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ WikiAdapter  │  │ GitObserver  │  │ DocAnalyzer  │         │
│  │              │  │              │  │              │         │
│  │ 封装wiki-auto│  │ Git hooks    │  │ 代码分析     │         │
│  │ 提供Python API│  │ 变更检测     │  │ 影响范围     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      通用工具层（Tools）                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │  bash   │ │  glob   │ │  read   │ │  write  │ │  edit   │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Agent 详细设计

### 3.1 Monitor Agent（监控者）

**职责**：唯一入口，监听所有变更

```python
class MonitorAgent:
    """监控代码变更并触发文档更新工作流"""

    def __init__(self, config: MonitorConfig):
        self.git_observer = GitObserver(config.git_hooks_path)
        self.fs_watcher = FileSystemWatcher(config.watch_paths)
        self.impact_analyzer = ImpactAnalyzer(config.mappings)

    async def start_monitoring(self):
        """启动所有监控"""
        # 1. 设置 Git hooks
        self.git_observer.setup_hooks()

        # 2. 启动文件系统监控
        self.fs_watcher.start()

        # 3. 监听事件
        async for event in self.event_queue:
            await self.handle_event(event)

    async def handle_event(self, event: ChangeEvent):
        """处理变更事件"""
        # 分析影响范围
        impact = self.impact_analyzer.analyze(event)

        if impact.requires_doc_update:
            # 提交给 Coordinator
            await coordinator.submit_task({
                "type": "doc_update",
                "source_change": event,
                "impact": impact,
                "priority": impact.priority
            })
```

### 3.2 Generator Agent（生成者）

**职责**：文档内容生成，不直接操作存储

```python
class GeneratorAgent:
    """生成和更新文档内容"""

    def __init__(self, config: GeneratorConfig):
        self.templates = TemplateManager(config.template_dir)
        self.llm_client = LLMClient(config.llm_provider)
        self.code_parser = CodeParser()

    async def generate_doc(self, request: DocRequest) -> DocContent:
        """根据请求生成文档"""
        # 1. 读取相关代码
        code_context = await self.read_related_code(request.code_paths)

        # 2. 读取现有文档（用于增量更新）
        existing_doc = await self.read_existing_doc(request.doc_path)

        # 3. 选择模板
        template = self.templates.get_template(request.doc_type)

        # 4. 生成内容
        if existing_doc:
            content = await self.update_existing_doc(
                existing=existing_doc,
                changes=code_context,
                template=template
            )
        else:
            content = await self.generate_new_doc(
                context=code_context,
                template=template
            )

        return DocContent(
            content=content,
            metadata=self.extract_metadata(code_context),
            suggestions=self.generate_suggestions(content)
        )

    async def update_existing_doc(self, existing, changes, template) -> str:
        """智能更新现有文档"""
        # 分析变更类型
        change_type = self.classify_changes(changes)

        if change_type == "breaking":
            # API 破坏性变更，需要人工确认
            return await self.generate_with_warning(existing, changes)
        elif change_type == "additive":
            # 新增功能，自动追加
            return await self.merge_additions(existing, changes, template)
        else:
            # 其他变更，智能合并
            return await self.smart_merge(existing, changes)
```

### 3.3 Coordinator Agent（协调者）

**职责**：决策中心，处理冲突，人工交互

```python
class CoordinatorAgent:
    """协调文档更新工作流"""

    def __init__(self, config: CoordinatorConfig):
        self.task_queue = PriorityQueue()
        self.conflict_resolver = ConflictResolver(config.conflict_rules)
        self.human_interface = HumanInterface(config.approval_levels)
        self.state_manager = StateManager()

    async def submit_task(self, task: DocTask):
        """提交文档更新任务"""
        # 1. 检查冲突
        conflicts = await self.detect_conflicts(task)

        if conflicts:
            resolution = await self.conflict_resolver.resolve(conflicts)
            if resolution.requires_human:
                # 转人工确认
                await self.human_interface.request_approval(task, conflicts)
                return

        # 2. 确定自动化级别
        auto_level = self.determine_automation_level(task)

        if auto_level == "full_auto":
            # 全自动执行
            await self.execute_task(task)
        elif auto_level == "suggest":
            # 生成建议，人工确认
            suggestion = await self.generator.generate_doc(task)
            await self.human_interface.present_suggestion(task, suggestion)
        else:
            # 人工全程参与
            await self.human_interface.interactive_mode(task)

    async def execute_task(self, task: DocTask):
        """执行文档更新任务"""
        # 1. 生成文档内容
        doc_content = await self.generator.generate_doc(task)

        # 2. 同步到各存储后端
        for target in task.targets:
            await self.sync_to_target(doc_content, target)

        # 3. 记录审计日志
        await self.audit_logger.log(task, doc_content)

    def determine_automation_level(self, task: DocTask) -> str:
        """确定自动化级别"""
        rules = [
            (task.is_breaking_change, "manual"),
            (task.affects_multiple_teams, "suggest"),
            (task.doc_type == "api", "suggest"),
            (task.change_size > 1000, "suggest"),
        ]

        for condition, level in rules:
            if condition:
                return level

        return "full_auto"
```

---

## 4. 代码-文档映射配置

```yaml
# config/mappings.yaml

# 定义代码与文档的映射关系
mappings:
  # API 文档映射
  - name: "api_docs"
    description: "API 接口文档"
    trigger:
      paths:
        - "src/**/*.api.ts"
        - "src/**/*.controller.java"
      events: ["modify", "add"]
    doc_targets:
      - type: "wiki"
        location: "API文档/{service_name}"
        template: "api_doc"
      - type: "git"
        location: "docs/api/{service_name}.md"
    auto_update: true
    approval_required: false

  # 架构文档映射
  - name: "architecture_docs"
    description: "系统架构文档"
    trigger:
      paths:
        - "architecture/**/*.md"
        - "docs/architecture/*.md"
      events: ["modify"]
    doc_targets:
      - type: "wiki"
        location: "架构文档/{component_name}"
    auto_update: false  # 架构变更需要人工确认
    approval_required: true

  # 产品功能文档
  - name: "feature_docs"
    description: "产品功能文档"
    trigger:
      pr_labels: ["feature", "product"]
      paths:
        - "features/**/*.md"
    doc_targets:
      - type: "wiki"
        location: "产品文档/{feature_name}"
      - type: "notion"
        database: "产品文档"
    auto_update: false
    approval_required: true
    notify: ["product-team"]

  # 版本更新日志
  - name: "changelog"
    description: "版本更新日志"
    trigger:
      events: ["tag_push"]
      tag_pattern: "v*"
    doc_targets:
      - type: "git"
        location: "CHANGELOG.md"
      - type: "wiki"
        location: "版本更新/{version}"
    auto_update: true
    generator: "changelog_generator"

# 冲突解决规则
conflict_rules:
  # 多源文档优先级
  priority:
    - wiki: 100      # Wiki 是主源
    - git: 80        # Git 次之
    - notion: 60     # Notion 用于产品文档

  # 自动合并策略
  merge_strategy:
    - condition: "all_sources_same_type"
      action: "newest_wins"
    - condition: "text_conflict_small"
      action: "attempt_merge"
    - condition: "api_doc_conflict"
      action: "require_manual"
```

---

## 5. WikiAdapter 设计（封装 wiki-auto.sh）

```python
class WikiAdapter:
    """Wiki 操作适配器 - 封装 wiki-auto.sh"""

    def __init__(self, config: WikiConfig):
        self.base_url = config.base_url
        self.space_key = config.space_key
        self.script_path = config.wiki_auto_script
        self.cache = CacheManager()

    async def search_page(self, query: str) -> List[WikiPage]:
        """搜索页面"""
        # 优先查缓存
        cached = self.cache.get(f"search:{query}")
        if cached:
            return cached

        # 调用 wiki-auto.sh 或 REST API
        result = await self._exec_script(
            "search",
            {"query": query, "space": self.space_key}
        )

        pages = self._parse_search_result(result)
        self.cache.set(f"search:{query}", pages, ttl=3600)
        return pages

    async def create_page(
        self,
        title: str,
        content: str,
        parent_id: Optional[str] = None,
        parent_title: Optional[str] = None
    ) -> WikiPage:
        """创建页面"""
        # 如果提供了 parent_title，先搜索获取 ID
        if parent_title and not parent_id:
            parents = await self.search_page(parent_title)
            if parents:
                parent_id = parents[0].id

        # 准备参数
        params = {
            "title": title,
            "content": content,
            "space": self.space_key
        }
        if parent_id:
            params["parent_id"] = parent_id

        # 执行创建
        result = await self._exec_script("create", params)
        return self._parse_page_result(result)

    async def update_page(
        self,
        page_id: str,
        content: str,
        version_comment: Optional[str] = None
    ) -> WikiPage:
        """更新页面"""
        params = {
            "page_id": page_id,
            "content": content,
            "space": self.space_key
        }
        if version_comment:
            params["comment"] = version_comment

        result = await self._exec_script("update", params)

        # 清除缓存
        self.cache.invalidate(f"page:{page_id}")

        return self._parse_page_result(result)

    async def get_page(self, page_id: str) -> Optional[WikiPage]:
        """获取页面内容"""
        # 优先查缓存
        cached = self.cache.get(f"page:{page_id}")
        if cached:
            return cached

        result = await self._exec_script(
            "get",
            {"page_id": page_id}
        )

        page = self._parse_page_result(result)
        self.cache.set(f"page:{page_id}", page, ttl=1800)
        return page

    async def _exec_script(self, action: str, params: Dict) -> str:
        """执行 wiki-auto.sh 脚本"""
        cmd = ["bash", self.script_path, f"--{action}"]

        for key, value in params.items():
            cmd.extend([f"--{key}", str(value)])

        result = await self.tool_executor.execute("bash", {
            "command": " ".join(cmd),
            "timeout": 60
        })

        if not result.success:
            raise WikiOperationError(f"Wiki operation failed: {result.error}")

        return result.data["stdout"]
```

---

## 6. 实施路线图

### Phase 1: 基础封装（第 1 周）
- [ ] 封装 wiki-auto.sh 为 WikiAdapter
- [ ] 实现基础工具层（bash/glob/read/write/edit/grep）
- [ ] 创建项目骨架和配置系统

### Phase 2: Monitor Agent（第 2 周）
- [ ] 实现 Git hooks 监听
- [ ] 实现文件系统监控（watchdog）
- [ ] 实现代码-文档映射解析

### Phase 3: Generator Agent（第 3 周）
- [ ] 实现模板系统
- [ ] 集成 LLM 生成能力
- [ ] 实现文档更新策略

### Phase 4: Coordinator Agent（第 4 周）
- [ ] 实现任务调度
- [ ] 实现冲突检测与解决
- [ ] 实现人工确认流程

### Phase 5: 集成测试（第 5 周）
- [ ] 端到端测试
- [ ] 性能优化
- [ ] 文档和部署

---

## 7. 关键决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Agent 数量 | 3 个 | 职责清晰，避免过度工程 |
| Wiki 操作 | 封装 wiki-auto.sh | 复用现有功能，减少重复开发 |
| 监控方式 | Git hooks + watchdog | 主动监控，实时响应 |
| 配置格式 | YAML | 易读易维护，支持注释 |
| 缓存策略 | SQLite + TTL | 轻量，无需额外依赖 |

这个设计方案是否满足你的需求？有哪些部分需要调整？
