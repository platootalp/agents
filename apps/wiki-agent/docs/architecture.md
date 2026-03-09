# Wiki Agent Team - 架构设计文档

## 1. 核心定位

**使命**：维护代码与文档的一致性，覆盖产品、技术、用户、版本、运维等多维度文档。

**与 DeepWiki 的区别**：
- DeepWiki：仅技术文档，单向生成
- Wiki Agent：多类型文档，双向同步，持续维护

## 2. 架构设计

### 2.1 Agent Team 组织

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Team 架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                Coordinator (协调者)                       │  │
│  │         任务分发、结果整合、冲突解决                        │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                       │
│         ┌───────────────┼───────────────┐                       │
│         ▼               ▼               ▼                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │   Scanner   │ │  Generator  │ │   Syncer    │               │
│  │   (扫描者)   │ │  (生成者)    │ │  (同步者)    │               │
│  │             │ │             │ │             │               │
│  │ 监控代码变更 │ │ 生成文档内容 │ │ 多源同步    │               │
│  │ 检测文档过期 │ │ 多类型输出   │ │ 一致性维护  │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      通用工具层 (OpenCode Style)                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │  bash   │ │  glob   │ │  read   │ │  write  │ │  edit   │  │
│  │  命令行 │ │  文件搜索│ │  读取   │ │  写入   │ │  编辑   │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │  grep   │ │   lsp   │ │  skill  │ │  task   │ │  web    │  │
│  │  搜索   │ │ 代码分析│ │ 技能调用│ │ 子任务  │ │  访问   │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent 角色定义

#### Coordinator（协调者）
```python
class Coordinator:
    """任务协调中心"""

    def analyze_request(self, request: str) -> TaskPlan:
        """分析用户需求，拆解为子任务"""
        # 示例："API 有变更，更新相关文档"
        # 拆解为：
        # 1. Scanner.scan_api_changes()
        # 2. Generator.update_tech_docs()
        # 3. Syncer.sync_to_wiki()
        pass

    def dispatch(self, subtask: SubTask) -> Agent:
        """分发给合适的 Agent"""
        pass

    def integrate(self, results: List[Result]) -> FinalOutput:
        """整合各 Agent 结果"""
        pass
```

#### Scanner（扫描者）
```python
class Scanner:
    """监控代码与文档状态"""

    tools = ["bash", "git", "glob", "grep", "lsp"]

    def detect_code_changes(self) -> ChangeReport:
        """检测代码变更
        - API 接口变更
        - 配置文件变更
        - 数据库 Schema 变更
        """
        pass

    def check_doc_freshness(self) -> DocStatus:
        """检查文档新鲜度
        - 最后更新时间
        - 关联代码是否变更
        - 版本号是否匹配
        """
        pass

    def identify_impact_scope(self, change: Change) -> ImpactScope:
        """识别影响范围
        - 哪些文档需要更新
        - 哪些团队需要通知
        """
        pass
```

#### Generator（生成者）
```python
class Generator:
    """生成各类文档内容"""

    tools = ["read", "write", "skill", "task"]

    def generate_tech_doc(self, code_path: str) -> DocContent:
        """生成技术文档
        - API 文档
        - 架构文档
        - 部署文档
        """
        pass

    def generate_product_doc(self, feature: Feature) -> DocContent:
        """生成产品文档
        - 功能说明
        - 使用手册
        - 变更日志
        """
        pass

    def generate_user_guide(self, feature: Feature) -> DocContent:
        """生成用户文档
        - 操作指南
        - FAQ
        - 视频脚本
        """
        pass

    def generate_ops_doc(self, service: Service) -> DocContent:
        """生成运维文档
        - 监控配置
        - 应急预案
        - 上线检查单
        """
        pass
```

#### Syncer（同步者）
```python
class Syncer:
    """维护多源文档一致性"""

    tools = ["bash", "read", "write", "web"]

    def sync_to_wiki(self, content: DocContent, target: WikiTarget):
        """同步到 Wiki"""
        # 调用 wiki-auto.sh 或 MCP
        pass

    def sync_to_git(self, content: DocContent, target: GitTarget):
        """同步到 Git"""
        # 更新 docs/ 目录
        pass

    def sync_to_notion(self, content: DocContent, target: NotionTarget):
        """同步到 Notion（产品团队可能用）"""
        pass

    def resolve_conflict(self, sources: List[Source]) -> Resolution:
        """解决多源冲突"""
        pass
```

## 3. 文档类型映射

| 文档类型 | 存储位置 | 生成来源 | 更新触发 | 责任人 |
|---------|---------|---------|---------|--------|
| **API 文档** | Wiki + Git | 代码注释/OpenAPI | API 变更 | Scanner → Generator |
| **架构文档** | Wiki | 代码结构/配置 | 架构变更 | Scanner → Generator |
| **产品文档** | Wiki + Notion | PRD/Feature 描述 | 需求变更 | Generator |
| **用户手册** | Wiki + GitBook | 产品文档 + 截图 | 功能发布 | Generator |
| **版本更新** | Wiki + Git Release | Git 提交记录 | 版本发布 | Scanner → Generator |
| **运维手册** | Wiki + 内部系统 | 监控配置/部署脚本 | 配置变更 | Scanner → Generator |

## 4. 典型工作流

### 场景 1：API 变更自动更新文档

```
1. Scanner.detect_code_changes()
   └─ 工具: git diff + grep API 定义
   └─ 发现: /api/v1/users 新增字段 "phone"

2. Scanner.identify_impact_scope()
   └─ 工具: grep + lsp
   └─ 影响: API文档、SDK文档、前端对接文档

3. Generator.generate_tech_doc()
   └─ 工具: read(代码) + skill(生成文档)
   └─ 生成: OpenAPI 规范更新说明

4. Syncer.sync_to_wiki()
   └─ 工具: bash(调用 wiki-auto.sh)
   └─ 同步: 更新 Wiki API 文档页面

5. Coordinator.notify()
   └─ 通知: 相关开发者确认
```

### 场景 2：版本发布自动生成更新日志

```
1. Scanner.scan_git_history()
   └─ 工具: bash(git log)
   └─ 获取: v1.2.0 到 v1.3.0 的所有提交

2. Generator.generate_changelog()
   └─ 工具: skill(分析提交分类)
   └─ 生成: 按 feat/fix/docs 分类的更新日志

3. Syncer.sync_multi_target()
   └─ 工具: write(Git Release) + bash(Wiki)
   └─ 同步: GitHub Release + Wiki 版本页面
```

### 场景 3：新功能开发文档前置

```
1. Coordinator.receive_request()
   └─ 输入: "开发文件上传功能，需要配套文档"

2. Generator.generate_doc_template()
   └─ 工具: skill(文档模板)
   └─ 生成: 技术文档 + 产品文档 + 用户手册 框架

3. Generator.fill_with_prd()
   └─ 工具: read(PRD) + skill(内容填充)
   └─ 填充: 根据 PRD 填充产品文档

4. Syncer.create_draft()
   └─ 工具: bash(wiki-auto.sh)
   └─ 创建: Wiki 草稿页面，标记"开发中"

5. Scanner.monitor_dev_progress()
   └─ 持续监控: 代码提交自动补充技术细节
```

## 5. 配置设计

```yaml
# config/wiki-agent.yaml

team:
  coordinator:
    model: "kimi-k2.5"
    max_iterations: 10

  scanner:
    enabled: true
    watch_paths:
      - "src/**/*.api.ts"
      - "docs/architecture/*.md"
      - "config/**/*.yaml"
    check_interval: 3600  # 每小时检查一次

  generator:
    enabled: true
    doc_types:
      - tech
      - product
      - user
      - ops
      - changelog

  syncer:
    enabled: true
    targets:
      wiki:
        enabled: true
        base_url: "https://wiki.tuhu.cn"
        space_key: "engineer"
        auth_method: "mcp"  # 或 "api"
      git:
        enabled: true
        docs_path: "docs/"
        auto_commit: false

consistency:
  rules:
    - name: "api_doc_sync"
      trigger: "api_change"
      actions: ["update_wiki", "notify_owner"]
    - name: "version_changelog"
      trigger: "tag_push"
      actions: ["generate_changelog", "create_release"]
```

## 6. 工具接口设计

所有 Agent 通过统一接口调用工具：

```python
class ToolExecutor:
    """工具执行器"""

    async def execute(self, tool_name: str, params: dict) -> Result:
        """执行工具"""
        tools = {
            "bash": self._exec_bash,
            "glob": self._exec_glob,
            "read": self._exec_read,
            "write": self._exec_write,
            "edit": self._exec_edit,
            "grep": self._exec_grep,
            "lsp": self._exec_lsp,
            "skill": self._exec_skill,
            "task": self._exec_task,
            "web": self._exec_web,
        }
        return await tools[tool_name](params)
```

这个架构设计是否符合你的预期？需要调整哪些部分？
