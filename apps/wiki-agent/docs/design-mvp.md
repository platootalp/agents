# Wiki Agent - MVP 设计文档

## 1. 一期目标

**一句话描述**：让 LLM 操作远程 Wiki 文档像操作本地文件一样简单。

### 核心能力

| 操作 | 本地文件 | Wiki 文档 | 说明 |
|------|---------|-----------|------|
| 读取 | `read file.md` | `read wiki://page` | 统一接口 |
| 写入 | `write file.md` | `write wiki://page` | 自动创建或更新 |
| 编辑 | `edit file.md` | `edit wiki://page` | 查找替换 |
| 搜索 | `glob *.md` | `search wiki://query` | 支持全文搜索 |
| 删除 | `rm file.md` | `delete wiki://page` | 软删除 |

## 2. 极简架构

```
┌─────────────────────────────────────────┐
│             User / LLM                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│        DocumentManager（统一入口）        │
│                                          │
│  • open(path) → Document                │
│  • read() → content                     │
│  • write(content)                       │
│  • edit(old, new)                       │
│  • delete()                             │
│  • search(query) → List[Document]       │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐ ┌──────────────┐
│LocalProvider │ │ WikiProvider │
│（本地文件）   │ │ （Wiki API） │
├──────────────┤ ├──────────────┤
│• Path操作    │ │• WikiAdapter │
│• Git版本     │ │• REST API    │
└──────────────┘ └──────────────┘
```

## 3. 核心抽象

### 3.1 Document（统一文档对象）

```python
class Document:
    """统一文档对象 - 本地文件和 Wiki 页面都封装成这个"""

    def __init__(self, path: str, provider: Provider):
        self.path = path          # 路径标识：file:///path 或 wiki://space/page
        self.provider = provider  # 提供方：LocalProvider 或 WikiProvider
        self._content = None      # 缓存内容
        self._metadata = None     # 元数据

    def read(self) -> str:
        """读取文档内容"""
        if self._content is None:
            self._content = self.provider.read(self.path)
        return self._content

    def write(self, content: str) -> None:
        """写入文档内容"""
        self.provider.write(self.path, content)
        self._content = content

    def edit(self, old_str: str, new_str: str) -> bool:
        """查找替换编辑"""
        content = self.read()
        if old_str not in content:
            return False
        new_content = content.replace(old_str, new_str, 1)
        self.write(new_content)
        return True

    def delete(self) -> None:
        """删除文档"""
        self.provider.delete(self.path)

    @property
    def metadata(self) -> Dict:
        """文档元数据（标题、作者、更新时间等）"""
        if self._metadata is None:
            self._metadata = self.provider.get_metadata(self.path)
        return self._metadata

    def __repr__(self):
        return f"Document({self.path})"
```

### 3.2 DocumentManager（统一入口）

```python
class DocumentManager:
    """文档管理器 - LLM 和用户的统一操作入口"""

    def __init__(self, config: Config):
        self.providers = {
            "file": LocalProvider(config.local_root),
            "wiki": WikiProvider(config.wiki_config),
        }

    def open(self, path: str) -> Document:
        """
        打开文档

        支持的路径格式：
        - file:///path/to/file.md     → 本地文件
        - wiki://space/page-title     → Wiki 页面
        - ./relative/path.md          → 本地文件（简写）
        - space/page-title            → Wiki 页面（简写）
        """
        # 解析路径
        if path.startswith("file://"):
            provider_name = "file"
            actual_path = path[7:]
        elif path.startswith("wiki://"):
            provider_name = "wiki"
            actual_path = path[7:]
        elif path.startswith("./") or path.startswith("/"):
            provider_name = "file"
            actual_path = path
        else:
            # 默认视为 Wiki 路径
            provider_name = "wiki"
            actual_path = path

        provider = self.providers[provider_name]
        return Document(f"{provider_name}://{actual_path}", provider)

    def search(self, query: str, source: str = "all") -> List[Document]:
        """
        搜索文档

        Args:
            query: 搜索关键词
            source: "file" | "wiki" | "all"
        """
        results = []

        if source in ("file", "all"):
            file_results = self.providers["file"].search(query)
            results.extend(file_results)

        if source in ("wiki", "all"):
            wiki_results = self.providers["wiki"].search(query)
            results.extend(wiki_results)

        return results

    def create(self, path: str, content: str = "") -> Document:
        """创建新文档"""
        doc = self.open(path)
        doc.write(content)
        return doc
```

### 3.3 Provider 接口

```python
from abc import ABC, abstractmethod

class Provider(ABC):
    """文档提供方抽象基类"""

    @abstractmethod
    def read(self, path: str) -> str:
        pass

    @abstractmethod
    def write(self, path: str, content: str) -> None:
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        pass

    @abstractmethod
    def search(self, query: str) -> List[str]:
        """返回匹配的 path 列表"""
        pass

    @abstractmethod
    def get_metadata(self, path: str) -> Dict:
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        pass
```

## 4. WikiProvider 实现（基于现有脚本）

```python
class WikiProvider(Provider):
    """Wiki 文档提供方 - 封装 wiki-auto.sh"""

    def __init__(self, config: WikiConfig):
        self.base_url = config.base_url
        self.space_key = config.space_key
        self.script_path = config.wiki_auto_script

    def read(self, path: str) -> str:
        """
        读取 Wiki 页面内容

        path 格式: "space/page-title" 或 "page-id"
        """
        # 如果是数字，视为 pageId
        if path.isdigit():
            page_id = path
        else:
            # 先搜索获取 pageId
            page_id = self._get_page_id(path)

        # 调用 wiki-auto.sh get 命令
        result = self._exec_script("get", {"page_id": page_id})
        return self._parse_content(result)

    def write(self, path: str, content: str) -> None:
        """
        写入 Wiki 页面

        如果页面存在则更新，不存在则创建
        """
        # 解析路径
        parts = path.split("/", 1)
        if len(parts) == 2:
            space, title = parts
        else:
            space = self.space_key
            title = parts[0]

        # 检查页面是否存在
        existing_id = self._find_page_id(space, title)

        if existing_id:
            # 更新
            self._exec_script("update", {
                "page_id": existing_id,
                "content": content
            })
        else:
            # 创建
            self._exec_script("create", {
                "space": space,
                "title": title,
                "content": content
            })

    def search(self, query: str) -> List[str]:
        """搜索 Wiki 页面"""
        result = self._exec_script("search", {"query": query})
        return self._parse_search_results(result)

    def delete(self, path: str) -> None:
        """删除 Wiki 页面（软删除）"""
        page_id = self._get_page_id(path)
        self._exec_script("delete", {"page_id": page_id})

    def exists(self, path: str) -> bool:
        """检查页面是否存在"""
        try:
            self._get_page_id(path)
            return True
        except PageNotFoundError:
            return False

    def _exec_script(self, action: str, params: Dict) -> str:
        """执行 wiki-auto.sh"""
        cmd_parts = ["bash", self.script_path, f"--{action}"]
        for key, value in params.items():
            cmd_parts.extend([f"--{key}", str(value)])

        # 使用 ToolExecutor 执行
        result = tool_executor.execute("bash", {
            "command": " ".join(cmd_parts),
            "timeout": 60
        })

        if not result.success:
            raise WikiError(f"Wiki operation failed: {result.error}")

        return result.data["stdout"]
```

## 5. 使用示例

### 5.1 LLM 使用示例

```python
# 初始化
doc_manager = DocumentManager(config)

# 1. 读取本地文档
local_doc = doc_manager.open("./docs/api.md")
content = local_doc.read()

# 2. 读取 Wiki 文档（方式1：完整路径）
wiki_doc = doc_manager.open("wiki://engineer/API文档")
wiki_content = wiki_doc.read()

# 3. 读取 Wiki 文档（方式2：简写，默认使用配置的 space）
wiki_doc2 = doc_manager.open("API文档")

# 4. 搜索文档
results = doc_manager.search("FileUpload", source="wiki")
for doc in results:
    print(f"Found: {doc.path}")

# 5. 创建 Wiki 文档
new_doc = doc_manager.create(
    "wiki://engineer/新功能文档",
    content="# 新功能\n\n这是内容"
)

# 6. 编辑 Wiki 文档
doc = doc_manager.open("API文档")
success = doc.edit(
    old_str="旧版本：v1.0",
    new_str="新版本：v2.0"
)

# 7. 本地和 Wiki 同步
local_api_doc = doc_manager.open("./docs/api.md")
wiki_api_doc = doc_manager.open("wiki://API文档")

# 将本地文档同步到 Wiki
wiki_api_doc.write(local_api_doc.read())
```

### 5.2 CLI 使用示例

```bash
# 读取 Wiki 页面
wiki-agent read "API文档"

# 创建新页面
wiki-agent create "新功能/设计文档" --content "# 设计\n\n内容"

# 从文件创建
wiki-agent create "新功能/设计文档" --file ./design.md

# 编辑页面（查找替换）
wiki-agent edit "API文档" --old "v1.0" --new "v2.0"

# 搜索页面
wiki-agent search "FileUpload"

# 同步本地到 Wiki
wiki-agent sync ./docs/api.md --to wiki://API文档

# 批量同步
wiki-agent sync ./docs/ --to wiki://docs/ --pattern "*.md"
```

## 6. 项目结构

```
wiki-agent/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── document.py         # Document 类
│   │   ├── manager.py          # DocumentManager 类
│   │   └── provider.py         # Provider 抽象基类
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── local.py            # LocalProvider
│   │   └── wiki.py             # WikiProvider
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py             # CLI 入口
│   └── utils/
│       ├── __init__.py
│       └── tool_executor.py    # 工具执行器
├── wiki-auto.sh                # 现有脚本
├── config.yaml                 # 配置文件
├── requirements.txt
└── tests/
    └── test_document.py
```

## 7. 配置设计

```yaml
# config.yaml

# Wiki 配置
wiki:
  base_url: "https://wiki.tuhu.cn"
  space_key: "engineer"
  auth:
    method: "mcp"  # 或 "api"
    # api 模式需要配置
    # username: "xxx"
    # password: "xxx"

# 本地配置
local:
  root: "./"

# 脚本路径
scripts:
  wiki_auto: "./wiki-auto.sh"

# LLM 配置（可选，用于智能功能）
llm:
  provider: "openai"  # 或 "kimi", "deepseek"
  model: "gpt-4"
  api_key: "${LLM_API_KEY}"
```

## 8. 一期实现计划

### Week 1: 核心骨架
- [ ] Document 类实现
- [ ] Provider 抽象接口
- [ ] LocalProvider 实现
- [ ] ToolExecutor 工具层

### Week 2: Wiki 集成
- [ ] WikiProvider 实现（封装 wiki-auto.sh）
- [ ] DocumentManager 统一入口
- [ ] 路径解析逻辑
- [ ] 基础 CLI 命令（read/write/create）

### Week 3: 功能完善
- [ ] search 功能实现
- [ ] edit 功能实现
- [ ] delete 功能实现
- [ ] 批量操作支持

### Week 4: 测试 & 优化
- [ ] 单元测试
- [ ] 集成测试（对接真实 Wiki）
- [ ] 错误处理优化
- [ ] 文档编写

## 9. 与 v2 扩展设计的衔接

MVP 实现后，可以平滑扩展到 v2 设计：

```
MVP DocumentManager ──▶ 作为 v2 Agent 的基础工具
      │
      ▼
v2 Monitor Agent ────▶ 使用 DocumentManager 读取/监控
v2 Generator Agent ──▶ 使用 DocumentManager 写入文档
v2 Coordinator Agent ─▶ 使用 DocumentManager 协调操作
```

MVP 的 DocumentManager 就是 v2 中各 Agent 使用的底层工具。

---

这个 MVP 设计是否符合预期？有什么需要调整的吗？
