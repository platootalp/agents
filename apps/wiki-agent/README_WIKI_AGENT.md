# Wiki Agent - Playwright MCP Agent

基于 Python + LangChain 1.0+ 的 Wiki 操作 Agent，使用 Playwright MCP 实现对 Wiki (Confluence) 的搜索、读取、创建、修改、列表等文件系统式操作。

## 特性

- **MCP Server**: 基于 FastMCP 的 Wiki 操作服务器
- **Playwright 驱动**: 通过浏览器自动化操作 Wiki
- **LangChain Agent**: 支持自然语言指令的智能 Agent
- **文件系统式 API**: 类似文件系统的操作接口 (search/read/create/update/list)
- **CLI 工具**: 命令行接口支持各种操作

## 项目结构

```
wiki-agent/
├── src/
│   ├── wiki_mcp_server.py      # MCP Server 实现 (Playwright 操作)
│   ├── wiki_mcp_client.py      # MCP Client 实现
│   ├── wiki_agent.py           # LangChain Agent 实现
│   ├── cli.py                  # 命令行接口
│   └── config.py               # 配置管理
├── examples/
│   └── example_usage.py        # 使用示例
├── requirements.txt            # 依赖
└── README_WIKI_AGENT.md        # 本文档
```

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt

# 安装 Playwright 浏览器
playwright install chromium
```

### 2. 初始化配置

```bash
cd src
python cli.py init
```

这会创建一个 `.env` 文件模板，编辑它设置你的配置：

```bash
# 设置 OpenAI API Key (Agent 模式需要)
export OPENAI_API_KEY=your-api-key

# 或使用 .env 文件
```

### 3. 启动 Chrome 远程调试

```bash
# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

# Linux
google-chrome --remote-debugging-port=9222

# Windows
chrome.exe --remote-debugging-port=9222
```

确保已登录 Wiki 后再使用 Agent。

## 使用方式

### 方式1: CLI 命令行

```bash
cd src

# 初始化配置
python cli.py init

# 搜索页面
python cli.py search "API 文档" --limit 5

# 读取页面
python cli.py read 659448790

# 创建页面
python cli.py create 12345678 "新页面标题" "<h1>内容</h1>"

# 更新页面
python cli.py update 12345678 --title "新标题" --content "<p>新内容</p>"

# 列出子页面
python cli.py list-children 12345678 --recursive

# Agent 模式 (自然语言)
python cli.py agent "搜索 FileUpload 文档并在其下创建一个测试页面"
```

### 方式2: Python 代码

```python
import asyncio
from wiki_mcp_client import WikiMCPClient

async def main():
    client = WikiMCPClient()

    # 搜索页面
    result = await client.search("API 文档")
    print(result)

    # 读取页面
    result = await client.read("12345678")
    print(result)

    # 创建页面
    result = await client.create(
        parent_id="12345678",
        title="新页面",
        content="<h1>标题</h1><p>内容</p>"
    )
    print(result)

    await client.close()

asyncio.run(main())
```

### 方式3: LangChain Agent

```python
import asyncio
from wiki_agent import WikiAgent

async def main():
    agent = WikiAgent()

    # 使用自然语言指令
    result = await agent.run(""
        "搜索 '部署文档'，找到后在其下创建一个标题为 '测试部署' 的页面，"
        "内容为 '<h1>测试部署文档</h1><p>这是自动创建的测试页面</p>'"
    """)

    print(result)
    await agent.close()

asyncio.run(main())
```

## MCP Tools 列表

| 工具名 | 描述 | 参数 |
|--------|------|------|
| `wiki_search` | 搜索 Wiki 页面 | query, space_key, limit |
| `wiki_read` | 读取页面内容 | page_id, include_metadata |
| `wiki_create` | 创建子页面 | parent_id, title, content, space_key |
| `wiki_update` | 更新页面 | page_id, title, content |
| `wiki_list_children` | 列出子页面 | page_id, recursive |
| `wiki_delete` | 删除页面 | page_id, confirm |
| `wiki_configure` | 配置客户端 | base_url, cdp_endpoint |

## 与原始 wiki-auto.sh 的对比

| 功能 | wiki-auto.sh (Bash) | Wiki Agent (Python) |
|------|---------------------|---------------------|
| 搜索 | ✅ curl API | ✅ Playwright 浏览器 |
| 读取 | ✅ curl API | ✅ Playwright 浏览器 |
| 创建 | ✅ API + Playwright | ✅ Playwright MCP |
| 更新 | ❌ | ✅ Playwright MCP |
| 删除 | ❌ | ✅ Playwright MCP |
| 列表 | ❌ | ✅ Playwright MCP |
| Agent 模式 | ❌ | ✅ LangChain |
| CLI | ✅ | ✅ 更丰富的命令 |

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API Key (Agent 模式需要) | - |
| `OPENAI_BASE_URL` | OpenAI Base URL | - |
| `OPENAI_MODEL` | 使用的模型 | gpt-4o |
| `WIKI_BASE_URL` | Wiki 基础 URL | https://wiki.tuhu.cn |
| `WIKI_SPACE_KEY` | 默认空间 Key | engineer |
| `WIKI_CDP_ENDPOINT` | Chrome CDP 端点 | http://localhost:9222 |

## 注意事项

1. **Chrome 远程调试**: 使用前必须先启动 Chrome 并开启远程调试端口
2. **Wiki 登录**: 确保 Chrome 中已登录 Wiki，Agent 会使用当前登录状态
3. **API Key**: Agent 模式需要 OpenAI API Key，直接客户端模式不需要
4. **HTML 内容**: 创建/更新页面时，内容必须是 HTML 格式

## 示例场景

### 场景1: 批量创建文档结构

```python
async def create_doc_structure():
    agent = WikiAgent()

    # 搜索父页面
    search_result = await agent.search("项目文档")

    # 批量创建子页面
    pages = ["需求文档", "设计文档", "API文档", "部署文档"]
    for page in pages:
        await agent.create(
            parent_id="12345678",
            title=page,
            content=f"<h1>{page}</h1><p>TODO: 补充内容</p>"
        )

    await agent.close()
```

### 场景2: 使用自然语言指令

```bash
python cli.py agent "找到 FileUpload 相关的文档，总结它们的内容，"
                   "然后创建一个汇总页面，列出所有的 FileUpload 文档链接"
```

### 场景3: 文档同步

```python
async def sync_docs():
    client = WikiMCPClient()

    # 读取本地文档
    with open("docs/api.md") as f:
        content = f.read()

    # 转换为 HTML (可以使用 markdown 库)
    html_content = markdown_to_html(content)

    # 更新 Wiki 页面
    await client.update(
        page_id="12345678",
        content=html_content
    )

    await client.close()
```

## 故障排查

### Chrome 连接失败

```
Error: 无法连接到 Chrome CDP
```

解决方案：
1. 确保 Chrome 已启动并开启远程调试：`chrome --remote-debugging-port=9222`
2. 检查端口是否正确：`curl http://localhost:9222/json/version`
3. 确保没有防火墙阻止

### MCP Server 启动失败

```
Error: mcp module not found
```

解决方案：
```bash
pip install mcp>=1.0.0
```

### Agent 无法找到工具

```
Error: Unknown tool: wiki_search
```

解决方案：
1. 确保 MCP Server 已正确启动
2. 检查工具名称拼写
3. 查看 MCP Server 日志

## 开发计划

- [ ] 支持 Markdown 到 HTML 自动转换
- [ ] 添加更多 Wiki 操作（移动页面、复制页面等）
- [ ] 支持批量操作和并发控制
- [ ] 添加页面模板功能
- [ ] 支持 Wiki 版本控制
- [ ] 添加更多 LLM 提供商支持

## License

MIT
