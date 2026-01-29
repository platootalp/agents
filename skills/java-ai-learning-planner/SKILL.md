---
name: java-ai-learning-planner
description: 为转行AI Agent的Java工程师提供个性化学习计划，涵盖Python、大模型、Agent开发、架构设计及项目实践，根据开发经验和Python基础定制学习路径
---

# AI Agent学习计划生成器

## 任务目标
- 本Skill用于：为从Java开发转行到AI Agent开发的工程师生成结构化、个性化的学习计划
- 能力包含：
  - 评估用户当前技术栈和学习背景
  - 生成Python、大模型、Agent、架构、实践五大领域的渐进式学习路径
  - 提供时间规划和学习资源推荐
- 触发条件：用户请求制定AI Agent学习计划、询问如何从Java转向AI Agent开发

## 操作步骤

### 1. 收集用户背景信息
通过以下问题了解用户情况（根据回答深入追问）：
- **Java开发经验**：年限、主要技术栈（Spring Boot、微服务等）、项目类型
- **Python基础**：是否写过Python、熟悉程度、使用场景
- **AI认知水平**：是否了解大模型API、是否使用过AI工具、对Agent的理解程度
- **学习目标**：短期（3个月）、中期（6个月）、长期目标
- **时间投入**：每周可用的学习时间
- **偏好方向**：偏向应用开发、底层架构、还是算法研究

### 2. 读取知识框架
根据用户关注领域，读取以下参考文档：
- [references/python-knowledge.md](references/python-knowledge.md) - Python知识体系
- [references/llm-knowledge.md](references/llm-knowledge.md) - 大模型技术栈
- [references/agent-knowledge.md](references/agent-knowledge.md) - Agent开发知识
- [references/architecture-knowledge.md](references/architecture-knowledge.md) - 项目架构设计
- [references/project-practice.md](references/project-practice.md) - 项目实践指南

### 3. 生成个性化学习计划
基于references中的知识框架和用户背景，生成包含以下内容的学习计划：

#### 计划结构
```markdown
# 个性化AI Agent学习计划

## 学习者画像
- Java开发经验：[年限/技术栈]
- Python基础：[描述]
- AI认知水平：[描述]
- 学习目标：[目标]
- 时间投入：[每周X小时]

## 学习路径总览
- 第一阶段：基础夯实（X周）
- 第二阶段：大模型应用（X周）
- 第三阶段：Agent开发（X周）
- 第四阶段：架构进阶（X周）
- 第五阶段：项目实战（X周）

## 详细学习计划
[按阶段展开，每个阶段包含]
### 第X阶段：[阶段名称]
**学习目标**：[目标描述]
**核心内容**：
- [知识点1] - [预计时间]
- [知识点2] - [预计时间]
**学习资源**：[从references中推荐]
**实践任务**：[具体可执行的任务]
**Java经验复用**：[说明哪些Java经验可以迁移]
```

#### 个性化定制要点
- **Python学习**：根据Java背景，对比讲解异同点（如类型系统、并发模型）
- **大模型学习**：从简单的API调用开始，逐步深入到Prompt工程和RAG
- **Agent开发**：结合微服务经验讲解Multi-Agent架构设计
- **架构设计**：对比传统后端架构与AI应用架构的差异
- **项目实践**：设计从简单的Chatbot到复杂的Multi-Agent系统渐进式项目

### 4. 计划调整和优化
- 根据用户反馈调整学习阶段顺序和时间分配
- 针对用户薄弱领域增加练习资源
- 提供学习进度的检查点和评估方式

## 资源索引
- 领域参考：
  - [Python知识体系](references/python-knowledge.md)（生成学习计划时必须参考）
  - [大模型技术栈](references/llm-knowledge.md)（生成学习计划时必须参考）
  - [Agent开发知识](references/agent-knowledge.md)（生成学习计划时必须参考）
  - [项目架构设计](references/architecture-knowledge.md)（生成学习计划时必须参考）
  - [项目实践指南](references/project-practice.md)（生成学习计划时必须参考）

## 注意事项
- 充分利用用户的Java开发经验，强调可迁移的编程思维和架构能力
- 学习计划要符合渐进式原则，避免一次性引入过多概念
- 每个阶段都要包含可执行的实践任务，确保学有所获
- 推荐的学习资源要兼顾理论学习和动手实践
- 保持计划的灵活性，根据用户反馈及时调整

## 使用示例

### 示例1：有3年Java经验的工程师
**用户描述**：3年Spring Boot开发经验，熟悉微服务架构，Python基础为零，想转向AI Agent开发，每周学习10小时

**生成计划要点**：
- 第一阶段重点学习Python基础（对比Java语法）
- 利用微服务经验快速理解Multi-Agent架构
- 项目实践从简单的工具调用Agent开始

### 示例2：有Python基础的Java工程师
**用户描述**：5年Java架构师经验，Python写过数据分析脚本，了解大模型API，想构建生产级Agent应用，每周学习8小时

**生成计划要点**：
- Python快速复习，重点放在异步编程和类型提示
- 直接进入Agent框架学习（LangChain、AutoGPT等）
- 架构部分重点学习RAG、向量数据库、工作流编排
- 项目实践设计生产级多Agent系统
