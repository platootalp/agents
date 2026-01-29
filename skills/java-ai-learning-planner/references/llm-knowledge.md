# 大模型技术栈

## 目录
- [大模型基础概念](#大模型基础概念)
- [主流大模型平台](#主流大模型平台)
- [API调用基础](#api调用基础)
- [提示工程](#提示工程)
- [RAG（检索增强生成）](#rag检索增强生成)
- [微调与训练](#微调与训练)
- [多模态应用](#多模态应用)
- [学习阶段规划](#学习阶段规划)
- [推荐资源](#推荐资源)

## 大模型基础概念

### 核心概念
- **LLM（Large Language Model）**：大规模语言模型，通过海量文本训练的神经网络
- **Transformer架构**：当前主流大模型的底层架构，基于自注意力机制
- **Token**：文本的最小处理单位，约等于半个单词
- **Context Window**：模型能处理的最大输入长度
- **Temperature**：控制输出随机性的参数（0=确定性，1=创造性）

### 关键参数对比
| 参数 | 含义 | 推荐值 |
|------|------|--------|
| Temperature | 输出随机性 | 0.0-0.3（任务型），0.7-1.0（创意型） |
| Max Tokens | 最大输出长度 | 根据需求调整 |
| Top P | 核采样 | 0.9-1.0 |
| Frequency Penalty | 避免重复 | 0-0.5 |

## 主流大模型平台

### 国际平台
1. **OpenAI**
   - 模型：GPT-4、GPT-3.5、GPT-4o
   - 优势：能力强、生态完善
   - 成本：较高

2. **Anthropic**
   - 模型：Claude 3.5、Claude 3
   - 优势：长文本、安全性
   - 成本：中等

3. **Google**
   - 模型：Gemini Pro、Gemini Ultra
   - 优势：多模态能力强
   - 成本：中等

### 国内平台
1. **智谱AI**
   - 模型：GLM-4、ChatGLM系列
   - 优势：中文能力强，开源生态

2. **阿里云**
   - 模型：Qwen系列
   - 优势：企业服务、多语言

3. **百度文心**
   - 模型：ERNIE系列
   - 优势：中文理解、企业级

4. **字节跳动**
   - 模型：豆包系列
   - 优势：性价比高

## API调用基础

### OpenAI API示例
```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 简单对话
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手"},
        {"role": "user", "content": "解释什么是大语言模型"}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

### 流式输出
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "写一首关于Python的诗"}],
    stream=True  # 启用流式输出
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 国内API示例（智谱）
```python
import requests

def call_zhipu(prompt: str) -> str:
    api_key = os.getenv("ZHIPU_API_KEY")
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "glm-4",
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    return response.json()["choices"][0]["message"]["content"]
```

## 提示工程

### 基础模式

#### 1. 角色设定
```
你是一个有10年经验的Java架构师，精通Spring Boot和微服务架构。
请帮我设计一个电商系统的技术架构方案。
```

#### 2. 少样本学习（Few-shot）
```
任务：将自然语言转换为SQL查询

示例1：
输入：查询所有年龄大于30的用户
输出：SELECT * FROM users WHERE age > 30

示例2：
输入：查询销售额最高的5个产品
输出：SELECT * FROM products ORDER BY sales DESC LIMIT 5

输入：查询上个月注册的用户
输出：
```

#### 3. 思维链（Chain of Thought）
```
问题：如果一个商店的苹果每个3元，橙子每个5元，小明买了3个苹果和2个橙子，他一共花了多少钱？

请一步步思考：
1. 计算苹果的总价
2. 计算橙子的总价
3. 计算总价
```

### 高级技巧

#### 结构化输出
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": """请以JSON格式输出，包含以下字段：
            {
                "summary": "简要总结",
                "key_points": ["要点1", "要点2"],
                "sentiment": "正面/负面/中性"
            }"""
        },
        {"role": "user", "content": "分析这段文本的情感和要点"}
    ]
)

import json
result = json.loads(response.choices[0].message.content)
```

#### 提示词模板
```python
from jinja2 import Template

prompt_template = """
你是一个{{role}}。
你的任务是{{task}}。

要求：
{% for req in requirements %}
- {{req}}
{% endfor %}

用户输入：
{{user_input}}
"""

def generate_prompt(role, task, requirements, user_input):
    template = Template(prompt_template)
    return template.render(
        role=role,
        task=task,
        requirements=requirements,
        user_input=user_input
    )
```

## RAG（检索增强生成）

### 基本架构
```
用户问题
    ↓
向量检索 → 相关文档片段
    ↓                ↓
    └────→ LLM生成 ←──┘
          ↓
      回答用户
```

### 向量数据库选择
| 数据库 | 优势 | 适用场景 |
|--------|------|----------|
| FAISS | 开源、高性能 | 本地嵌入 |
| Pinecone | 托管服务、易用 | 快速原型 |
| Chroma | 轻量级、本地优先 | 中小规模 |
| Milvus | 开源、功能全 | 企业级应用 |

### RAG实现示例
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

# 1. 加载文档
loader = TextLoader("docs/java-spring.txt")
documents = loader.load()

# 2. 创建向量索引
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# 3. 创建RAG链
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 4. 查询
query = "Spring Boot中如何配置数据库连接？"
result = qa_chain.invoke(query)
print(result["result"])
```

### RAG优化技巧
1. **文档分块**：合理切分文档（500-1000 tokens）
2. **混合检索**：向量检索+关键词检索
3. **重排序**：检索后对结果重新排序
4. **元数据过滤**：基于文档属性筛选

## 微调与训练

### LoRA微调（轻量级）
```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B")

# 配置LoRA
lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

# 应用LoRA
model = get_peft_model(model, lora_config)
```

### 微调场景
- **领域适配**：医疗、法律等专业领域
- **风格定制**：企业品牌调性
- **任务优化**：特定任务性能提升

### 工具推荐
- **训练**：LoRA、QLoRA（PEFT库）
- **数据处理**：datasets（Hugging Face）
- **训练框架**：Trainer（transformers）、DeepSpeed

## 多模态应用

### 图像理解
```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",  # 支持多模态
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图片的内容"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                }
            ]
        }
    ]
)
```

### 多模态应用场景
- **文档分析**：OCR+理解
- **图表解读**：数据可视化分析
- **图像生成**：根据文本生成图片
- **视频理解**：视频内容分析

## 学习阶段规划

### 阶段1：API基础（2-3周）
- **目标**：掌握大模型API调用，能实现简单对话应用
- **内容**：
  - 理解LLM核心概念（1周）
  - OpenAI/智谱API调用（1周）
  - 基础提示工程（1周）
- **实践任务**：
  - 实现一个简单的聊天机器人
  - 创建文本摘要工具
- **Java经验复用**：
  - HTTP客户端知识直接应用
  - 异常处理和日志记录

### 阶段2：提示工程（2-3周）
- **目标**：掌握高级提示技巧，优化模型输出质量
- **内容**：
  - Few-shot、CoT等高级技巧（1周）
  - 结构化输出（1周）
  - 提示词模板化（1周）
- **实践任务**：
  - 构建代码生成助手
  - 实现数据分析工具
- **Java经验复用**：
  - 模板引擎经验（如Thymeleaf）
  - 数据验证思维

### 阶段3：RAG应用（4-6周）
- **目标**：掌握检索增强生成，能构建知识库问答系统
- **内容**：
  - 向量数据库基础（1-2周）
  - 文档处理和分块（1-2周）
  - RAG链路优化（1-2周）
- **实践任务**：
  - 构建企业知识库问答
  - 实现文档智能分析系统
- **Java经验复用**：
  - 数据库知识迁移到向量库
  - 搜索引擎相关经验

### 阶段4：微调入门（3-4周）
- **目标**：了解微调原理，能进行简单的LoRA微调
- **内容**：
  - 微调原理和LoRA（1-2周）
  - 数据准备和预处理（1周）
  - 模型评估（1周）
- **实践任务**：
  - 微调一个小型模型
  - 对比微调前后效果
- **Java经验复用**：
  - 机器学习基础（如了解过）
  - 数据处理经验

## 推荐资源

### 官方文档
- OpenAI API文档：https://platform.openai.com/docs
- LangChain文档：https://python.langchain.com/
- Hugging Face：https://huggingface.co/docs

### 学习资源
- 吴恩达《Generative AI for Everyone》
- 《Prompt Engineering Guide》
- OpenAI Cookbook

### 实践平台
- OpenAI Playground
- Hugging Face Spaces
- Coze、Dify等Agent平台

### 书籍推荐
- 《Building LLM Applications for Production》
- 《Designing Machine Learning Systems》
- 《Prompt Engineering for Developers》
