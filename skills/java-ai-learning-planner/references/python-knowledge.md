# Python知识体系

## 目录
- [Java vs Python对比](#java-vs-python对比)
- [Python核心语法](#python核心语法)
- [面向对象编程](#面向对象编程)
- [异步编程](#异步编程)
- [常用标准库](#常用标准库)
- [数据科学栈](#数据科学栈)
- [Web开发框架](#web开发框架)
- [AI/ML常用库](#aiml常用库)
- [学习阶段规划](#学习阶段规划)
- [推荐资源](#推荐资源)

## Java vs Python对比

### 语法差异
| 特性 | Java | Python |
|------|------|--------|
| 类型系统 | 静态类型，强类型 | 动态类型，强类型（可选类型提示） |
| 代码结构 | 类必须包含在文件中 | 缩进定义代码块 |
| 内存管理 | 手动管理，垃圾回收 | 自动垃圾回收 |
| 多线程 | 线程模型，GIL限制 | 协程模型，async/await |
| 异常处理 | 必须声明checked异常 | 不强制声明 |

### Java经验迁移点
- **面向对象思维**：类、继承、多态概念完全适用
- **设计模式**：GoF设计模式在Python中同样有效
- **测试驱动**：unittest、pytest对应JUnit
- **依赖注入**：可通过依赖注入框架实现（如dependency_injector）
- **日志框架**：logging模块对应Log4j/SLF4J

### Python特有优势
- **语法简洁**：代码量通常比Java少30-50%
- **动态特性**：元编程、装饰器、上下文管理器
- **生态丰富**：PyPI拥有30万+第三方包
- **AI优先**：TensorFlow、PyTorch等主流框架首选Python

## Python核心语法

### 基础语法（1-2周）
```python
# 变量和类型（动态类型）
name = "Python"
age = 30
is_active = True
numbers = [1, 2, 3, 4, 5]

# 控制流
if age > 18:
    print("Adult")
elif age > 12:
    print("Teenager")
else:
    print("Child")

# 循环
for num in numbers:
    print(num)

# 字典推导（Pythonic）
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### 函数定义
```python
def greet(name: str, greeting: str = "Hello") -> str:
    """问候函数"""
    return f"{greeting}, {name}!"

# 可变参数
def sum_all(*numbers: int) -> int:
    return sum(numbers)

# 关键字参数
def create_person(**kwargs):
    return kwargs
```

### 类与对象（1周）
```python
from dataclasses import dataclass
from typing import Optional

@dataclass  # 相当于Java的Lombok @Data
class Person:
    name: str
    age: int
    email: Optional[str] = None

    def greet(self) -> str:
        return f"Hello, I'm {self.name}"

class Student(Person):
    def __init__(self, name: str, age: int, grade: int):
        super().__init__(name, age)
        self.grade = grade

    def study(self) -> None:
        print(f"{self.name} is studying in grade {self.grade}")
```

## 面向对象编程

### 高级特性（1-2周）
```python
# 装饰器（类似Java注解，但运行时行为）
def log_execution(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@log_execution
def add(a: int, b: int) -> int:
    return a + b

# 上下文管理器（类似Java try-with-resources）
class FileHandler:
    def __enter__(self):
        print("Opening file")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing file")

with FileHandler():
    print("Processing file")
```

## 异步编程

### 协程和asyncio（2-3周）
```python
import asyncio
from typing import List

# 异步函数
async def fetch_data(url: str) -> str:
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # 模拟IO
    return f"Data from {url}"

# 并发执行
async def fetch_multiple(urls: List[str]) -> List[str]:
    tasks = [fetch_data(url) for url in urls]
    return await asyncio.gather(*tasks)

# 异步上下文管理器
async def main():
    urls = ["url1", "url2", "url3"]
    results = await fetch_multiple(urls)
    print(results)

asyncio.run(main())
```

### 对比Java并发
- **Java**：线程池、Future、CompletableFuture
- **Python**：事件循环、协程、async/await
- **关键差异**：Python的GIL限制CPU多线程，但协程适合高IO场景（如API调用、数据库查询）

## 常用标准库

### 核心库（1周）
```python
# JSON处理
import json
data = {"name": "Python", "version": 3.9}
json_str = json.dumps(data)
parsed = json.loads(json_str)

# HTTP请求
import urllib.request
with urllib.request.urlopen("https://api.example.com") as response:
    data = response.read()

# 正则表达式
import re
pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
emails = re.findall(pattern, text)

# 日期时间
from datetime import datetime, timedelta
now = datetime.now()
next_week = now + timedelta(days=7)
```

## 数据科学栈

### 核心库（2-3周）
```python
# NumPy（数值计算）
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
mean = np.mean(arr)  # 平均值

# Pandas（数据分析）
import pandas as pd
df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
df.to_csv("data.csv", index=False)

# Matplotlib（可视化）
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.show()
```

## Web开发框架

### FastAPI（2-3周）
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    description: str = None

@app.post("/items/")
async def create_item(item: Item):
    # 自动数据验证
    return {"item": item}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id < 1:
        raise HTTPException(status_code=400, detail="Invalid ID")
    return {"item_id": item_id}
```

### 对比Spring Boot
- **路由**：@Path对应@app.get
- **依赖注入**：@Autowired对应FastAPI的Depends
- **数据验证**：@Valid对应Pydantic模型
- **异步**：Java用WebFlux，Python原生支持async

## AI/ML常用库

### LangChain基础（3-4周）
```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model="gpt-4")

response = llm.invoke([HumanMessage(content="Hello")])
print(response.content)
```

### 向量数据库
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    ["Hello world", "Python programming"],
    embeddings
)
```

## 学习阶段规划

### 阶段1：Python基础（2-3周）
- **目标**：掌握Python核心语法，能编写简单程序
- **内容**：
  - 基本语法、控制流、数据结构（1周）
  - 函数、类、模块化（1周）
  - 标准库使用（1周）
- **实践任务**：
  - 编写一个命令行待办事项应用
  - 实现一个简单的文本处理工具
- **Java经验复用**：
  - 类和继承概念直接迁移
  - 测试框架对应JUnit使用方式

### 阶段2：进阶特性（2-3周）
- **目标**：掌握Python高级特性，理解异步编程
- **内容**：
  - 装饰器、上下文管理器、元类（1周）
  - 异步编程async/await（1-2周）
  - 类型提示和类型检查（1周）
- **实践任务**：
  - 实现一个异步HTTP客户端
  - 编写装饰器用于性能监控
- **Java经验复用**：
  - 异步思维对比Java的CompletableFuture
  - 类型提示对比Java的泛型

### 阶段3：生态库（3-4周）
- **目标**：掌握常用第三方库，能处理数据和构建API
- **内容**：
  - 数据处理（NumPy、Pandas）（1周）
  - Web开发（FastAPI）（1-2周）
  - 测试框架（pytest）（1周）
- **实践任务**：
  - 构建一个RESTful API
  - 实现数据分析pipeline
- **Java经验复用**：
  - FastAPI对比Spring Boot
  - pytest对比JUnit

### 阶段4：AI/ML基础（4-6周）
- **目标**：掌握AI开发基础库，能调用大模型API
- **内容**：
  - 大模型API调用（OpenAI、其他模型）（2周）
  - LangChain基础（2周）
  - 向量数据库和RAG（2周）
- **实践任务**：
  - 构建一个基于大模型的问答系统
  - 实现文档检索应用
- **Java经验复用**：
  - API设计经验直接应用
  - 数据库知识迁移到向量数据库

## 推荐资源

### 官方文档
- Python官方文档：https://docs.python.org/3/
- FastAPI文档：https://fastapi.tiangolo.com/
- LangChain文档：https://python.langchain.com/

### 在线课程
- Python for Java Developers（Coursera）
- Real Python（https://realpython.com/）
- FastAPI官方教程

### 实践平台
- LeetCode（Python版）
- Kaggle（数据科学）
- Hugging Face（模型和datasets）

### 书籍推荐
- 《Fluent Python》（进阶必读）
- 《Effective Python》（最佳实践）
- 《Python Crash Course》（快速入门）
