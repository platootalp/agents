from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from typing import TypedDict, Optional

from langgraph.prebuilt import create_react_agent

import util


# ------- Shared State 类型定义 -------
class AgentState(TypedDict):
    input: str
    schema_json_str: Optional[str]
    sql_code: Optional[str]
    java_code: Optional[str]


# ------- 工具函数与节点定义 -------
model = util.get_qwen_model()


@tool
def parse_schema_intent(input: str) -> str:
    """将自然语言解析成 schema JSON"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个数据库设计专家，请将以下业务需求转化为建表结构的 JSON 格式。你只需要给出JSON结果，不要做多余的解释。"),
        ("human", "{input}")
    ])
    chain = prompt | model | (lambda x: x.content)
    return chain.invoke({"input": input})


@tool
def generate_sql(schema_json_str: str) -> str:
    """根据 schema JSON 生成建表 SQL"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个资深的数据库工程师，请将以下 JSON 转换为标准的 MySQL 建表 SQL 语句。你只需要给出SQL语句，不要做多余的解释。"),
        ("human", "{schema_json_str}")
    ])
    chain = prompt | model | (lambda x: x.content)
    return chain.invoke({"schema_json_str": schema_json_str})


@tool
def generate_java_entity(schema_json_str: str) -> str:
    """根据 schema JSON 生成 Java 实体类"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 Java 开发专家，请根据以下 JSON 表结构生成对应的实体类代码，"
                   "使用的是Mybatis框架，使用Lombok和Mybatis相关注解。你只需要给出Java实体类，不要做多余的解释。"),
        ("human", "{schema_json_str}")
    ])
    chain = prompt | model | (lambda x: x.content)
    return chain.invoke({"schema_json_str": schema_json_str})


# ------- 节点函数实现 -------

def parse_node(state: AgentState):
    schema = parse_schema_intent.invoke(state["input"])
    return {"schema_json_str": schema, **state}


def sql_node(state: AgentState):
    sql = generate_sql.invoke(state["schema_json_str"])
    return {"sql_code": sql, **state}


def java_node(state: AgentState):
    java = generate_java_entity.invoke(state["schema_json_str"])
    return {"java_code": java, **state}


# ------- LangGraph 工作流构建 -------
def create_agent():
    graph = StateGraph(AgentState)
    graph.add_node("parse", parse_node)
    graph.add_node("sql", sql_node)
    graph.add_node("java", java_node)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "sql")
    graph.add_edge("sql", "java")
    graph.add_edge("java", END)

    return graph.compile()


# ------- 示例运行 -------
if __name__ == "__main__":
    user_input = "我们需要一个名为 user_profile 的表，用于存储用户信息，包括用户 ID（主键）、姓名、邮箱地址、注册时间、状态等。"
    result = create_agent().invoke({"input": user_input})
    print("\n--- SQL ---\n")
    print(result["sql_code"])
    print("\n--- Java Entity ---\n")
    print(result["java_code"])
