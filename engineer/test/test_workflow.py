from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph


# 定义一个用于更新状态的reducer函数
def reducer(a: list, b: int | None) -> list:
    print(f"Reducer called: a={a}, b={b}")
    if b is not None:
        return a + [b]
    return a


# 定义状态类
class State(TypedDict):
    x: Annotated[list, reducer]  # x是一个list类型，经过reducer处理


# 定义配置类
class ConfigSchema(TypedDict):
    r: float  # 配置项r是一个float类型
    node_name: str  # 节点名称


# 初始化状态图（StateGraph）并传入State类型和ConfigSchema
graph = StateGraph(state_schema=State, config_schema=ConfigSchema)


# 定义节点处理函数
def node(state: State, config: RunnableConfig) -> dict:
    r = config["configurable"].get("r", 1.0)  # 从配置中获取r的值，默认1.0
    x = state["x"][-1]  # 获取状态中的x列表的最后一个值
    # 获取当前节点名称（假设通过传递一个节点名参数）
    node_name = config.get("node_name", "Unknown Node")  # 默认值为"Unknown Node"

    print(f"Running node: {node_name}")  # 打印当前节点名称
    next_value = x * r  # 根据公式计算新的x值
    return {"x": next_value}  # 返回新的x值


# 向图中添加节点
graph.add_node(node="A", action=node)
graph.add_node(node)
graph.add_node("B", node)
graph.add_edge("A", "B")
graph.add_sequence([("C", node), ("D", node)])
graph.add_edge("B", "C")

# 设置图的入口和结束点
graph.set_entry_point("A")
graph.set_finish_point("D")


# try:
#     display(Image(compiled.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass

# 编译图
compiled = graph.compile()

# 打印配置规范
print(compiled.config_specs)

# 使用输入状态和配置执行图，并打印结果
step1 = compiled.invoke({"x": 1}, {"configurable": {"r": 3.0}})
