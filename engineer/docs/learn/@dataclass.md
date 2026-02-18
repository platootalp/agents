`@dataclass` 是 Python 3.7+ 引入的一个装饰器（来自 `dataclasses` 模块），用于自动生成常见的类方法（如 `__init__`、`__repr__`、
`__eq__` 等），从而减少样板代码，让数据类的定义更简洁清晰。

---

### 基本用法

```python
from dataclasses import dataclass


@dataclass
class Person:
    name: str
    age: int
    email: str = ""
```

等价于手写：

```python
class Person:
    def __init__(self, name: str, age: int, email: str = ""):
        self.name = name
        self.age = age
        self.email = email

    def __repr__(self):
        return f"Person(name={self.name!r}, age={self.age!r}, email={self.email!r})"

    def __eq__(self, other):
        if not isinstance(other, Person):
            return NotImplemented
        return (self.name, self.age, self.email) == (other.name, other.age, other.email)
```

---

### 常用参数（`@dataclass(...)` 的选项）

| 参数                  | 说明                                        |
|---------------------|-------------------------------------------|
| `init=True`         | 是否生成 `__init__` 方法（默认是）                   |
| `repr=True`         | 是否生成 `__repr__` 方法（默认是）                   |
| `eq=True`           | 是否生成 `__eq__` 方法（默认是）                     |
| `order=False`       | 是否生成 `<`, `<=`, `>`, `>=` 方法（需 `eq=True`） |
| `unsafe_hash=False` | 控制是否生成 `__hash__` 方法                      |
| `frozen=False`      | 若为 `True`，则实例不可变（类似 `namedtuple`）         |
| `slots=False`       | （Python 3.10+）是否使用 `__slots__` 减少内存占用     |

示例：不可变数据类

```python
@dataclass(frozen=True)
class Point:
    x: float
    y: float
```

尝试修改属性会抛出 `FrozenInstanceError`。

---

### 字段定制：`field()`

使用 `field()` 可对字段进行更细粒度控制：

```python
from dataclasses import dataclass, field
from typing import List


@dataclass
class Student:
    name: str
    grades: List[int] = field(default_factory=list)
    id: int = field(init=False, default=0)  # 不在 __init__ 中传入
```

常用 `field` 参数：

- `default`：默认值（不可变对象）
- `default_factory`：可调用对象，用于生成默认值（如 `list`, `dict`）
- `init=False`：该字段不参与构造
- `repr=False`：不在 `__repr__` 中显示
- `compare=False`：不参与相等/排序比较

---

### 继承支持

`@dataclass` 支持继承，子类字段会排在父类字段之后：

```python
@dataclass
class Animal:
    species: str


@dataclass
class Dog(Animal):
    breed: str
```

创建 `Dog("Canis lupus", "Golden Retriever")` 是合法的。

---

### 与 `typing.NamedTuple` / `collections.namedtuple` 对比

- `@dataclass` 更灵活（支持可变、方法、继承等）
- `NamedTuple` 更轻量、不可变、支持元组解包
- 如需高性能且只存数据，可考虑 `NamedTuple`；否则推荐 `@dataclass`

---

### 注意事项

- 类型注解（type hints）是必须的（即使你不用类型检查）
- 默认值必须是**不可变对象**，否则所有实例共享同一个可变对象（如 `list`、`dict`）——此时应使用 `default_factory`
- 若需自定义 `__post_init__`，可在类中定义该方法，在 `__init__` 后自动调用

```python
@dataclass
class Circle:
    radius: float

    def __post_init__(self):
        if self.radius < 0:
            raise ValueError("Radius must be non-negative")
```


