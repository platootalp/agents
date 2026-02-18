"""
实体记忆实现（长期记忆）
提取和存储对话中的实体信息
"""

# genAI_main_start
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import re
import copy

from .base_memory import (
    BaseMemory, BaseChatMessageHistory,
    MemoryVariables, MemoryType
)
from ..messages import BaseMessage, HumanMessage, AIMessage, Message
from .chat_history import InMemoryChatMessageHistory

if TYPE_CHECKING:
    from ..language_models import BaseChatModel


# 实体提取提示词模板
DEFAULT_ENTITY_EXTRACTION_PROMPT = """从以下对话中提取所有重要的实体（人名、地点、组织、产品等）。

对话内容：
{history}

请以JSON格式返回实体列表，格式如下：
{{"entities": ["实体1", "实体2", ...]}}

只返回JSON，不要其他内容。"""


# 实体摘要提示词模板
DEFAULT_ENTITY_SUMMARY_PROMPT = """根据以下对话内容，更新关于"{entity}"的描述信息。

当前已知信息：
{existing_info}

新的对话内容：
{history}

请根据新的对话内容，生成关于"{entity}"的简洁描述（如果没有新信息，返回"无更新"）："""


@dataclass
class EntityInfo:
    """实体信息
    
    存储单个实体的相关信息
    
    Attributes:
        name: 实体名称
        description: 实体描述
        mentions: 提及次数
        last_mentioned: 最后提及时间
        metadata: 额外元数据
    """
    name: str
    description: str = ""
    mentions: int = 1
    last_mentioned: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "mentions": self.mentions,
            "last_mentioned": self.last_mentioned.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityInfo":
        """从字典创建"""
        last_mentioned = data.get("last_mentioned")
        if isinstance(last_mentioned, str):
            last_mentioned = datetime.fromisoformat(last_mentioned)
        elif last_mentioned is None:
            last_mentioned = datetime.now()
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            mentions=data.get("mentions", 1),
            last_mentioned=last_mentioned,
            metadata=data.get("metadata", {})
        )


class EntityStore:
    """实体存储
    
    管理和存储实体信息
    """
    
    def __init__(self):
        """初始化实体存储"""
        self._entities: Dict[str, EntityInfo] = {}
    
    @property
    def entities(self) -> Dict[str, EntityInfo]:
        """获取所有实体"""
        return self._entities
    
    def get(self, name: str) -> Optional[EntityInfo]:
        """获取实体信息
        
        Args:
            name: 实体名称
        
        Returns:
            实体信息或None
        """
        # 不区分大小写查找
        name_lower = name.lower()
        for entity_name, entity in self._entities.items():
            if entity_name.lower() == name_lower:
                return entity
        return None
    
    def set(self, name: str, description: str, metadata: Optional[Dict] = None) -> None:
        """设置实体信息
        
        Args:
            name: 实体名称
            description: 实体描述
            metadata: 额外元数据
        """
        existing = self.get(name)
        if existing:
            existing.description = description
            existing.mentions += 1
            existing.last_mentioned = datetime.now()
            if metadata:
                existing.metadata.update(metadata)
        else:
            self._entities[name] = EntityInfo(
                name=name,
                description=description,
                metadata=metadata or {}
            )
    
    def update_mention(self, name: str) -> None:
        """更新实体提及
        
        Args:
            name: 实体名称
        """
        entity = self.get(name)
        if entity:
            entity.mentions += 1
            entity.last_mentioned = datetime.now()
        else:
            self._entities[name] = EntityInfo(name=name)
    
    def delete(self, name: str) -> bool:
        """删除实体
        
        Args:
            name: 实体名称
        
        Returns:
            是否删除成功
        """
        name_lower = name.lower()
        for entity_name in list(self._entities.keys()):
            if entity_name.lower() == name_lower:
                del self._entities[entity_name]
                return True
        return False
    
    def get_all_names(self) -> List[str]:
        """获取所有实体名称"""
        return list(self._entities.keys())
    
    def get_most_mentioned(self, k: int = 5) -> List[EntityInfo]:
        """获取最常提及的实体
        
        Args:
            k: 返回数量
        
        Returns:
            实体列表
        """
        sorted_entities = sorted(
            self._entities.values(),
            key=lambda x: x.mentions,
            reverse=True
        )
        return sorted_entities[:k]
    
    def get_recent(self, k: int = 5) -> List[EntityInfo]:
        """获取最近提及的实体
        
        Args:
            k: 返回数量
        
        Returns:
            实体列表
        """
        sorted_entities = sorted(
            self._entities.values(),
            key=lambda x: x.last_mentioned,
            reverse=True
        )
        return sorted_entities[:k]
    
    def clear(self) -> None:
        """清空所有实体"""
        self._entities.clear()
    
    def to_dict(self) -> Dict[str, Dict]:
        """导出为字典"""
        return {name: entity.to_dict() for name, entity in self._entities.items()}
    
    def from_dict(self, data: Dict[str, Dict]) -> None:
        """从字典导入"""
        self._entities = {
            name: EntityInfo.from_dict(entity_data)
            for name, entity_data in data.items()
        }
    
    def __len__(self) -> int:
        """返回实体数量"""
        return len(self._entities)
    
    def __contains__(self, name: str) -> bool:
        """检查实体是否存在"""
        return self.get(name) is not None


class ConversationEntityMemory(BaseMemory):
    """对话实体记忆
    
    从对话中提取实体并维护实体信息
    
    Attributes:
        llm: 用于实体提取的语言模型
        entity_store: 实体存储
        entity_extraction_prompt: 实体提取提示词
        entity_summary_prompt: 实体摘要提示词
    """
    
    def __init__(
        self,
        llm: Optional["BaseChatModel"] = None,
        entity_store: Optional[EntityStore] = None,
        entity_extraction_prompt: str = DEFAULT_ENTITY_EXTRACTION_PROMPT,
        entity_summary_prompt: str = DEFAULT_ENTITY_SUMMARY_PROMPT,
        chat_history: Optional[BaseChatMessageHistory] = None,
        k: int = 3,
        memory_key: str = "entities",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = False,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        verbose: bool = False
    ):
        """初始化实体记忆
        
        Args:
            llm: 用于实体提取的语言模型（可选）
            entity_store: 实体存储
            entity_extraction_prompt: 实体提取提示词
            entity_summary_prompt: 实体摘要提示词
            chat_history: 聊天消息历史
            k: 返回最近的k条消息用于上下文
            memory_key: 记忆键名
            input_key: 输入键名
            output_key: 输出键名
            return_messages: 是否返回消息列表
            human_prefix: 人类消息前缀
            ai_prefix: AI消息前缀
            verbose: 是否输出详细日志
        """
        super().__init__(
            memory_type=MemoryType.LONG_TERM,
            memory_key=memory_key,
            return_messages=return_messages,
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
            verbose=verbose
        )
        self.llm = llm
        self.entity_store = entity_store or EntityStore()
        self.entity_extraction_prompt = entity_extraction_prompt
        self.entity_summary_prompt = entity_summary_prompt
        self.chat_history = chat_history or InMemoryChatMessageHistory()
        self.k = k
        self.input_key = input_key
        self.output_key = output_key
    
    @property
    def memory_variables(self) -> List[str]:
        """返回记忆变量名列表"""
        return [self.memory_key, "history"]
    
    @property
    def buffer(self) -> List[Message]:
        """获取最近的消息"""
        messages = self.chat_history.messages
        return messages[-self.k * 2:] if len(messages) > self.k * 2 else messages
    
    def load_memory_variables(self, inputs: Optional[Dict[str, Any]] = None) -> MemoryVariables:
        """加载记忆变量"""
        # 获取输入中提到的实体
        mentioned_entities = []
        if inputs:
            input_text = inputs.get(self.input_key, str(inputs))
            mentioned_entities = self._extract_mentioned_entities(input_text)
        
        # 构建实体上下文
        entity_context_parts = []
        for entity_name in mentioned_entities:
            entity = self.entity_store.get(entity_name)
            if entity and entity.description:
                entity_context_parts.append(f"{entity.name}: {entity.description}")
        
        entities_str = "\n".join(entity_context_parts) if entity_context_parts else "无已知实体信息"
        
        # 构建历史
        history = self.format_messages(self.buffer)
        
        return MemoryVariables(
            history=history,
            entities={
                name: entity.to_dict() 
                for name, entity in self.entity_store.entities.items()
            },
            context=entities_str,
            extra={"mentioned_entities": mentioned_entities}
        )
    
    def _extract_mentioned_entities(self, text: str) -> List[str]:
        """提取文本中提到的已知实体
        
        Args:
            text: 输入文本
        
        Returns:
            实体名称列表
        """
        mentioned = []
        text_lower = text.lower()
        
        for entity_name in self.entity_store.get_all_names():
            if entity_name.lower() in text_lower:
                mentioned.append(entity_name)
        
        return mentioned
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存对话上下文并更新实体"""
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")
        
        # 保存消息
        if input_str:
            self.chat_history.add_user_message(input_str)
        if output_str:
            self.chat_history.add_ai_message(output_str)
        
        # 提取和更新实体
        combined_text = f"{input_str}\n{output_str}"
        self._update_entities(combined_text)
    
    def _update_entities(self, text: str) -> None:
        """从文本中提取并更新实体
        
        Args:
            text: 对话文本
        """
        if self.llm:
            # 使用LLM提取实体
            self._llm_extract_entities(text)
        else:
            # 使用简单规则提取
            self._simple_extract_entities(text)
    
    def _llm_extract_entities(self, text: str) -> None:
        """使用LLM提取实体"""
        try:
            # 提取实体
            prompt = self.entity_extraction_prompt.format(history=text)
            response = self.llm.invoke(prompt)
            
            # 解析响应
            import json
            try:
                result = json.loads(response.content)
                entities = result.get("entities", [])
            except json.JSONDecodeError:
                # 尝试简单提取
                entities = re.findall(r'"([^"]+)"', response.content)
            
            # 更新实体信息
            for entity_name in entities:
                if entity_name.strip():
                    existing = self.entity_store.get(entity_name)
                    existing_info = existing.description if existing else "无"
                    
                    # 生成实体摘要
                    summary_prompt = self.entity_summary_prompt.format(
                        entity=entity_name,
                        existing_info=existing_info,
                        history=text
                    )
                    summary_response = self.llm.invoke(summary_prompt)
                    description = summary_response.content.strip()
                    
                    if description and description != "无更新":
                        self.entity_store.set(entity_name, description)
                        if self.verbose:
                            print(f"[Memory] 更新实体 '{entity_name}': {description[:50]}...")
                    else:
                        self.entity_store.update_mention(entity_name)
                        
        except Exception as e:
            if self.verbose:
                print(f"[Memory] LLM实体提取失败: {e}")
            # 降级到简单提取
            self._simple_extract_entities(text)
    
    def _simple_extract_entities(self, text: str) -> None:
        """简单规则实体提取"""
        # 提取可能的实体（大写开头的词、引号内的内容等）
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # 英文名
            r'["「]([^"」]+)["」]',  # 引号内容
            r'(?:叫做|名为|称为|是)[\s]*([^\s,.!?，。！？]+)',  # 中文名称模式
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 1 and len(match) < 20:
                    self.entity_store.update_mention(match)
    
    def add_entity(self, name: str, description: str, metadata: Optional[Dict] = None) -> None:
        """手动添加实体
        
        Args:
            name: 实体名称
            description: 实体描述
            metadata: 额外元数据
        """
        self.entity_store.set(name, description, metadata)
    
    def get_entity(self, name: str) -> Optional[EntityInfo]:
        """获取实体信息
        
        Args:
            name: 实体名称
        
        Returns:
            实体信息或None
        """
        return self.entity_store.get(name)
    
    def clear(self) -> None:
        """清空记忆"""
        self.chat_history.clear()
        self.entity_store.clear()
    
    def clear_entities(self) -> None:
        """只清空实体（保留对话历史）"""
        self.entity_store.clear()
    
    def __repr__(self) -> str:
        return f"ConversationEntityMemory(entities={len(self.entity_store)}, messages={len(self.chat_history)})"
# genAI_main_end
