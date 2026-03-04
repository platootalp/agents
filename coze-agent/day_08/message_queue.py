"""
消息队列模块
模拟Kafka/RabbitMQ实现异步Agent通信
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class Message:
    """消息定义"""
    id: str
    topic: str
    data: Dict[str, Any]
    timestamp: float
    producer_id: str
    consumed_by: Optional[str] = None
    consumed_at: Optional[float] = None
    delivery_attempts: int = 0
    priority: int = 1  # 1-5，5为最高优先级
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "topic": self.topic,
            "data": self.data,
            "timestamp": self.timestamp,
            "producer_id": self.producer_id,
            "consumed_by": self.consumed_by,
            "consumed_at": self.consumed_at,
            "delivery_attempts": self.delivery_attempts,
            "priority": self.priority
        }


class MessageQueue:
    """基于内存的消息队列（模拟Kafka）"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.queues: Dict[str, List[Message]] = {}
        self.consumer_groups: Dict[str, Dict[str, int]] = {}  # topic -> consumer_id -> offset
        self.max_queue_size = max_queue_size
        self.delivered_count = 0
        self.failed_count = 0
        self.total_messages = 0
        
    async def create_topic(self, topic: str):
        """创建主题"""
        if topic not in self.queues:
            self.queues[topic] = []
            self.consumer_groups[topic] = {}
    
    async def produce(self, topic: str, data: Dict[str, Any], 
                     producer_id: str = "default", priority: int = 1) -> str:
        """生产消息"""
        if topic not in self.queues:
            await self.create_topic(topic)
        
        # 检查队列大小
        if len(self.queues[topic]) >= self.max_queue_size:
            # 按优先级淘汰低优先级消息
            self.queues[topic].sort(key=lambda x: x.priority)
            if self.queues[topic][0].priority < priority:
                removed = self.queues[topic].pop(0)
                self.failed_count += 1
                print(f"队列满，淘汰低优先级消息: {removed.id}")
        
        # 创建消息
        message = Message(
            id=str(uuid.uuid4()),
            topic=topic,
            data=data,
            timestamp=time.time(),
            producer_id=producer_id,
            priority=priority
        )
        
        # 加入队列（按优先级排序）
        self.queues[topic].append(message)
        self.queues[topic].sort(key=lambda x: (-x.priority, x.timestamp))
        
        self.total_messages += 1
        
        print(f"[消息队列] 生产消息: {message.id} 到主题 {topic}, 优先级: {priority}")
        
        return message.id
    
    async def consume(self, topic: str, consumer_id: str, 
                     timeout: float = 1.0) -> Optional[Message]:
        """消费消息"""
        if topic not in self.queues or not self.queues[topic]:
            # 等待新消息
            try:
                await asyncio.sleep(0.1)
                if not self.queues.get(topic):
                    return None
            except asyncio.TimeoutError:
                return None
        
        # 获取消费者偏移量
        if topic not in self.consumer_groups:
            self.consumer_groups[topic] = {}
        
        if consumer_id not in self.consumer_groups[topic]:
            self.consumer_groups[topic][consumer_id] = 0
        
        offset = self.consumer_groups[topic][consumer_id]
        
        # 检查偏移量是否有效
        if offset >= len(self.queues[topic]):
            return None
        
        # 获取消息
        message = self.queues[topic][offset]
        
        # 更新偏移量
        self.consumer_groups[topic][consumer_id] = offset + 1
        
        # 标记为已消费
        message.consumed_by = consumer_id
        message.consumed_at = time.time()
        message.delivery_attempts += 1
        
        self.delivered_count += 1
        
        print(f"[消息队列] 消费消息: {message.id} 由消费者 {consumer_id}")
        
        return message
    
    async def ack(self, message_id: str):
        """确认消息处理成功"""
        # 在实际系统中，这里会从持久化存储中移除消息
        # 这里简化处理，只记录日志
        print(f"[消息队列] 消息确认: {message_id}")
    
    async def nack(self, message_id: str, requeue: bool = True):
        """消息处理失败，可选择重新入队"""
        print(f"[消息队列] 消息处理失败: {message_id}, 重新入队: {requeue}")
        
        if requeue:
            # 在实际系统中，这里会更新消息状态并重新入队
            self.failed_count += 1
    
    def get_queue_stats(self, topic: str) -> Dict[str, Any]:
        """获取队列统计信息"""
        if topic not in self.queues:
            return {}
        
        queue = self.queues[topic]
        
        return {
            "total_messages": len(queue),
            "pending_messages": len([m for m in queue if m.consumed_by is None]),
            "consumed_messages": len([m for m in queue if m.consumed_by is not None]),
            "average_priority": sum(m.priority for m in queue) / len(queue) if queue else 0,
            "oldest_message": min(m.timestamp for m in queue) if queue else None,
            "newest_message": max(m.timestamp for m in queue) if queue else None
        }
    
    @property
    def loss_rate(self) -> float:
        """计算消息丢失率"""
        if self.total_messages == 0:
            return 0.0
        
        delivered = self.delivered_count
        failed = self.failed_count
        
        # 简化计算：失败消息比例
        return failed / (delivered + failed) if (delivered + failed) > 0 else 0.0


class MessageReliability:
    """消息可靠性保障"""
    
    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0,
                 max_delay: float = 10.0, jitter: float = 0.1):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter
        
        self.delivered_count = 0
        self.failed_count = 0
        self.retry_count = 0
        
    async def send_with_retry(self, message: Dict[str, Any], 
                             send_func, context: Dict[str, Any] = None) -> bool:
        """带重试的消息发送"""
        context = context or {}
        
        for attempt in range(self.max_retries):
            try:
                # 添加重试相关信息
                message_with_retry = message.copy()
                message_with_retry["_retry_attempt"] = attempt
                message_with_retry["_retry_context"] = context
                
                # 发送消息
                await send_func(message_with_retry)
                
                self.delivered_count += 1
                
                if attempt > 0:
                    print(f"✅ 消息发送成功（第{attempt+1}次重试）")
                
                return True
                
            except Exception as e:
                print(f"⚠️  消息发送失败（尝试 {attempt+1}/{self.max_retries}）: {e}")
                
                if attempt < self.max_retries - 1:
                    # 计算延迟时间（指数退避 + 抖动）
                    delay = min(
                        self.initial_delay * (2 ** attempt),
                        self.max_delay
                    )
                    
                    # 添加随机抖动
                    jitter_amount = delay * self.jitter
                    delay += (jitter_amount * (2 * (attempt % 2) - 1))  # 正负抖动
                    
                    print(f"等待 {delay:.2f} 秒后重试...")
                    await asyncio.sleep(delay)
                    
                    self.retry_count += 1
        
        # 所有重试都失败
        self.failed_count += 1
        print(f"❌ 消息发送失败，已达最大重试次数")
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.delivered_count + self.failed_count
        
        return {
            "delivered_count": self.delivered_count,
            "failed_count": self.failed_count,
            "retry_count": self.retry_count,
            "delivery_rate": self.delivered_count / total if total > 0 else 1.0,
            "loss_rate": self.failed_count / total if total > 0 else 0.0
        }


# 全局消息队列实例
global_message_queue = MessageQueue()


async def test_message_queue():
    """测试消息队列"""
    print("\n测试消息队列功能...")
    
    mq = MessageQueue()
    
    # 生产消息
    msg_id = await mq.produce(
        topic="test_topic",
        data={"type": "test", "content": "Hello World"},
        producer_id="test_producer",
        priority=3
    )
    
    print(f"生产消息ID: {msg_id}")
    
    # 消费消息
    message = await mq.consume("test_topic", "test_consumer")
    
    if message:
        print(f"消费消息: {message.id}, 数据: {message.data}")
        await mq.ack(message.id)
    else:
        print("无消息可消费")
    
    # 测试统计
    stats = mq.get_queue_stats("test_topic")
    print(f"队列统计: {stats}")
    print(f"消息丢失率: {mq.loss_rate:.2%}")
    
    return mq.loss_rate < 0.001  # 应接近0


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_message_queue())