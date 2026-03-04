"""
协调器模块
实现分布式协调机制：领导者选举、状态同步、故障恢复
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random


class NodeState(Enum):
    """节点状态"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    FAILED = "failed"


class MessageType(Enum):
    """消息类型"""
    HEARTBEAT = "heartbeat"
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    STATE_SYNC = "state_sync"
    FAILURE_DETECTION = "failure_detection"


@dataclass
class RaftMessage:
    """Raft消息"""
    type: MessageType
    sender_id: str
    term: int
    data: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    address: str
    last_heartbeat: float = 0
    state: NodeState = NodeState.FOLLOWER
    is_alive: bool = True


class Coordinator:
    """分布式协调器（简化Raft实现）"""
    
    def __init__(self, node_id: str, all_nodes: List[Dict[str, str]]):
        self.node_id = node_id
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.state = NodeState.FOLLOWER
        self.leader_id: Optional[str] = None
        
        # 节点信息
        self.nodes: Dict[str, NodeInfo] = {}
        for node in all_nodes:
            node_info = NodeInfo(
                node_id=node["id"],
                address=node.get("address", "localhost")
            )
            self.nodes[node["id"]] = node_info
        
        # 选举相关
        self.election_timeout = random.uniform(1.5, 3.0)  # 随机超时时间
        self.last_heartbeat_received = 0
        self.election_timer: Optional[asyncio.Task] = None
        
        # 投票统计
        self.votes_received = 0
        self.vote_requests: Dict[str, Dict[str, Any]] = {}
        
        # 状态同步
        self.commit_index = 0
        self.last_applied = 0
        self.log: List[Dict[str, Any]] = []
        
        # 故障检测
        self.failure_detection_interval = 1.0
        self.failure_detection_task: Optional[asyncio.Task] = None
        
        print(f"[协调器] 节点 {node_id} 初始化完成，状态: {self.state.value}")
    
    async def start(self):
        """启动协调器"""
        # 启动选举定时器
        self.election_timer = asyncio.create_task(self._election_timer_task())
        
        # 启动故障检测
        self.failure_detection_task = asyncio.create_task(self._failure_detection_task())
        
        print(f"[协调器] 节点 {self.node_id} 已启动")
    
    async def stop(self):
        """停止协调器"""
        if self.election_timer:
            self.election_timer.cancel()
        
        if self.failure_detection_task:
            self.failure_detection_task.cancel()
        
        print(f"[协调器] 节点 {self.node_id} 已停止")
    
    async def _election_timer_task(self):
        """选举定时器任务"""
        try:
            while True:
                await asyncio.sleep(self.election_timeout)
                
                # 检查是否收到心跳
                time_since_heartbeat = time.time() - self.last_heartbeat_received
                
                if time_since_heartbeat > self.election_timeout:
                    print(f"[协调器] 选举超时，开始新一轮选举")
                    await self.start_election()
                    
                    # 重置超时时间
                    self.election_timeout = random.uniform(1.5, 3.0)
                
        except asyncio.CancelledError:
            print(f"[协调器] 选举定时器已取消")
    
    async def start_election(self):
        """开始领导者选举"""
        if self.state == NodeState.LEADER:
            return
        
        # 转换为候选者状态
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = 1  # 自己的一票
        
        print(f"[协调器] 节点 {self.node_id} 成为候选者，任期: {self.current_term}")
        
        # 请求其他节点投票
        vote_requests = []
        for node_id, node_info in self.nodes.items():
            if node_id == self.node_id:
                continue
            
            if node_info.is_alive:
                vote_request = self._create_vote_request(node_id)
                vote_requests.append(vote_request)
        
        # 并行发送投票请求
        if vote_requests:
            results = await asyncio.gather(*vote_requests, return_exceptions=True)
            
            # 统计投票结果
            for result in results:
                if isinstance(result, tuple) and result[0]:  # 投票成功
                    self.votes_received += 1
        
        # 检查是否获得多数票
        total_nodes = len(self.nodes)
        majority = total_nodes // 2 + 1
        
        if self.votes_received >= majority:
            await self._become_leader()
        else:
            # 选举失败，回到跟随者状态
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            print(f"[协调器] 选举失败，节点 {self.node_id} 回到跟随者状态")
    
    async def _create_vote_request(self, target_node_id: str) -> Tuple[bool, str]:
        """创建投票请求（模拟RPC调用）"""
        try:
            # 模拟网络延迟
            await asyncio.sleep(random.uniform(0.01, 0.1))
            
            # 模拟投票决策
            # 在实际Raft中，节点会根据任期、日志完整性等条件决定是否投票
            vote_granted = random.random() > 0.4  # 60%概率投票
            
            if vote_granted:
                print(f"[协调器] 节点 {target_node_id} 投票给 {self.node_id}")
                return True, target_node_id
            else:
                print(f"[协调器] 节点 {target_node_id} 拒绝投票给 {self.node_id}")
                return False, target_node_id
                
        except Exception as e:
            print(f"[协调器] 向节点 {target_node_id} 请求投票失败: {e}")
            return False, target_node_id
    
    async def _become_leader(self):
        """成为领导者"""
        self.state = NodeState.LEADER
        self.leader_id = self.node_id
        
        print(f"[协调器] 🎉 节点 {self.node_id} 成为第 {self.current_term} 任期的领导者")
        
        # 开始发送心跳
        asyncio.create_task(self._send_heartbeats())
    
    async def _send_heartbeats(self):
        """发送心跳（领导者调用）"""
        try:
            while self.state == NodeState.LEADER:
                heartbeat_tasks = []
                
                for node_id, node_info in self.nodes.items():
                    if node_id == self.node_id:
                        continue
                    
                    if node_info.is_alive:
                        task = self._send_heartbeat(node_id)
                        heartbeat_tasks.append(task)
                
                # 并行发送心跳
                if heartbeat_tasks:
                    await asyncio.gather(*heartbeat_tasks, return_exceptions=True)
                
                # 心跳间隔
                await asyncio.sleep(0.5)
                
        except asyncio.CancelledError:
            print(f"[协调器] 心跳发送任务已取消")
        except Exception as e:
            print(f"[协调器] 心跳发送失败: {e}")
            self.state = NodeState.FOLLOWER
    
    async def _send_heartbeat(self, target_node_id: str):
        """向单个节点发送心跳（模拟）"""
        try:
            # 模拟网络延迟
            await asyncio.sleep(random.uniform(0.005, 0.05))
            
            # 在实际系统中，这里会发送AppendEntries RPC
            print(f"[协调器] 向节点 {target_node_id} 发送心跳")
            
            # 更新目标节点最后心跳时间
            if target_node_id in self.nodes:
                self.nodes[target_node_id].last_heartbeat = time.time()
            
            return True
            
        except Exception as e:
            print(f"[协调器] 向节点 {target_node_id} 发送心跳失败: {e}")
            return False
    
    async def receive_heartbeat(self, leader_id: str, term: int):
        """接收心跳"""
        if term >= self.current_term:
            self.current_term = term
            self.leader_id = leader_id
            self.state = NodeState.FOLLOWER
            self.voted_for = None
            self.last_heartbeat_received = time.time()
            
            print(f"[协调器] 收到领导者 {leader_id} 的心跳，任期: {term}")
    
    async def _failure_detection_task(self):
        """故障检测任务"""
        try:
            while True:
                await asyncio.sleep(self.failure_detection_interval)
                
                # 检查节点健康状态
                current_time = time.time()
                
                for node_id, node_info in self.nodes.items():
                    if node_id == self.node_id:
                        continue
                    
                    # 简单检测：如果超过3秒没收到心跳，认为节点故障
                    time_since_heartbeat = current_time - node_info.last_heartbeat
                    
                    if time_since_heartbeat > 3.0:
                        if node_info.is_alive:
                            node_info.is_alive = False
                            print(f"[协调器] ⚠️  检测到节点 {node_id} 故障")
                            
                            # 触发故障恢复
                            await self._handle_node_failure(node_id)
                    else:
                        if not node_info.is_alive:
                            node_info.is_alive = True
                            print(f"[协调器] ✅ 节点 {node_id} 恢复在线")
                
        except asyncio.CancelledError:
            print(f"[协调器] 故障检测任务已取消")
    
    async def _handle_node_failure(self, failed_node_id: str):
        """处理节点故障"""
        print(f"[协调器] 开始处理节点 {failed_node_id} 的故障")
        
        # 1. 如果故障节点是领导者，触发重新选举
        if self.leader_id == failed_node_id:
            print(f"[协调器] 领导者节点故障，准备重新选举")
            self.leader_id = None
            
            # 如果当前节点是跟随者，可以尝试成为候选者
            if self.state == NodeState.FOLLOWER:
                await self.start_election()
        
        # 2. 重新分配故障节点的任务
        # 在实际系统中，这里会从状态存储中获取故障节点的未完成任务
        # 并重新分配给其他节点
        
        print(f"[协调器] 节点 {failed_node_id} 故障处理完成")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        alive_nodes = [n for n in self.nodes.values() if n.is_alive]
        dead_nodes = [n for n in self.nodes.values() if not n.is_alive]
        
        return {
            "current_term": self.current_term,
            "state": self.state.value,
            "leader_id": self.leader_id,
            "voted_for": self.voted_for,
            "total_nodes": len(self.nodes),
            "alive_nodes": len(alive_nodes),
            "dead_nodes": len(dead_nodes),
            "alive_node_ids": [n.node_id for n in alive_nodes],
            "dead_node_ids": [n.node_id for n in dead_nodes],
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "log_length": len(self.log)
        }


class StateSynchronizer:
    """状态同步器"""
    
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        self.state_version = 0
        self.sync_interval = 2.0  # 同步间隔
        self.sync_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """启动状态同步"""
        self.sync_task = asyncio.create_task(self._sync_task())
        print(f"[状态同步器] 已启动")
    
    async def stop(self):
        """停止状态同步"""
        if self.sync_task:
            self.sync_task.cancel()
        print(f"[状态同步器] 已停止")
    
    async def _sync_task(self):
        """状态同步任务"""
        try:
            while True:
                await asyncio.sleep(self.sync_interval)
                
                # 只有领导者负责状态同步
                if self.coordinator.state == NodeState.LEADER:
                    await self._sync_state_with_followers()
                
        except asyncio.CancelledError:
            print(f"[状态同步器] 同步任务已取消")
    
    async def _sync_state_with_followers(self):
        """与跟随者同步状态"""
        print(f"[状态同步器] 开始同步状态到跟随者")
        
        # 获取当前状态
        cluster_status = self.coordinator.get_cluster_status()
        
        # 同步到所有存活的跟随者
        sync_tasks = []
        
        for node_id, node_info in self.coordinator.nodes.items():
            if node_id == self.coordinator.node_id:
                continue
            
            if node_info.is_alive:
                task = self._send_state_sync(node_id, cluster_status)
                sync_tasks.append(task)
        
        if sync_tasks:
            results = await asyncio.gather(*sync_tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if r is True)
            print(f"[状态同步器] 状态同步完成，成功: {success_count}/{len(sync_tasks)}")
    
    async def _send_state_sync(self, target_node_id: str, state: Dict[str, Any]) -> bool:
        """向单个节点发送状态同步（模拟）"""
        try:
            # 模拟网络延迟
            await asyncio.sleep(random.uniform(0.01, 0.05))
            
            # 在实际系统中，这里会发送状态同步RPC
            print(f"[状态同步器] 向节点 {target_node_id} 同步状态")
            
            # 更新版本号
            self.state_version += 1
            
            return True
            
        except Exception as e:
            print(f"[状态同步器] 向节点 {target_node_id} 同步状态失败: {e}")
            return False
    
    async def save_state(self, state_key: str, state_data: Dict[str, Any]):
        """保存状态"""
        self.state_version += 1
        
        state_record = {
            "key": state_key,
            "data": state_data,
            "version": self.state_version,
            "timestamp": time.time(),
            "node_id": self.coordinator.node_id
        }
        
        # 在实际系统中，这里会保存到共享存储（如Redis）
        print(f"[状态同步器] 保存状态: {state_key}, 版本: {self.state_version}")
        
        return state_record
    
    async def restore_state(self, state_key: str) -> Optional[Dict[str, Any]]:
        """恢复状态"""
        # 模拟从存储中恢复
        print(f"[状态同步器] 恢复状态: {state_key}")
        
        # 返回模拟数据
        return {
            "key": state_key,
            "data": {"status": "restored"},
            "version": self.state_version,
            "timestamp": time.time()
        }


async def test_coordinator():
    """测试协调器"""
    print("\n测试分布式协调机制...")
    
    # 创建3个节点的集群
    nodes = [
        {"id": "node_1", "address": "localhost:8001"},
        {"id": "node_2", "address": "localhost:8002"},
        {"id": "node_3", "address": "localhost:8003"}
    ]
    
    # 创建协调器（模拟节点1）
    coordinator = Coordinator("node_1", nodes)
    await coordinator.start()
    
    # 等待一段时间，观察选举过程
    print("\n等待选举过程...")
    await asyncio.sleep(5)
    
    # 获取集群状态
    status = coordinator.get_cluster_status()
    print(f"\n集群状态:")
    print(f"- 当前任期: {status['current_term']}")
    print(f"- 节点状态: {status['state']}")
    print(f"- 领导者: {status['leader_id']}")
    print(f"- 存活节点: {status['alive_nodes']}/{status['total_nodes']}")
    
    # 测试故障检测
    print("\n模拟节点故障...")
    # 标记节点2为故障
    coordinator.nodes["node_2"].is_alive = False
    coordinator.nodes["node_2"].last_heartbeat = time.time() - 5
    
    # 等待故障检测
    await asyncio.sleep(2)
    
    # 检查故障处理
    status_after_failure = coordinator.get_cluster_status()
    print(f"故障后集群状态:")
    print(f"- 死亡节点: {status_after_failure['dead_nodes']}")
    print(f"- 死亡节点ID: {status_after_failure['dead_node_ids']}")
    
    # 测试状态同步器
    print("\n测试状态同步...")
    synchronizer = StateSynchronizer(coordinator)
    await synchronizer.start()
    
    # 模拟保存状态
    test_state = {"task": "test", "status": "running"}
    await synchronizer.save_state("test_task", test_state)
    
    # 等待同步
    await asyncio.sleep(3)
    
    # 停止服务
    await synchronizer.stop()
    await coordinator.stop()
    
    # 评估结果
    if status["leader_id"] is not None:
        print("✅ 领导者选举测试通过")
        return True
    else:
        print("❌ 领导者选举测试失败")
        return False


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_coordinator())