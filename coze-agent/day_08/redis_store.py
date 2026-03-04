"""
Redis状态存储模块
实现工作流状态持久化与恢复
"""

import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import hashlib


@dataclass
class WorkflowState:
    """工作流状态记录"""
    workflow_id: str
    state_data: Dict[str, Any]
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    node_id: str = "unknown"
    checksum: str = ""
    
    def __post_init__(self):
        # 计算数据校验和
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """计算状态数据的校验和"""
        data_str = json.dumps(self.state_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def update(self, new_state: Dict[str, Any], node_id: str = None):
        """更新状态"""
        self.state_data.update(new_state)
        self.version += 1
        self.updated_at = time.time()
        
        if node_id:
            self.node_id = node_id
        
        # 重新计算校验和
        self.checksum = self._calculate_checksum()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "workflow_id": self.workflow_id,
            "state_data": self.state_data,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "node_id": self.node_id,
            "checksum": self.checksum
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())


class RedisStateStore:
    """Redis状态存储（模拟实现）"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        
        # 模拟Redis存储
        self._storage: Dict[str, str] = {}
        self._expiry: Dict[str, float] = {}
        
        # 统计信息
        self.stats = {
            "set_operations": 0,
            "get_operations": 0,
            "delete_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "expired_keys": 0
        }
        
        # 连接池模拟
        self.pool_size = 10
        self.connection_pool = [f"conn_{i}" for i in range(self.pool_size)]
        
        print(f"[Redis存储] 初始化完成，主机: {host}:{port}, 数据库: {db}")
    
    async def connect(self):
        """连接到Redis（模拟）"""
        # 模拟连接延迟
        await asyncio.sleep(0.01)
        print(f"[Redis存储] 连接成功")
        return True
    
    async def disconnect(self):
        """断开连接"""
        print(f"[Redis存储] 连接断开")
    
    async def save_workflow_state(self, workflow_id: str, 
                                 state_data: Dict[str, Any],
                                 expire_seconds: int = 7200) -> bool:
        """保存工作流状态"""
        try:
            # 创建状态记录
            state = WorkflowState(
                workflow_id=workflow_id,
                state_data=state_data,
                node_id="coordinator"
            )
            
            # 存储到Redis（模拟）
            key = f"workflow:{workflow_id}"
            self._storage[key] = state.to_json()
            
            # 设置过期时间
            if expire_seconds > 0:
                self._expiry[key] = time.time() + expire_seconds
            
            self.stats["set_operations"] += 1
            
            print(f"[Redis存储] 保存状态: {workflow_id}, 版本: {state.version}")
            
            return True
            
        except Exception as e:
            print(f"[Redis存储] 保存状态失败: {e}")
            return False
    
    async def load_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """加载工作流状态"""
        try:
            key = f"workflow:{workflow_id}"
            
            # 检查是否过期
            if key in self._expiry and time.time() > self._expiry[key]:
                del self._storage[key]
                del self._expiry[key]
                self.stats["expired_keys"] += 1
                print(f"[Redis存储] 状态已过期: {workflow_id}")
                return None
            
            # 获取数据
            if key in self._storage:
                state_json = self._storage[key]
                state_dict = json.loads(state_json)
                
                # 验证校验和
                stored_state = WorkflowState(**state_dict)
                
                # 重新计算校验和
                current_checksum = stored_state._calculate_checksum()
                
                if stored_state.checksum != current_checksum:
                    print(f"[Redis存储] ⚠️  校验和不匹配: {workflow_id}")
                    # 可以选择修复或返回错误
                
                self.stats["get_operations"] += 1
                self.stats["cache_hits"] += 1
                
                print(f"[Redis存储] 加载状态: {workflow_id}, 版本: {stored_state.version}")
                
                return stored_state.state_data
            else:
                self.stats["cache_misses"] += 1
                print(f"[Redis存储] 状态不存在: {workflow_id}")
                return None
                
        except Exception as e:
            print(f"[Redis存储] 加载状态失败: {e}")
            return None
    
    async def update_workflow_state(self, workflow_id: str, 
                                   updates: Dict[str, Any]) -> bool:
        """更新工作流状态"""
        try:
            # 加载现有状态
            current_state = await self.load_workflow_state(workflow_id)
            
            if current_state is None:
                # 创建新状态
                return await self.save_workflow_state(workflow_id, updates)
            
            # 合并更新
            current_state.update(updates)
            
            # 保存更新后的状态
            key = f"workflow:{workflow_id}"
            
            # 加载完整状态记录
            if key in self._storage:
                state_json = self._storage[key]
                state_dict = json.loads(state_json)
                stored_state = WorkflowState(**state_dict)
                
                # 更新状态
                stored_state.update(updates)
                
                # 保存
                self._storage[key] = stored_state.to_json()
                
                print(f"[Redis存储] 更新状态: {workflow_id}, 新版本: {stored_state.version}")
                
                return True
            else:
                return await self.save_workflow_state(workflow_id, current_state)
                
        except Exception as e:
            print(f"[Redis存储] 更新状态失败: {e}")
            return False
    
    async def delete_workflow_state(self, workflow_id: str) -> bool:
        """删除工作流状态"""
        try:
            key = f"workflow:{workflow_id}"
            
            if key in self._storage:
                del self._storage[key]
                
                if key in self._expiry:
                    del self._expiry[key]
                
                self.stats["delete_operations"] += 1
                
                print(f"[Redis存储] 删除状态: {workflow_id}")
                
                return True
            else:
                print(f"[Redis存储] 状态不存在，无需删除: {workflow_id}")
                return False
                
        except Exception as e:
            print(f"[Redis存储] 删除状态失败: {e}")
            return False
    
    async def list_workflows(self, pattern: str = "workflow:*") -> List[str]:
        """列出所有工作流"""
        try:
            matching_keys = []
            
            for key in self._storage.keys():
                # 简单模式匹配
                if pattern.endswith("*"):
                    prefix = pattern[:-1]
                    if key.startswith(prefix):
                        matching_keys.append(key)
                elif pattern == key:
                    matching_keys.append(key)
            
            # 提取工作流ID
            workflow_ids = []
            for key in matching_keys:
                if key.startswith("workflow:"):
                    workflow_id = key.split(":", 1)[1]
                    workflow_ids.append(workflow_id)
            
            print(f"[Redis存储] 列出工作流，匹配模式: {pattern}, 数量: {len(workflow_ids)}")
            
            return workflow_ids
            
        except Exception as e:
            print(f"[Redis存储] 列出工作流失败: {e}")
            return []
    
    async def cleanup_expired(self) -> int:
        """清理过期的键"""
        try:
            expired_count = 0
            current_time = time.time()
            
            keys_to_delete = []
            
            for key, expiry in self._expiry.items():
                if current_time > expiry:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._storage[key]
                del self._expiry[key]
                expired_count += 1
            
            self.stats["expired_keys"] += expired_count
            
            if expired_count > 0:
                print(f"[Redis存储] 清理过期键，数量: {expired_count}")
            
            return expired_count
            
        except Exception as e:
            print(f"[Redis存储] 清理过期键失败: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        hit_rate = 0.0
        total_access = self.stats["cache_hits"] + self.stats["cache_misses"]
        
        if total_access > 0:
            hit_rate = self.stats["cache_hits"] / total_access
        
        return {
            "storage_size": len(self._storage),
            "expiry_count": len(self._expiry),
            "set_operations": self.stats["set_operations"],
            "get_operations": self.stats["get_operations"],
            "delete_operations": self.stats["delete_operations"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": hit_rate,
            "expired_keys": self.stats["expired_keys"],
            "connection_pool_size": self.pool_size,
            "active_connections": len(self.connection_pool)
        }
    
    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        total_size = 0
        workflow_count = 0
        
        for key, value in self._storage.items():
            if key.startswith("workflow:"):
                workflow_count += 1
                total_size += len(value.encode('utf-8'))
        
        avg_size = total_size / workflow_count if workflow_count > 0 else 0
        
        return {
            "total_workflows": workflow_count,
            "total_storage_bytes": total_size,
            "average_workflow_size_bytes": avg_size,
            "keys_by_prefix": {
                "workflow": workflow_count,
                "other": len(self._storage) - workflow_count
            }
        }


class StateRecoveryManager:
    """状态恢复管理器"""
    
    def __init__(self, redis_store: RedisStateStore):
        self.redis_store = redis_store
        self.recovery_log: List[Dict[str, Any]] = []
    
    async def recover_workflow(self, workflow_id: str, 
                              max_attempts: int = 3) -> Optional[Dict[str, Any]]:
        """恢复工作流"""
        print(f"[状态恢复] 开始恢复工作流: {workflow_id}")
        
        for attempt in range(max_attempts):
            try:
                # 尝试加载状态
                state = await self.redis_store.load_workflow_state(workflow_id)
                
                if state:
                    # 验证状态完整性
                    if self._validate_state_integrity(state):
                        print(f"[状态恢复] ✅ 工作流恢复成功: {workflow_id}")
                        
                        # 记录恢复日志
                        self._log_recovery(workflow_id, True, attempt + 1)
                        
                        return state
                    else:
                        print(f"[状态恢复] ⚠️  状态完整性验证失败: {workflow_id}")
                        
                        # 尝试修复
                        repaired_state = await self._repair_state(workflow_id, state)
                        
                        if repaired_state:
                            print(f"[状态恢复] ✅ 状态修复成功: {workflow_id}")
                            
                            self._log_recovery(workflow_id, True, attempt + 1, repaired=True)
                            
                            return repaired_state
                
                # 等待后重试
                if attempt < max_attempts - 1:
                    wait_time = (attempt + 1) * 0.5  # 指数退避
                    print(f"[状态恢复] 等待 {wait_time:.1f} 秒后重试...")
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                print(f"[状态恢复] 恢复尝试 {attempt + 1} 失败: {e}")
                
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1)
        
        print(f"[状态恢复] ❌ 工作流恢复失败: {workflow_id}")
        
        self._log_recovery(workflow_id, False, max_attempts)
        
        return None
    
    def _validate_state_integrity(self, state: Dict[str, Any]) -> bool:
        """验证状态完整性"""
        required_keys = ["workflow_status", "task_id"]
        
        for key in required_keys:
            if key not in state:
                return False
        
        # 检查工作流状态是否有效
        valid_statuses = ["running", "completed", "failed", "pending"]
        if state.get("workflow_status") not in valid_statuses:
            return False
        
        # 检查必要的数据字段
        if "code" not in state and state.get("workflow_status") != "completed":
            return False
        
        return True
    
    async def _repair_state(self, workflow_id: str, 
                           state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """尝试修复状态"""
        try:
            # 基础修复：确保必要字段存在
            repaired_state = state.copy()
            
            if "workflow_status" not in repaired_state:
                repaired_state["workflow_status"] = "failed"  # 安全默认值
            
            if "task_id" not in repaired_state:
                repaired_state["task_id"] = workflow_id
            
            if "start_time" not in repaired_state:
                repaired_state["start_time"] = time.time()
            
            # 保存修复后的状态
            success = await self.redis_store.save_workflow_state(
                workflow_id, 
                repaired_state
            )
            
            if success:
                return repaired_state
            else:
                return None
                
        except Exception as e:
            print(f"[状态恢复] 状态修复失败: {e}")
            return None
    
    def _log_recovery(self, workflow_id: str, success: bool, 
                     attempts: int, repaired: bool = False):
        """记录恢复日志"""
        log_entry = {
            "workflow_id": workflow_id,
            "success": success,
            "attempts": attempts,
            "repaired": repaired,
            "timestamp": time.time()
        }
        
        self.recovery_log.append(log_entry)
        
        print(f"[状态恢复] 记录恢复日志: {workflow_id}, "
              f"成功: {success}, 尝试次数: {attempts}, 修复: {repaired}")
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """获取恢复统计信息"""
        total_recoveries = len(self.recovery_log)
        successful = sum(1 for log in self.recovery_log if log["success"])
        failed = total_recoveries - successful
        
        success_rate = successful / total_recoveries if total_recoveries > 0 else 0.0
        
        repaired_count = sum(1 for log in self.recovery_log if log.get("repaired", False))
        
        avg_attempts = 0.0
        if successful > 0:
            avg_attempts = sum(log["attempts"] for log in self.recovery_log if log["success"]) / successful
        
        return {
            "total_recoveries": total_recoveries,
            "successful_recoveries": successful,
            "failed_recoveries": failed,
            "success_rate": success_rate,
            "repaired_states": repaired_count,
            "average_attempts": avg_attempts,
            "recent_recoveries": self.recovery_log[-5:] if self.recovery_log else []
        }


async def test_redis_store():
    """测试Redis存储"""
    print("\n测试Redis状态存储...")
    
    # 创建存储实例
    redis_store = RedisStateStore()
    await redis_store.connect()
    
    # 测试保存状态
    test_state = {
        "code": "def test(): pass",
        "status": "running",
        "priority": 2,
        "assigned_agent": "analyzer"
    }
    
    success = await redis_store.save_workflow_state("test_workflow_1", test_state)
    print(f"保存状态结果: {'成功' if success else '失败'}")
    
    # 测试加载状态
    loaded_state = await redis_store.load_workflow_state("test_workflow_1")
    print(f"加载状态结果: {'成功' if loaded_state else '失败'}")
    
    if loaded_state:
        print(f"加载的状态数据: {json.dumps(loaded_state, indent=2)[:200]}...")
    
    # 测试更新状态
    updates = {"status": "completed", "execution_time": 123.45}
    update_success = await redis_store.update_workflow_state("test_workflow_1", updates)
    print(f"更新状态结果: {'成功' if update_success else '失败'}")
    
    # 测试列出工作流
    workflows = await redis_store.list_workflows()
    print(f"列出工作流: {workflows}")
    
    # 测试统计信息
    stats = redis_store.get_stats()
    print(f"存储统计: {json.dumps(stats, indent=2)}")
    
    # 测试状态恢复管理器
    recovery_manager = StateRecoveryManager(redis_store)
    recovered_state = await recovery_manager.recover_workflow("test_workflow_1")
    
    if recovered_state:
        print(f"✅ 状态恢复测试通过")
        recovery_stats = recovery_manager.get_recovery_stats()
        print(f"恢复统计: {json.dumps(recovery_stats, indent=2)}")
    else:
        print(f"❌ 状态恢复测试失败")
    
    # 清理测试数据
    await redis_store.delete_workflow_state("test_workflow_1")
    
    # 断开连接
    await redis_store.disconnect()
    
    return recovered_state is not None


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_redis_store())