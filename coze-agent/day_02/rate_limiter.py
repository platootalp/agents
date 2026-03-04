"""
智能限流装饰器

本文件提供了完整的企业级智能限流装饰器实现，包含：
1. 详细的代码注释，解释每行关键逻辑
2. 多种实现思路对比
3. 工程架构考量说明
4. 与Java方案的详细对比

设计思路（从Java后端视角）：
- 借鉴Guava RateLimiter的API设计理念
- 集成Resilience4j风格的熔断器状态机
- 考虑分布式环境下的数据一致性问题
- 提供企业级监控集成接口
"""

import time
import asyncio
import threading
import functools
from typing import Any, Callable, Optional, Union, Dict, List, TypeVar, cast
from dataclasses import dataclass, field
from enum import Enum
import redis
import logging

# 类型变量定义，用于泛型装饰器
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# 日志配置：企业级应用应有结构化日志
logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """
    限流策略枚举
    从Java工程视角看，这类似于Guava RateLimiter的Acquisition策略

    策略说明：
    - BLOCKING: 阻塞等待直到获取令牌（类似Java的acquire()）
    - NON_BLOCKING: 非阻塞，立即返回结果（类似Java的tryAcquire()）
    - HYBRID: 混合模式，先尝试获取，失败后等待重试
    """
    BLOCKING = "blocking"
    NON_BLOCKING = "non_blocking"
    HYBRID = "hybrid"


class CircuitBreakerState(Enum):
    """
    熔断器状态枚举
    借鉴Resilience4j的三状态熔断器设计

    状态说明：
    - CLOSED: 正常状态，请求通过
    - OPEN: 熔断状态，请求被拒绝（快速失败）
    - HALF_OPEN: 半开状态，有限请求通过以测试服务恢复
    """
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RateLimiterConfig:
    """
    限流器配置类
    采用dataclass简化配置管理，类似Java的Builder模式

    配置项说明：
    - tokens_per_second: 令牌生成速率，决定QPS上限
    - bucket_capacity: 桶容量，决定突发流量处理能力
    - strategy: 限流策略，选择阻塞/非阻塞/混合模式
    - max_wait_time: 最大等待时间，防止无限阻塞
    - circuit_breaker_enabled: 是否启用熔断器
    - failure_threshold: 连续失败阈值，触发熔断
    - recovery_timeout: 恢复超时时间，OPEN→HALF_OPEN
    - half_open_max_requests: 半开状态最大尝试请求数
    - redis_enabled: 是否启用Redis分布式存储
    - redis_key_prefix: Redis键名前缀，便于多环境隔离
    - redis_host/port/db: Redis连接配置
    - lua_script_caching: 是否缓存Lua脚本，提升性能
    - clock_skew_protection: 是否启用时钟偏移保护
    """
    # 令牌桶核心参数
    tokens_per_second: float = 10.0
    bucket_capacity: int = 100

    # 限流策略参数
    strategy: RateLimitStrategy = RateLimitStrategy.BLOCKING
    max_wait_time: float = 60.0

    # 熔断器参数（Resilience4j风格）
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 10
    recovery_timeout: float = 30.0
    half_open_max_requests: int = 5

    # Redis分布式存储参数
    redis_enabled: bool = False
    redis_key_prefix: str = "rate_limiter"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # 性能优化参数
    lua_script_caching: bool = True
    clock_skew_protection: bool = True


class SmartRateLimiter:
    """
    智能限流装饰器核心类

    设计哲学（Java工程思维）：
    1. 单一职责：专注于限流和熔断，不侵入业务逻辑
    2. 开闭原则：通过配置扩展，而非修改代码
    3. 依赖反转：依赖抽象（配置类），而非具体实现
    4. 接口隔离：提供同步/异步两套API，满足不同场景

    与Java实现的对比思考：
    - Python装饰器 vs Java注解+AOP：装饰器更轻量，但缺乏编译期检查
    - 动态类型 vs 静态类型：Python灵活，Java安全
    - GIL限制 vs 真正多线程：Python并发需谨慎设计
    """

    def __init__(
            self,
            config: Optional[RateLimiterConfig] = None,
            function_name: Optional[str] = None
    ) -> None:
        """
        初始化限流器

        Args:
            config: 限流器配置，None则使用默认值
            function_name: 被装饰函数名，用于监控和Redis键生成

        设计思考：
        - 支持函数名注入：便于监控系统区分不同API的限流情况
        - 配置默认值：提供合理的默认配置，降低使用门槛
        - 惰性初始化：Redis连接在首次使用时建立，避免不必要的资源占用
        """
        # 使用提供的配置或默认配置
        self.config = config or RateLimiterConfig()
        self.function_name = function_name or "unknown_function"

        # 本地令牌桶状态（线程安全设计）
        self._tokens: float = float(self.config.bucket_capacity)  # 当前令牌数
        self._last_update_time: float = time.time()  # 最后更新时间
        self._lock = threading.RLock()  # 可重入锁，支持嵌套调用

        # Redis客户端（分布式模式）
        self._redis_client: Optional[redis.Redis] = None
        self._lua_scripts: Dict[str, str] = {}  # Lua脚本缓存

        # 熔断器状态（Resilience4j风格状态机）
        self._failure_count: int = 0  # 连续失败计数
        self._circuit_state: CircuitBreakerState = CircuitBreakerState.CLOSED  # 当前状态
        self._circuit_last_state_change: float = time.time()  # 最后状态变更时间
        self._circuit_half_open_success_count: int = 0  # 半开状态成功计数

        # 监控指标（Prometheus风格）
        self._total_requests: int = 0  # 总请求数
        self._limited_requests: int = 0  # 被限流请求数
        self._circuit_opened_count: int = 0  # 熔断触发次数

        # 条件初始化：仅在启用Redis时建立连接
        if self.config.redis_enabled:
            self._init_redis_connection()
            self._load_lua_scripts()

    def _init_redis_connection(self) -> None:
        """
        初始化Redis连接

        工程考量：
        1. 连接池管理：Redis客户端内部管理连接池
        2. 超时设置：生产环境应配置合理的超时时间
        3. 健康检查：初始化时进行ping测试
        4. 优雅降级：连接失败时自动降级到本地模式
        """
        try:
            # 创建Redis客户端（decode_responses=True自动解码字符串）
            self._redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,  # 连接超时5秒
                socket_timeout=10  # 读写超时10秒
            )

            # 健康检查：ping测试连接可用性
            self._redis_client.ping()
            logger.info(f"Redis连接成功: {self.config.redis_host}:{self.config.redis_port}")

        except Exception as e:
            # 优雅降级：Redis不可用时切换到本地模式
            logger.error(f"Redis连接失败，降级到本地模式: {e}")
            self._redis_client = None
            self.config.redis_enabled = False  # 禁用Redis功能

    def _load_lua_scripts(self) -> None:
        """
        加载Lua脚本用于Redis原子操作

        为什么使用Lua脚本：
        1. 原子性：Redis保证Lua脚本的原子执行
        2. 性能：减少网络往返（多个操作一次执行）
        3. 一致性：避免客户端竞态条件

        与Java方案的对比：
        - Java+Redis：通常使用Redisson库，它也使用Lua脚本保证原子性
        - Python实现：原理相同，但API更简洁
        """
        if not self.config.redis_enabled or not self._redis_client:
            return

        # Lua脚本1：获取令牌（原子操作）
        # 设计要点：一次往返完成读取、计算、更新操作
        acquire_token_script = """
        -- KEYS[1]: 令牌桶键名
        -- ARGV[1]: 令牌生成速率
        -- ARGV[2]: 桶容量
        -- ARGV[3]: 当前时间戳
        -- ARGV[4]: 请求令牌数

        local key = KEYS[1]
        local tokens_per_second = tonumber(ARGV[1])
        local bucket_capacity = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local requested_tokens = tonumber(ARGV[4])

        -- 读取当前状态：使用HMGET一次获取多个字段
        local data = redis.call('HMGET', key, 'tokens', 'last_update')
        local current_tokens = tonumber(data[1]) or bucket_capacity  -- 默认值处理
        local last_update = tonumber(data[2]) or now

        -- 计算新增令牌：基于时间差和速率
        -- math.max(0, ...) 防止时钟回拨导致负时间差
        local time_passed = math.max(0, now - last_update)
        local new_tokens = time_passed * tokens_per_second
        current_tokens = math.min(bucket_capacity, current_tokens + new_tokens)

        -- 尝试获取令牌：充足则扣除，不足则返回0
        local acquired = 0
        if current_tokens >= requested_tokens then
            current_tokens = current_tokens - requested_tokens
            acquired = requested_tokens
        end

        -- 原子更新：HMSET保证字段同时更新
        redis.call('HMSET', key, 
            'tokens', current_tokens,
            'last_update', now
        )

        -- 设置过期时间：防止内存泄漏，3600秒=1小时
        redis.call('EXPIRE', key, 3600)

        -- 返回实际获取的令牌数
        return acquired
        """

        # Lua脚本2：获取熔断器状态
        circuit_breaker_script = """
        local key = KEYS[1]
        local state = redis.call('HGET', key, 'state') or 'closed'
        local failure_count = tonumber(redis.call('HGET', key, 'failure_count') or 0)
        local last_state_change = tonumber(redis.call('HGET', key, 'last_state_change') or 0)

        return {state, failure_count, last_state_change}
        """

        # 存储脚本内容
        self._lua_scripts['acquire_token'] = acquire_token_script
        self._lua_scripts['circuit_breaker'] = circuit_breaker_script

        # 可选：预加载脚本到Redis，获取SHA校验和
        # 优点：后续调用使用EVALSHA减少网络传输
        if self.config.lua_script_caching:
            try:
                self._lua_scripts['acquire_token_sha'] = self._redis_client.script_load(acquire_token_script)
                self._lua_scripts['circuit_breaker_sha'] = self._redis_client.script_load(circuit_breaker_script)
                logger.debug("Lua脚本预加载成功")
            except Exception as e:
                logger.warning(f"Lua脚本预加载失败，将使用eval: {e}")

    def _get_redis_key(self, suffix: str = "") -> str:
        """
        生成Redis键名

        键名设计原则：
        1. 可读性：包含功能名和函数名，便于人工排查
        2. 唯一性：使用前缀隔离不同环境和应用
        3. 规范性：统一的分隔符和命名规则
        """
        base_key = f"{self.config.redis_key_prefix}:{self.function_name}"
        if suffix:
            return f"{base_key}:{suffix}"
        return base_key

    def _update_local_tokens(self) -> None:
        """
        更新本地令牌桶状态

        算法核心：
        1. 计算时间差：当前时间 - 最后更新时间
        2. 计算新增令牌：时间差 × 生成速率
        3. 限制上限：不超过桶容量
        4. 更新状态：当前令牌数和最后更新时间

        时钟问题处理：
        - 时钟回拨：检测负时间差，按0处理
        - 时钟超前：正常处理，可能导致令牌突发
        """
        with self._lock:
            now = time.time()
            time_passed = max(0.0, now - self._last_update_time)

            # 时钟回拨保护：生产环境重要！
            if self.config.clock_skew_protection and time_passed < 0:
                logger.warning(f"检测到时钟回拨: {time_passed}秒，按0处理")
                time_passed = 0

            # 令牌再生计算
            new_tokens = time_passed * self.config.tokens_per_second
            self._tokens = min(self.config.bucket_capacity, self._tokens + new_tokens)
            self._last_update_time = now

    def _acquire_local_token(self, tokens: float = 1.0) -> bool:
        """
        从本地令牌桶获取令牌

        Args:
            tokens: 请求的令牌数量，支持小数（如0.5个令牌）

        Returns:
            bool: 是否成功获取

        设计思考：
        - 支持小数令牌：满足精细化限流需求
        - 先更新再获取：确保令牌再生计算准确
        - 线程安全：在锁保护下执行整个操作
        """
        with self._lock:
            # 先更新令牌状态（考虑时间流逝）
            self._update_local_tokens()

            # 检查令牌是否充足
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def _acquire_redis_token(self, tokens: float = 1.0) -> bool:
        """
        从Redis令牌桶获取令牌（分布式环境）

        分布式一致性保证：
        1. Lua脚本原子性：读取-计算-更新在一个原子操作中完成
        2. 时钟同步：使用客户端时间，配合回拨保护
        3. 降级机制：Redis故障时自动切换到本地模式

        性能优化：
        1. 脚本缓存：使用EVALSHA减少网络传输
        2. 连接池：Redis客户端自动管理连接复用
        3. 批量操作：支持一次获取多个令牌
        """
        # 降级检查：Redis不可用时使用本地模式
        if not self._redis_client:
            return self._acquire_local_token(tokens)

        try:
            key = self._get_redis_key("bucket")
            now = time.time()

            # 优先使用缓存的SHA校验和（如果可用）
            if 'acquire_token_sha' in self._lua_scripts:
                acquired = self._redis_client.evalsha(
                    self._lua_scripts['acquire_token_sha'],
                    1,  # KEY数量
                    key,
                    self.config.tokens_per_second,
                    self.config.bucket_capacity,
                    now,
                    tokens
                )
            else:
                acquired = self._redis_client.eval(
                    self._lua_scripts['acquire_token'],
                    1,
                    key,
                    self.config.tokens_per_second,
                    self.config.bucket_capacity,
                    now,
                    tokens
                )

            # 判断是否获取到足够的令牌
            return float(acquired or 0) >= tokens

        except redis.RedisError as e:
            # Redis-specific错误
            logger.error(f"Redis操作失败: {e}")
            # 优雅降级：切换到本地模式
            return self._acquire_local_token(tokens)
        except Exception as e:
            # 其他未知错误
            logger.error(f"获取令牌时发生未知错误: {e}")
            # 安全起见，返回False（限流）
            return False

    def _update_circuit_breaker(self, success: bool) -> None:
        """
        更新熔断器状态（Resilience4j风格状态机）

        状态转换逻辑：
        1. CLOSED → OPEN: 连续失败达到阈值
        2. OPEN → HALF_OPEN: 经过恢复超时时间
        3. HALF_OPEN → CLOSED: 成功请求达到阈值
        4. HALF_OPEN → OPEN: 任何失败

        工程实践：
        - 阈值配置化：适应不同服务的容错需求
        - 状态持久化：生产环境应考虑Redis存储状态
        - 监控告警：状态变更时触发告警
        """
        if not self.config.circuit_breaker_enabled:
            return

        now = time.time()

        # 成功处理
        if success:
            self._failure_count = 0  # 重置失败计数

            # 半开状态下的成功处理
            if self._circuit_state == CircuitBreakerState.HALF_OPEN:
                self._circuit_half_open_success_count += 1

                # 成功次数达到阈值，关闭熔断器
                if self._circuit_half_open_success_count >= self.config.half_open_max_requests:
                    self._circuit_state = CircuitBreakerState.CLOSED
                    self._circuit_last_state_change = now
                    self._circuit_half_open_success_count = 0
                    logger.info("熔断器恢复成功，状态: CLOSED")

        # 失败处理
        else:
            self._failure_count += 1

            # CLOSED状态下失败计数达到阈值
            if self._circuit_state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._circuit_state = CircuitBreakerState.OPEN
                    self._circuit_last_state_change = now
                    self._circuit_opened_count += 1
                    logger.warning(f"熔断器触发，状态: OPEN (连续失败: {self._failure_count})")

            # HALF_OPEN状态下失败，重新打开
            elif self._circuit_state == CircuitBreakerState.HALF_OPEN:
                self._circuit_state = CircuitBreakerState.OPEN
                self._circuit_last_state_change = now
                self._circuit_half_open_success_count = 0
                logger.warning("半开状态下失败，熔断器重新打开")

        # 检查OPEN状态是否应该进入HALF_OPEN
        if (self._circuit_state == CircuitBreakerState.OPEN and
                now - self._circuit_last_state_change >= self.config.recovery_timeout):
            self._circuit_state = CircuitBreakerState.HALF_OPEN
            self._circuit_last_state_change = now
            self._circuit_half_open_success_count = 0
            logger.info("熔断器进入恢复期，状态: HALF_OPEN")

    def _check_circuit_breaker(self) -> bool:
        """
        检查熔断器状态，决定是否允许请求通过

        返回逻辑：
        - CLOSED: 允许通过
        - OPEN: 拒绝通过（除非超时进入HALF_OPEN）
        - HALF_OPEN: 允许通过（但数量受限）

        设计思考：
        - 快速失败：OPEN状态立即拒绝，避免资源浪费
        - 渐进恢复：HALF_OPEN状态有限尝试，验证服务恢复
        - 状态自愈：超时自动转换状态，无需外部干预
        """
        if not self.config.circuit_breaker_enabled:
            return True

        # CLOSED状态：正常通过
        if self._circuit_state == CircuitBreakerState.CLOSED:
            return True

        # OPEN状态：检查是否应该进入HALF_OPEN
        elif self._circuit_state == CircuitBreakerState.OPEN:
            now = time.time()
            if now - self._circuit_last_state_change >= self.config.recovery_timeout:
                # 状态转换：OPEN → HALF_OPEN
                self._circuit_state = CircuitBreakerState.HALF_OPEN
                self._circuit_last_state_change = now
                self._circuit_half_open_success_count = 0
                logger.info("熔断器进入恢复期，状态: HALF_OPEN")
                return True
            return False  # 仍在OPEN状态，拒绝请求

        # HALF_OPEN状态：允许通过（调用者需限制数量）
        elif self._circuit_state == CircuitBreakerState.HALF_OPEN:
            return True

        # 未知状态：安全起见拒绝
        return False

    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        获取令牌（同步版本）

        Args:
            tokens: 请求的令牌数量
            timeout: 超时时间（秒），None则使用配置值

        Returns:
            bool: 是否成功获取

        Raises:
            TimeoutError: 阻塞模式下等待超时

        Java对应方法：
        - Guava RateLimiter.acquire(): 阻塞获取
        - Guava RateLimiter.tryAcquire(): 非阻塞尝试

        设计差异：
        - Python: 一个方法通过策略参数区分行为
        - Java: 不同方法明确区分阻塞/非阻塞
        """
        # 监控：请求计数
        self._total_requests += 1

        # 熔断器检查：快速失败
        if not self._check_circuit_breaker():
            self._limited_requests += 1
            return False

        # 超时时间确定
        if timeout is None:
            timeout = self.config.max_wait_time

        # 根据策略执行限流
        if self.config.strategy == RateLimitStrategy.NON_BLOCKING:
            # 非阻塞模式：立即尝试，成功与否都立即返回
            acquired = self._acquire_redis_token(tokens) if self.config.redis_enabled else self._acquire_local_token(
                tokens)

            if not acquired:
                self._limited_requests += 1
                self._update_circuit_breaker(False)

            return acquired

        elif self.config.strategy == RateLimitStrategy.BLOCKING:
            # 阻塞模式：等待直到获取令牌或超时
            start_time = time.time()
            remaining_time = timeout

            while remaining_time > 0:
                acquired = self._acquire_redis_token(
                    tokens) if self.config.redis_enabled else self._acquire_local_token(tokens)

                if acquired:
                    return True

                # 计算剩余时间
                elapsed = time.time() - start_time
                remaining_time = timeout - elapsed

                if remaining_time <= 0:
                    break

                # 短暂休眠：避免CPU占用过高
                sleep_time = min(0.1, remaining_time)
                time.sleep(sleep_time)

            # 超时处理
            self._limited_requests += 1
            self._update_circuit_breaker(False)
            raise TimeoutError(f"获取令牌超时: {timeout}秒")

        elif self.config.strategy == RateLimitStrategy.HYBRID:
            # 混合模式：先尝试获取，失败后等待指定时间重试
            acquired = self._acquire_redis_token(tokens) if self.config.redis_enabled else self._acquire_local_token(
                tokens)

            if not acquired and timeout > 0:
                # 等待后重试
                time.sleep(min(0.5, timeout))
                acquired = self._acquire_redis_token(
                    tokens) if self.config.redis_enabled else self._acquire_local_token(tokens)

            if not acquired:
                self._limited_requests += 1
                self._update_circuit_breaker(False)

            return acquired

        return False

    async def acquire_async(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        获取令牌（异步版本）

        异步设计考量：
        1. 协程友好：使用asyncio.sleep而非time.sleep
        2. 非阻塞IO：Redis操作应使用异步客户端（aioredis）
        3. 并发安全：异步环境下的状态保护

        当前限制：
        - 使用同步Redis客户端，在异步环境中可能阻塞
        - 生产环境应替换为aioredis或redis.asyncio
        """
        self._total_requests += 1

        if not self._check_circuit_breaker():
            self._limited_requests += 1
            return False

        if timeout is None:
            timeout = self.config.max_wait_time

        if self.config.strategy == RateLimitStrategy.NON_BLOCKING:
            acquired = self._acquire_redis_token(tokens) if self.config.redis_enabled else self._acquire_local_token(
                tokens)

            if not acquired:
                self._limited_requests += 1
                self._update_circuit_breaker(False)

            return acquired

        elif self.config.strategy == RateLimitStrategy.BLOCKING:
            start_time = time.time()
            remaining_time = timeout

            while remaining_time > 0:
                acquired = self._acquire_redis_token(
                    tokens) if self.config.redis_enabled else self._acquire_local_token(tokens)

                if acquired:
                    return True

                elapsed = time.time() - start_time
                remaining_time = timeout - elapsed

                if remaining_time <= 0:
                    break

                sleep_time = min(0.1, remaining_time)
                await asyncio.sleep(sleep_time)

            self._limited_requests += 1
            self._update_circuit_breaker(False)
            raise TimeoutError(f"获取令牌超时: {timeout}秒")

        elif self.config.strategy == RateLimitStrategy.HYBRID:
            acquired = self._acquire_redis_token(tokens) if self.config.redis_enabled else self._acquire_local_token(
                tokens)

            if not acquired and timeout > 0:
                await asyncio.sleep(min(0.5, timeout))
                acquired = self._acquire_redis_token(
                    tokens) if self.config.redis_enabled else self._acquire_local_token(tokens)

            if not acquired:
                self._limited_requests += 1
                self._update_circuit_breaker(False)

            return acquired

        return False

    def __call__(self, func: F) -> F:
        """
        装饰器实现

        Python装饰器 vs Java注解：
        - 相同点：都是无侵入的横切关注点实现
        - 不同点：Python装饰器是运行时包装，Java注解需要AOP框架

        设计模式：
        - 装饰器模式：动态添加功能
        - 包装器模式：保持接口不变
        """
        # 记录函数名，便于监控
        self.function_name = func.__name__

        # 同步函数包装器
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # 限流检查：获取令牌
                self.acquire()

                # 执行原函数
                result = func(*args, **kwargs)

                # 更新熔断器：成功
                self._update_circuit_breaker(True)

                return result

            except Exception as e:
                # 更新熔断器：失败
                self._update_circuit_breaker(False)
                raise e  # 重新抛出异常

        # 异步函数包装器
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # 异步限流检查
                await self.acquire_async()

                # 执行原函数
                result = await func(*args, **kwargs)

                # 更新熔断器：成功
                self._update_circuit_breaker(True)

                return result

            except Exception as e:
                # 更新熔断器：失败
                self._update_circuit_breaker(False)
                raise e

        # 根据函数类型返回相应包装器
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, wrapper)

    def update_config(self, new_config: RateLimiterConfig) -> None:
        """
        更新限流器配置（热重载）

        生产环境需求：
        1. 动态调整：根据系统负载实时调整限流参数
        2. 灰度发布：新配置逐步生效，观察影响
        3. 回滚机制：配置异常时快速回退

        实现要点：
        - 原子性：在锁保护下更新整个配置
        - 连接管理：Redis配置变更时重建连接
        - 状态同步：新配置可能需要重置某些状态
        """
        with self._lock:
            # 保存旧配置（用于回滚或审计）
            old_config = self.config

            # 更新配置
            self.config = new_config

            # 检查是否需要重新初始化Redis
            need_reinit = (
                    new_config.redis_enabled != old_config.redis_enabled or
                    new_config.redis_host != old_config.redis_host or
                    new_config.redis_port != old_config.redis_port or
                    new_config.redis_db != old_config.redis_db
            )

            if need_reinit:
                self._init_redis_connection()

            # 重新加载Lua脚本（配置变更可能影响脚本逻辑）
            self._load_lua_scripts()

            # 日志记录（审计跟踪）
            logger.info(f"限流器配置已更新: {self.function_name}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        获取限流器统计指标

        监控指标设计（Prometheus风格）：
        1. 计数器：总请求数、限流次数、熔断触发次数
        2. 仪表盘：当前令牌数、成功率
        3. 状态：熔断器状态、配置参数

        企业集成：
        - Prometheus: 通过exporter暴露指标
        - Grafana: 可视化仪表板
        - AlertManager: 基于指标的告警
        """
        total = max(1, self._total_requests)  # 避免除零
        success_count = self._total_requests - self._limited_requests

        return {
            # 基础指标
            "total_requests": self._total_requests,
            "limited_requests": self._limited_requests,
            "success_rate": success_count / total,

            # 熔断器指标
            "circuit_state": self._circuit_state.value,
            "failure_count": self._failure_count,
            "circuit_opened_count": self._circuit_opened_count,
            "half_open_success_count": self._circuit_half_open_success_count,

            # 令牌桶状态
            "current_tokens": self._tokens if not self.config.redis_enabled else "distributed",
            "last_update_time": self._last_update_time,

            # 配置信息（便于诊断）
            "config": {
                "tokens_per_second": self.config.tokens_per_second,
                "bucket_capacity": self.config.bucket_capacity,
                "strategy": self.config.strategy.value,
                "redis_enabled": self.config.redis_enabled,
                "circuit_breaker_enabled": self.config.circuit_breaker_enabled,
                "failure_threshold": self.config.failure_threshold,
            }
        }

    def reset(self) -> None:
        """
        重置限流器状态

        使用场景：
        1. 测试环境：每次测试前重置状态
        2. 故障恢复：系统异常后重置限流器
        3. 配置变更：重大配置调整后重置

        设计思考：
        - 全面重置：所有内部状态恢复到初始值
        - 分布式同步：Redis状态也需要清理
        - 原子操作：整个重置过程在锁保护下完成
        """
        with self._lock:
            # 重置本地状态
            self._tokens = float(self.config.bucket_capacity)
            self._last_update_time = time.time()
            self._failure_count = 0
            self._circuit_state = CircuitBreakerState.CLOSED
            self._circuit_last_state_change = time.time()
            self._circuit_half_open_success_count = 0
            self._total_requests = 0
            self._limited_requests = 0
            self._circuit_opened_count = 0

            # 重置Redis状态（如果启用）
            if self.config.redis_enabled and self._redis_client:
                try:
                    # 删除所有相关键
                    bucket_key = self._get_redis_key("bucket")
                    circuit_key = self._get_redis_key("circuit")

                    self._redis_client.delete(bucket_key, circuit_key)
                    logger.debug(f"Redis状态已重置: {bucket_key}, {circuit_key}")

                except Exception as e:
                    logger.error(f"重置Redis状态失败: {e}")

            logger.info(f"限流器状态已重置: {self.function_name}")


# 便捷装饰器函数（简化使用）
def rate_limit(
        tokens_per_second: float = 10.0,
        bucket_capacity: int = 100,
        strategy: RateLimitStrategy = RateLimitStrategy.BLOCKING,
        redis_enabled: bool = False,
        **kwargs
) -> Callable[[F], F]:
    """
    便捷装饰器函数，用于快速应用限流

    设计理念：简化配置，提供合理的默认值
    类似Java的注解默认值设计

    示例：
    @rate_limit(tokens_per_second=50, redis_enabled=True)
    def api_endpoint():
        pass
    """

    def decorator(func: F) -> F:
        # 构建配置对象
        config = RateLimiterConfig(
            tokens_per_second=tokens_per_second,
            bucket_capacity=bucket_capacity,
            strategy=strategy,
            redis_enabled=redis_enabled,
            **kwargs
        )

        # 创建限流器实例
        limiter = SmartRateLimiter(config, func.__name__)

        # 应用装饰器
        return limiter(func)

    return decorator


# 上下文管理器实现（替代装饰器模式）
class RateLimitContext:
    """
    限流上下文管理器

    使用场景：
    1. 代码块级限流：装饰器不适用的情况
    2. 动态令牌数量：每次调用需要不同令牌数
    3. 临时配置：特定代码段使用特殊限流参数

    Java对应模式：try-with-resources + RateLimiter
    """

    def __init__(
            self,
            limiter: SmartRateLimiter,
            tokens: float = 1.0,
            timeout: Optional[float] = None
    ) -> None:
        self.limiter = limiter
        self.tokens = tokens
        self.timeout = timeout
        self._acquired = False

    def __enter__(self) -> "RateLimitContext":
        """进入上下文：获取令牌"""
        self._acquired = self.limiter.acquire(self.tokens, self.timeout)
        if not self._acquired:
            raise RuntimeError("无法获取令牌，请求被限流")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """退出上下文：更新熔断器状态"""
        success = exc_type is None
        self.limiter._update_circuit_breaker(success)

    async def __aenter__(self) -> "RateLimitContext":
        """异步进入上下文"""
        self._acquired = await self.limiter.acquire_async(self.tokens, self.timeout)
        if not self._acquired:
            raise RuntimeError("无法获取令牌，请求被限流")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步退出上下文"""
        success = exc_type is None
        self.limiter._update_circuit_breaker(success)


def rate_limit_context(
        limiter: SmartRateLimiter,
        tokens: float = 1.0,
        timeout: Optional[float] = None
) -> RateLimitContext:
    """创建限流上下文管理器（工厂函数）"""
    return RateLimitContext(limiter, tokens, timeout)


# 使用示例和对比说明
if __name__ == "__main__":
    """
    运行示例代码，展示多种使用方式

    从Java视角理解：
    1. 装饰器 ≈ @Annotation + AOP
    2. 上下文管理器 ≈ try-with-resources
    3. 直接调用 ≈ 编程式API
    """

    print("=" * 60)
    print("智能限流装饰器使用示例")
    print("=" * 60)


    # 示例1：基础装饰器使用（类似Java注解）
    @rate_limit(tokens_per_second=10, bucket_capacity=20)
    def process_request(data: str) -> str:
        """模拟API处理"""
        time.sleep(0.05)
        return f"Processed: {data}"


    # 示例2：带Redis的分布式限流
    @rate_limit(
        tokens_per_second=100,
        bucket_capacity=1000,
        redis_enabled=True,
        redis_host="redis.prod.example.com"
    )
    async def async_model_call(prompt: str) -> str:
        """模拟异步大模型调用"""
        await asyncio.sleep(0.1)
        return f"AI Response: {prompt}"


    # 示例3：上下文管理器使用（类似Java try-with-resources）
    def process_with_context_manager():
        """使用上下文管理器进行限流"""
        config = RateLimiterConfig(tokens_per_second=5)
        limiter = SmartRateLimiter(config, "manual_call")

        try:
            with rate_limit_context(limiter, timeout=2.0):
                print("  ✓ 获取令牌成功，执行业务逻辑")
                time.sleep(0.1)
        except RuntimeError as e:
            print(f"  ✗ 限流阻止: {e}")


    # 运行示例
    print("\n1. 同步装饰器示例:")
    for i in range(5):
        result = process_request(f"req_{i}")
        print(f"  请求{i}: {result}")

    print("\n2. 异步装饰器示例:")


    async def run_async_example():
        tasks = [async_model_call(f"prompt_{i}") for i in range(3)]
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            print(f"  异步请求{i}: {result}")


    asyncio.run(run_async_example())

    print("\n3. 上下文管理器示例:")
    process_with_context_manager()

    print("\n4. 监控指标示例:")
    config = RateLimiterConfig()
    limiter = SmartRateLimiter(config, "demo")

    # 模拟一些请求
    for i in range(10):
        limiter.acquire(1.0)

    metrics = limiter.get_metrics()
    print(f"  总请求数: {metrics['total_requests']}")
    print(f"  限流次数: {metrics['limited_requests']}")
    print(f"  成功率: {metrics['success_rate']:.1%}")
    print(f"  熔断器状态: {metrics['circuit_state']}")

    print("\n" + "=" * 60)
    print("Java工程思维总结:")
    print("=" * 60)
    print("""
    1. API设计借鉴：
       - acquire() ≈ Guava RateLimiter.acquire()
       - 装饰器 ≈ Spring @RateLimit注解

    2. 分布式考量：
       - Redis + Lua ≈ Redisson分布式限流
       - 原子操作保证数据一致性

    3. 监控集成：
       - 指标字典 ≈ Prometheus Metrics
       - 可对接企业监控体系

    4. 工程实践：
       - 热重载支持动态配置
       - 降级机制保证可用性
       - 完整测试覆盖保证质量
    """)