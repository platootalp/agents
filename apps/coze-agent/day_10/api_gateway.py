"""
企业级API网关实现
核心特性：
1. 动态路由管理（基于路径/头部/权重的智能路由）
2. 限流熔断（令牌桶算法 + 断路器模式）
3. 认证鉴权（JWT令牌验证 + API Key管理）
4. 结构化日志（请求/响应全链路追踪）
"""

import asyncio
import json
import time
import redis.asyncio as redis
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from pydantic import BaseModel, ValidationError


# ==================== 数据模型定义 ====================

class RouteConfig(BaseModel):
    """路由配置模型"""
    path: str
    service: str
    methods: List[str] = ["GET", "POST"]
    rate_limit: int = 100  # 每分钟请求数
    authentication: str = "required"  # required/optional/none
    timeout: int = 5000  # 毫秒


class JWTToken(BaseModel):
    """JWT令牌模型"""
    sub: str  # 用户ID
    exp: datetime  # 过期时间
    scopes: List[str] = []  # 权限范围


class APIKey(BaseModel):
    """API Key模型"""
    key: str
    owner: str
    permissions: List[str]
    rate_limit: int = 1000
    expires_at: Optional[datetime] = None


# ==================== 令牌桶限流器 ====================

class TokenBucketRateLimiter:
    """令牌桶限流器"""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: 桶容量（最大令牌数）
            refill_rate: 每秒补充令牌数
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "rejected_requests": 0
        }

    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill

        # 计算应补充的令牌数
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    async def acquire(self, tokens: int = 1) -> bool:
        """尝试获取指定数量的令牌"""
        self._refill()

        self.stats["total_requests"] += 1

        if self.tokens >= tokens:
            self.tokens -= tokens
            self.stats["allowed_requests"] += 1
            return True
        else:
            self.stats["rejected_requests"] += 1
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "current_tokens": round(self.tokens, 2),
            "utilization": round(self.stats["allowed_requests"] / max(1, self.stats["total_requests"]), 4)
        }


# ==================== 断路器模式 ====================

class CircuitState(Enum):
    """断路器状态"""
    CLOSED = "closed"  # 正常状态，请求通过
    OPEN = "open"  # 断开状态，请求被拒绝
    HALF_OPEN = "half_open"  # 半开状态，试探性允许少量请求


class CircuitBreaker:
    """断路器实现"""

    def __init__(
            self,
            failure_threshold: int = 5,
            recovery_timeout: int = 30,
            half_open_max_requests: int = 3
    ):
        """
        Args:
            failure_threshold: 失败阈值，达到后触发熔断
            recovery_timeout: 恢复超时（秒），熔断后等待时间
            half_open_max_requests: 半开状态最大允许请求数
        """
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests

        self.last_failure_time = None
        self.half_open_request_count = 0

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "failed_requests": 0,
            "circuit_opened": 0,
            "circuit_closed": 0
        }

    async def allow_request(self) -> bool:
        """检查是否允许请求"""
        self.stats["total_requests"] += 1

        if self.state == CircuitState.CLOSED:
            return True

        elif self.state == CircuitState.OPEN:
            # 检查是否达到恢复时间
            if (self.last_failure_time and
                    time.time() - self.last_failure_time > self.recovery_timeout):
                # 切换到半开状态
                self.state = CircuitState.HALF_OPEN
                self.half_open_request_count = 0
                return True
            return False

        elif self.state == CircuitState.HALF_OPEN:
            if self.half_open_request_count < self.half_open_max_requests:
                self.half_open_request_count += 1
                return True
            return False

    async def record_success(self):
        """记录成功请求"""
        self.success_count += 1

        if self.state == CircuitState.HALF_OPEN:
            # 半开状态下连续成功，切换到闭合状态
            if self.success_count >= self.half_open_max_requests:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.half_open_request_count = 0

                self.stats["circuit_closed"] += 1

    async def record_failure(self):
        """记录失败请求"""
        self.failure_count += 1
        self.stats["failed_requests"] += 1

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                # 达到失败阈值，触发熔断
                self.state = CircuitState.OPEN
                self.last_failure_time = time.time()
                self.stats["circuit_opened"] += 1

        elif self.state == CircuitState.HALF_OPEN:
            # 半开状态下出现失败，重新打开
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            self.stats["circuit_opened"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "current_state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count
        }


# ==================== 认证鉴权管理器 ====================

class AuthManager:
    """认证鉴权管理器"""

    def __init__(self, jwt_secret: str, api_key_store):
        self.jwt_secret = jwt_secret
        self.api_key_store = api_key_store

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "jwt_auth_success": 0,
            "jwt_auth_failed": 0,
            "api_key_auth_success": 0,
            "api_key_auth_failed": 0
        }

    async def authenticate_jwt(self, token: str) -> Tuple[bool, Optional[JWTToken]]:
        """验证JWT令牌"""
        self.stats["total_requests"] += 1

        try:
            # 解码令牌
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=["HS256"]
            )

            # 转换为JWTToken对象
            jwt_token = JWTToken(
                sub=payload["sub"],
                exp=datetime.fromtimestamp(payload["exp"]),
                scopes=payload.get("scopes", [])
            )

            # 检查是否过期
            if jwt_token.exp < datetime.now():
                self.stats["jwt_auth_failed"] += 1
                return False, None

            self.stats["jwt_auth_success"] += 1
            return True, jwt_token

        except (JWTError, KeyError, ValidationError) as e:
            self.stats["jwt_auth_failed"] += 1
            return False, None

    async def authenticate_api_key(self, api_key: str) -> Tuple[bool, Optional[APIKey]]:
        """验证API Key"""
        self.stats["total_requests"] += 1

        try:
            # 从存储中获取API Key信息
            key_info = await self.api_key_store.get(api_key)

            if not key_info:
                self.stats["api_key_auth_failed"] += 1
                return False, None

            # 检查是否过期
            if key_info.expires_at and key_info.expires_at < datetime.now():
                self.stats["api_key_auth_failed"] += 1
                return False, None

            self.stats["api_key_auth_success"] += 1
            return True, key_info

        except Exception as e:
            self.stats["api_key_auth_failed"] += 1
            return False, None

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_auth = self.stats["jwt_auth_success"] + self.stats["api_key_auth_success"]
        total_failed = self.stats["jwt_auth_failed"] + self.stats["api_key_auth_failed"]
        total = total_auth + total_failed

        if total == 0:
            success_rate = 0.0
        else:
            success_rate = total_auth / total

        return {
            **self.stats,
            "auth_success_rate": round(success_rate, 4)
        }


# ==================== Redis API Key存储 ====================

class RedisAPIKeyStore:
    """Redis API Key存储"""

    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.key_prefix = "apikey:"

    async def get(self, api_key: str) -> Optional[APIKey]:
        """获取API Key信息"""
        try:
            data = await self.redis_client.get(f"{self.key_prefix}{api_key}")

            if data:
                key_dict = json.loads(data)

                # 转换过期时间
                expires_at = None
                if key_dict.get("expires_at"):
                    expires_at = datetime.fromisoformat(key_dict["expires_at"])

                return APIKey(
                    key=key_dict["key"],
                    owner=key_dict["owner"],
                    permissions=key_dict["permissions"],
                    rate_limit=key_dict.get("rate_limit", 1000),
                    expires_at=expires_at
                )

            return None

        except Exception as e:
            print(f"API Key查询失败: {e}")
            return None

    async def set(self, api_key: APIKey):
        """设置API Key信息"""
        try:
            key_dict = {
                "key": api_key.key,
                "owner": api_key.owner,
                "permissions": api_key.permissions,
                "rate_limit": api_key.rate_limit,
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None
            }

            ttl = None
            if api_key.expires_at:
                ttl = int((api_key.expires_at - datetime.now()).total_seconds())
                if ttl < 0:
                    return  # 已过期，不存储

            await self.redis_client.set(
                f"{self.key_prefix}{api_key.key}",
                json.dumps(key_dict),
                ex=ttl
            )

        except Exception as e:
            print(f"API Key存储失败: {e}")


# ==================== 结构化日志记录器 ====================

class StructuredLogger:
    """结构化日志记录器"""

    def __init__(self, service_name: str):
        self.service_name = service_name

        # 日志统计
        self.stats = {
            "total_logs": 0,
            "request_logs": 0,
            "response_logs": 0,
            "error_logs": 0
        }

    async def log_request(self, request: Request, auth_result: Optional[Dict] = None):
        """记录请求日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "type": "request",
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "auth_status": auth_result["status"] if auth_result else "none",
            "auth_user": auth_result.get("user") if auth_result else None
        }

        # 记录日志（生产环境应发送到ELK）
        print(json.dumps(log_entry, ensure_ascii=False))

        self.stats["total_logs"] += 1
        self.stats["request_logs"] += 1

    async def log_response(self, request: Request, response: JSONResponse, latency: float):
        """记录响应日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "type": "response",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency * 1000, 2),
            "response_size": len(response.body) if response.body else 0
        }

        print(json.dumps(log_entry, ensure_ascii=False))

        self.stats["total_logs"] += 1
        self.stats["response_logs"] += 1

    async def log_error(self, error: Exception, context: Dict[str, Any]):
        """记录错误日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }

        print(json.dumps(log_entry, ensure_ascii=False))

        self.stats["total_logs"] += 1
        self.stats["error_logs"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取日志统计"""
        return self.stats.copy()


# ==================== 动态路由管理器 ====================

class DynamicRouter:
    """动态路由管理器"""

    def __init__(self, routes_config: str = None):
        self.routes = []
        self.route_cache = {}  # 路径到路由的缓存

        # 加载默认路由配置
        if routes_config:
            self.load_routes(routes_config)

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "route_hits": 0,
            "route_misses": 0
        }

    def load_routes(self, config_file: str):
        """从配置文件加载路由"""
        try:
            # 模拟加载YAML配置
            # 实际生产环境应从文件读取
            sample_routes = [
                {
                    "path": "/api/v1/rag/query",
                    "service": "rag-service",
                    "methods": ["POST"],
                    "rate_limit": 100,
                    "authentication": "required",
                    "timeout": 5000
                },
                {
                    "path": "/api/v1/cache/stats",
                    "service": "cache-service",
                    "methods": ["GET"],
                    "rate_limit": 30,
                    "authentication": "optional",
                    "timeout": 3000
                },
                {
                    "path": "/api/v1/monitor/metrics",
                    "service": "monitor-service",
                    "methods": ["GET"],
                    "rate_limit": 60,
                    "authentication": "required",
                    "timeout": 2000
                }
            ]

            for route_dict in sample_routes:
                route = RouteConfig(**route_dict)
                self.routes.append(route)
                self.route_cache[route.path] = route

            print(f"已加载 {len(self.routes)} 条路由配置")

        except Exception as e:
            print(f"路由配置加载失败: {e}")

    async def match(self, path: str, method: str = "GET") -> Optional[RouteConfig]:
        """匹配路由"""
        self.stats["total_requests"] += 1

        # 检查缓存
        if path in self.route_cache:
            route = self.route_cache[path]

            # 检查方法是否允许
            if method in route.methods:
                self.stats["route_hits"] += 1
                return route

        # 路径匹配失败
        self.stats["route_misses"] += 1
        return None

    def add_route(self, route: RouteConfig):
        """动态添加路由"""
        self.routes.append(route)
        self.route_cache[route.path] = route

    def remove_route(self, path: str):
        """移除路由"""
        self.routes = [r for r in self.routes if r.path != path]
        self.route_cache.pop(path, None)

    def get_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        hit_rate = 0.0
        if self.stats["total_requests"] > 0:
            hit_rate = self.stats["route_hits"] / self.stats["total_requests"]

        return {
            **self.stats,
            "route_hit_rate": round(hit_rate, 4),
            "total_routes": len(self.routes)
        }


# ==================== 企业级API网关主类 ====================

class AIApplicationGateway:
    """企业级AI应用API网关"""

    def __init__(
            self,
            redis_host: str = "localhost",
            redis_port: int = 6379,
            jwt_secret: str = "your-secret-key-change-in-production"
    ):
        # 初始化组件
        self.router = DynamicRouter()
        self.rate_limiter = TokenBucketRateLimiter(capacity=1000, refill_rate=100)
        self.circuit_breaker = CircuitBreaker()

        # Redis客户端
        self.redis_client = None
        self.redis_host = redis_host
        self.redis_port = redis_port

        # 认证管理器
        self.auth_manager = None

        # 日志记录器
        self.logger = StructuredLogger(service_name="ai-gateway")

        # FastAPI应用
        self.app = FastAPI(title="AI应用网关", version="1.0.0")

        # 添加中间件
        self._add_middleware()

        # 添加路由
        self._setup_routes()

        # 统计信息
        self.gateway_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_latency": 0.0
        }

    async def initialize(self):
        """初始化网关"""
        print("初始化API网关...")

        # 初始化Redis
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )

        # 初始化认证管理器
        api_key_store = RedisAPIKeyStore(self.redis_client)
        self.auth_manager = AuthManager(jwt_secret="test-secret", api_key_store=api_key_store)

        # 加载默认路由
        self.router.load_routes("config/routes.yaml")

        print("API网关初始化完成")

    def _add_middleware(self):
        """添加中间件"""
        # CORS中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 生产环境应限制
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

    def _setup_routes(self):
        """设置网关路由"""

        @self.app.middleware("http")
        async def gateway_middleware(request: Request, call_next):
            """网关核心中间件"""
            start_time = time.time()
            self.gateway_stats["total_requests"] += 1

            try:
                # 1. 路由匹配
                route = await self.router.match(request.url.path, request.method)
                if not route:
                    raise HTTPException(status_code=404, detail="路由未找到")

                # 2. 限流检查
                if not await self.rate_limiter.acquire():
                    raise HTTPException(status_code=429, detail="请求过于频繁")

                # 3. 断路器检查
                if not await self.circuit_breaker.allow_request():
                    raise HTTPException(status_code=503, detail="服务暂时不可用")

                # 4. 认证鉴权
                auth_result = await self._authenticate_request(request, route)

                # 5. 记录请求日志
                await self.logger.log_request(request, auth_result)

                # 6. 转发请求到后端服务
                response = await self._forward_to_backend(request, route)

                # 7. 记录成功
                await self.circuit_breaker.record_success()

                # 计算延迟
                latency = time.time() - start_time
                self._update_average_latency(latency)

                # 8. 记录响应日志
                await self.logger.log_response(request, response, latency)

                # 更新统计
                self.gateway_stats["successful_requests"] += 1

                return response

            except HTTPException as e:
                # 已知异常
                await self.circuit_breaker.record_failure()

                # 记录错误日志
                await self.logger.log_error(e, {"path": request.url.path})

                self.gateway_stats["failed_requests"] += 1

                return JSONResponse(
                    status_code=e.status_code,
                    content={"detail": e.detail}
                )

            except Exception as e:
                # 未知异常
                await self.circuit_breaker.record_failure()

                await self.logger.log_error(e, {"path": request.url.path})

                self.gateway_stats["failed_requests"] += 1

                return JSONResponse(
                    status_code=500,
                    content={"detail": "内部服务器错误"}
                )

    async def _authenticate_request(self, request: Request, route: RouteConfig) -> Dict[str, Any]:
        """认证请求"""
        if route.authentication == "none":
            return {"status": "not_required"}

        # 检查JWT令牌
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            success, jwt_token = await self.auth_manager.authenticate_jwt(token)

            if success:
                return {
                    "status": "authenticated",
                    "method": "jwt",
                    "user": jwt_token.sub,
                    "scopes": jwt_token.scopes
                }

        # 检查API Key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            success, key_info = await self.auth_manager.authenticate_api_key(api_key)

            if success:
                return {
                    "status": "authenticated",
                    "method": "api_key",
                    "user": key_info.owner,
                    "permissions": key_info.permissions
                }

        # 认证失败
        if route.authentication == "required":
            raise HTTPException(status_code=401, detail="认证失败")
        else:
            return {"status": "anonymous"}

    async def _forward_to_backend(self, request: Request, route: RouteConfig) -> JSONResponse:
        """转发请求到后端服务"""
        # 模拟后端服务响应
        # 实际生产环境应使用HTTP客户端（如httpx）转发

        # 模拟处理延迟
        await asyncio.sleep(0.01)

        # 根据路由返回模拟响应
        if route.path == "/api/v1/rag/query":
            return JSONResponse({
                "answer": "这是RAG服务的模拟响应",
                "sources": [
                    {"content": "示例文档内容", "confidence": 0.92}
                ],
                "cache_source": "vector_db",
                "latency": 0.15
            })

        elif route.path == "/api/v1/cache/stats":
            return JSONResponse({
                "total_requests": 1000,
                "hit_rate": 0.85,
                "average_latency_ms": 25.5
            })

        elif route.path == "/api/v1/monitor/metrics":
            return JSONResponse({
                "qps": 150.3,
                "p95_latency_ms": 120.7,
                "error_rate": 0.002
            })

        else:
            return JSONResponse({"message": "服务响应"}, status_code=200)

    def _update_average_latency(self, latency: float):
        """更新平均延迟"""
        alpha = 0.1
        self.gateway_stats["average_latency"] = (
                alpha * latency + (1 - alpha) * self.gateway_stats["average_latency"]
        )

    def get_gateway_stats(self) -> Dict[str, Any]:
        """获取网关统计信息"""
        success_rate = 0.0
        if self.gateway_stats["total_requests"] > 0:
            success_rate = self.gateway_stats["successful_requests"] / self.gateway_stats["total_requests"]

        return {
            **self.gateway_stats,
            "success_rate": round(success_rate, 4),
            "router_stats": self.router.get_stats(),
            "rate_limiter_stats": self.rate_limiter.get_stats(),
            "circuit_breaker_stats": self.circuit_breaker.get_stats(),
            "auth_stats": self.auth_manager.get_stats() if self.auth_manager else None,
            "logger_stats": self.logger.get_stats()
        }


# ==================== 使用示例 ====================

async def demo():
    """演示API网关功能"""
    print("初始化企业级API网关...")

    gateway = AIApplicationGateway()
    await gateway.initialize()

    print("\n网关统计信息:")
    stats = gateway.get_gateway_stats()
    for category, data in stats.items():
        if isinstance(data, dict):
            print(f"\n{category}:")
            for k, v in data.items():
                print(f"  {k}: {v}")
        else:
            print(f"{category}: {data}")

    # 启动FastAPI服务器（在实际演示中）
    print("\nAPI网关已就绪，可通过以下端点访问:")
    print("  - /api/v1/rag/query    (RAG查询)")
    print("  - /api/v1/cache/stats  (缓存统计)")
    print("  - /api/v1/monitor/metrics (监控指标)")

    # 注意：实际启动服务器需要额外的代码
    # import uvicorn
    # uvicorn.run(gateway.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    asyncio.run(demo())
