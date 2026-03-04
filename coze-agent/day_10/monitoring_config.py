"""
企业级AI应用监控体系配置
包含：
1. Prometheus指标配置
2. Grafana仪表板配置
3. ELK Stack日志配置
4. AlertManager告警规则
"""

import json
from typing import Dict, List, Any


# ==================== Prometheus指标配置 ====================

PROMETHEUS_CONFIG = {
    "global": {
        "scrape_interval": "15s",
        "evaluation_interval": "15s"
    },
    "scrape_configs": [
        {
            "job_name": "ai-gateway",
            "static_configs": [
                {
                    "targets": ["ai-gateway:8000"],
                    "labels": {
                        "service": "ai-gateway",
                        "env": "production"
                    }
                }
            ],
            "metrics_path": "/metrics"
        },
        {
            "job_name": "rag-service",
            "static_configs": [
                {
                    "targets": ["rag-service:8001"],
                    "labels": {
                        "service": "rag-service",
                        "env": "production"
                    }
                }
            ]
        },
        {
            "job_name": "cache-service",
            "static_configs": [
                {
                    "targets": ["cache-service:8002"],
                    "labels": {
                        "service": "cache-service",
                        "env": "production"
                    }
                }
            ]
        },
        {
            "job_name": "node-exporter",
            "static_configs": [
                {"targets": ["node-exporter:9100"]}
            ]
        },
        {
            "job_name": "redis-exporter",
            "static_configs": [
                {"targets": ["redis-exporter:9121"]}
            ]
        }
    ],
    "alerting": {
        "alertmanagers": [
            {
                "static_configs": [
                    {"targets": ["alertmanager:9093"]}
                ]
            }
        ]
    },
    "rule_files": [
        "/etc/prometheus/rules/ai_app_rules.yml"
    ]
}


# ==================== Grafana仪表板配置 ====================

GRAFANA_DASHBOARD = {
    "dashboard": {
        "title": "AI应用监控总览",
        "panels": [
            {
                "title": "请求QPS",
                "targets": [
                    {
                        "expr": "rate(ai_app_requests_total[5m])",
                        "legendFormat": "{{service}}"
                    }
                ],
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
            },
            {
                "title": "P95延迟",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(ai_app_request_duration_seconds_bucket[5m]))",
                        "legendFormat": "{{service}}"
                    }
                ],
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
            },
            {
                "title": "缓存命中率",
                "targets": [
                    {
                        "expr": "rate(ai_app_cache_hits_total[5m]) / rate(ai_app_cache_requests_total[5m])",
                        "legendFormat": "{{cache_level}}"
                    }
                ],
                "type": "stat",
                "gridPos": {"h": 8, "w": 8, "x": 0, "y": 8}
            },
            {
                "title": "RAG检索准确率",
                "targets": [
                    {
                        "expr": "ai_app_rag_accuracy",
                        "legendFormat": "准确率"
                    }
                ],
                "type": "gauge",
                "gridPos": {"h": 8, "w": 8, "x": 8, "y": 8}
            },
            {
                "title": "系统资源使用率",
                "targets": [
                    {
                        "expr": "node_memory_MemFree_bytes / node_memory_MemTotal_bytes * 100",
                        "legendFormat": "内存空闲率"
                    },
                    {
                        "expr": "100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
                        "legendFormat": "CPU使用率"
                    }
                ],
                "type": "graph",
                "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
            }
        ],
        "time": {
            "from": "now-1h",
            "to": "now"
        },
        "refresh": "10s"
    },
    "overwrite": True
}


# ==================== ELK Stack日志配置 ====================

ELK_LOGSTASH_CONFIG = """
input {
  # 从文件收集日志
  file {
    path => "/var/log/ai-app/*.log"
    type => "ai-app"
    sincedb_path => "/dev/null"
    start_position => "beginning"
  }
  
  # 从TCP端口收集日志
  tcp {
    port => 5000
    codec => json_lines
    type => "json-logs"
  }
}

filter {
  # 解析JSON日志
  if [type] == "ai-app" {
    json {
      source => "message"
    }
    
    # 添加时间戳
    date {
      match => ["timestamp", "ISO8601"]
    }
    
    # 添加服务标签
    if [service] {
      mutate {
        add_field => { "[@metadata][service]" => "%{service}" }
      }
    }
  }
}

output {
  # 输出到Elasticsearch
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "ai-app-logs-%{+YYYY.MM.dd}"
  }
  
  # 同时输出到控制台（开发环境）
  stdout {
    codec => rubydebug
  }
}
"""


# ==================== AlertManager告警规则 ====================

ALERT_RULES = {
    "groups": [
        {
            "name": "ai-application-alerts",
            "rules": [
                {
                    "alert": "RAGAccuracyBelowThreshold",
                    "expr": "ai_app_rag_accuracy < 0.85",
                    "for": "5m",
                    "labels": {
                        "severity": "warning",
                        "service": "rag-service"
                    },
                    "annotations": {
                        "summary": "RAG检索准确率低于85%阈值",
                        "description": "RAG服务 {{ $labels.service }} 的检索准确率已降至 {{ $value }}，低于85%的阈值。"
                    }
                },
                {
                    "alert": "CacheHitRateLow",
                    "expr": "rate(ai_app_cache_hits_total[5m]) / rate(ai_app_cache_requests_total[5m]) < 0.7",
                    "for": "5m",
                    "labels": {
                        "severity": "warning",
                        "service": "cache-service"
                    },
                    "annotations": {
                        "summary": "缓存命中率低于70%",
                        "description": "缓存服务 {{ $labels.service }} 的命中率已降至 {{ $value | humanizePercentage }}，低于70%的阈值。"
                    }
                },
                {
                    "alert": "GatewayHighErrorRate",
                    "expr": "rate(ai_app_requests_total{status=~\"5..\"}[5m]) / rate(ai_app_requests_total[5m]) > 0.01",
                    "for": "2m",
                    "labels": {
                        "severity": "critical",
                        "service": "ai-gateway"
                    },
                    "annotations": {
                        "summary": "API网关错误率超过1%",
                        "description": "API网关 {{ $labels.service }} 的错误率已升至 {{ $value | humanizePercentage }}，超过1%的阈值。"
                    }
                },
                {
                    "alert": "HighResponseLatency",
                    "expr": "histogram_quantile(0.95, rate(ai_app_request_duration_seconds_bucket[5m])) > 2",
                    "for": "3m",
                    "labels": {
                        "severity": "warning",
                        "service": "all"
                    },
                    "annotations": {
                        "summary": "P95响应延迟超过2秒",
                        "description": "服务 {{ $labels.service }} 的P95响应延迟已升至 {{ $value }} 秒，超过2秒的阈值。"
                    }
                },
                {
                    "alert": "ServiceDown",
                    "expr": "up == 0",
                    "for": "1m",
                    "labels": {
                        "severity": "critical"
                    },
                    "annotations": {
                        "summary": "服务 {{ $labels.job }} 已下线",
                        "description": "监控目标 {{ $labels.instance }} 的服务 {{ $labels.job }} 已超过1分钟不可用。"
                    }
                }
            ]
        }
    ]
}


# ==================== Prometheus指标定义类 ====================

class AIMetrics:
    """AI应用监控指标定义"""
    
    def __init__(self):
        # 指标定义字典
        self.metrics = {
            "requests_total": {
                "type": "counter",
                "help": "Total number of requests",
                "labels": ["service", "endpoint", "status"]
            },
            "request_duration_seconds": {
                "type": "histogram",
                "help": "Request duration in seconds",
                "labels": ["service", "endpoint"],
                "buckets": [0.1, 0.5, 1.0, 2.0, 5.0]
            },
            "cache_hits_total": {
                "type": "counter",
                "help": "Total cache hits",
                "labels": ["cache_level", "service"]
            },
            "cache_requests_total": {
                "type": "counter",
                "help": "Total cache requests",
                "labels": ["cache_level", "service"]
            },
            "rag_accuracy": {
                "type": "gauge",
                "help": "RAG retrieval accuracy",
                "labels": ["service"]
            },
            "vector_search_latency_seconds": {
                "type": "histogram",
                "help": "Vector search latency in seconds",
                "labels": ["service"],
                "buckets": [0.05, 0.1, 0.2, 0.5, 1.0]
            },
            "active_connections": {
                "type": "gauge",
                "help": "Number of active connections",
                "labels": ["service"]
            },
            "error_rate": {
                "type": "gauge",
                "help": "Error rate percentage",
                "labels": ["service", "error_type"]
            }
        }
    
    def get_metrics_promql(self) -> Dict[str, str]:
        """获取PromQL查询表达式"""
        return {
            "qps": 'rate(ai_app_requests_total[5m])',
            "error_rate": 'rate(ai_app_requests_total{status=~"5.."}[5m]) / rate(ai_app_requests_total[5m])',
            "p95_latency": 'histogram_quantile(0.95, rate(ai_app_request_duration_seconds_bucket[5m]))',
            "cache_hit_rate": 'rate(ai_app_cache_hits_total[5m]) / rate(ai_app_cache_requests_total[5m])',
            "memory_usage": 'node_memory_MemTotal_bytes - node_memory_MemFree_bytes',
            "cpu_usage": '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
        }
    
    def generate_prometheus_config(self) -> str:
        """生成Prometheus配置YAML"""
        import yaml
        
        config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "ai-applications",
                    "static_configs": [
                        {
                            "targets": [
                                "ai-gateway:8000",
                                "rag-service:8001",
                                "cache-service:8002"
                            ]
                        }
                    ]
                }
            ]
        }
        
        return yaml.dump(config, default_flow_style=False)


# ==================== 监控体系主配置类 ====================

class AIMonitoringSystem:
    """AI应用监控体系主配置"""
    
    def __init__(self):
        self.metrics = AIMetrics()
        
        # 配置存储
        self.configs = {
            "prometheus": PROMETHEUS_CONFIG,
            "grafana": GRAFANA_DASHBOARD,
            "elk": ELK_LOGSTASH_CONFIG,
            "alerts": ALERT_RULES
        }
    
    def get_config(self, component: str) -> Dict[str, Any]:
        """获取组件配置"""
        return self.configs.get(component, {})
    
    def save_configs(self, output_dir: str = "./monitoring_configs"):
        """保存所有配置文件"""
        import os
        import yaml
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存Prometheus配置
        with open(os.path.join(output_dir, "prometheus.yml"), "w") as f:
            yaml.dump(self.configs["prometheus"], f, default_flow_style=False)
        
        # 保存Grafana仪表板配置
        with open(os.path.join(output_dir, "grafana_dashboard.json"), "w") as f:
            json.dump(self.configs["grafana"], f, indent=2)
        
        # 保存Logstash配置
        with open(os.path.join(output_dir, "logstash.conf"), "w") as f:
            f.write(self.configs["elk"])
        
        # 保存告警规则
        with open(os.path.join(output_dir, "alert_rules.yml"), "w") as f:
            yaml.dump(self.configs["alerts"], f, default_flow_style=False)
        
        # 保存指标定义
        metrics_def = {
            "metrics": self.metrics.metrics,
            "promql_queries": self.metrics.get_metrics_promql()
        }
        
        with open(os.path.join(output_dir, "metrics_definitions.json"), "w") as f:
            json.dump(metrics_def, f, indent=2)
        
        print(f"配置文件已保存到: {output_dir}")
    
    def generate_docker_compose(self) -> str:
        """生成Docker Compose配置文件"""
        docker_compose = {
            "version": "3.8",
            "services": {
                "prometheus": {
                    "image": "prom/prometheus:latest",
                    "ports": ["9090:9090"],
                    "volumes": [
                        "./monitoring_configs/prometheus.yml:/etc/prometheus/prometheus.yml",
                        "./monitoring_configs/alert_rules.yml:/etc/prometheus/rules/ai_app_rules.yml",
                        "prometheus_data:/prometheus"
                    ],
                    "command": [
                        "--config.file=/etc/prometheus/prometheus.yml",
                        "--storage.tsdb.path=/prometheus",
                        "--web.console.libraries=/etc/prometheus/console_libraries",
                        "--web.console.templates=/etc/prometheus/consoles",
                        "--storage.tsdb.retention.time=30d"
                    ]
                },
                "grafana": {
                    "image": "grafana/grafana:latest",
                    "ports": ["3000:3000"],
                    "volumes": [
                        "grafana_data:/var/lib/grafana",
                        "./monitoring_configs/grafana_dashboard.json:/etc/grafana/provisioning/dashboards/ai_app_dashboard.json"
                    ],
                    "environment": {
                        "GF_SECURITY_ADMIN_PASSWORD": "admin"
                    }
                },
                "elasticsearch": {
                    "image": "elasticsearch:8.11.0",
                    "ports": ["9200:9200"],
                    "environment": {
                        "discovery.type": "single-node",
                        "xpack.security.enabled": "false"
                    },
                    "volumes": ["elasticsearch_data:/usr/share/elasticsearch/data"]
                },
                "logstash": {
                    "image": "logstash:8.11.0",
                    "ports": ["5000:5000"],
                    "volumes": [
                        "./monitoring_configs/logstash.conf:/usr/share/logstash/pipeline/logstash.conf"
                    ],
                    "depends_on": ["elasticsearch"]
                },
                "kibana": {
                    "image": "kibana:8.11.0",
                    "ports": ["5601:5601"],
                    "environment": {
                        "ELASTICSEARCH_HOSTS": "http://elasticsearch:9200"
                    },
                    "depends_on": ["elasticsearch"]
                },
                "alertmanager": {
                    "image": "prom/alertmanager:latest",
                    "ports": ["9093:9093"],
                    "volumes": [
                        "./monitoring_configs/alertmanager.yml:/etc/alertmanager/alertmanager.yml"
                    ]
                },
                "node-exporter": {
                    "image": "prom/node-exporter:latest",
                    "ports": ["9100:9100"],
                    "volumes": ["/proc:/host/proc", "/sys:/host/sys", "/:/rootfs"],
                    "command": [
                        "--path.procfs=/host/proc",
                        "--path.sysfs=/host/sys",
                        "--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)"
                    ]
                },
                "redis-exporter": {
                    "image": "oliver006/redis_exporter:latest",
                    "ports": ["9121:9121"],
                    "environment": {
                        "REDIS_ADDR": "redis:6379"
                    }
                }
            },
            "volumes": {
                "prometheus_data": {},
                "grafana_data": {},
                "elasticsearch_data": {}
            }
        }
        
        import yaml
        return yaml.dump(docker_compose, default_flow_style=False)


# ==================== 使用示例 ====================

def demo():
    """演示监控体系配置"""
    print("企业级AI应用监控体系配置")
    print("=" * 50)
    
    # 创建监控系统
    monitoring = AIMonitoringSystem()
    
    # 显示配置概览
    print("1. Prometheus配置:")
    print(f"   采集间隔: {monitoring.configs['prometheus']['global']['scrape_interval']}")
    print(f"   监控任务数: {len(monitoring.configs['prometheus']['scrape_configs'])}")
    
    print("\n2. Grafana仪表板:")
    print(f"   面板数量: {len(monitoring.configs['grafana']['dashboard']['panels'])}")
    
    print("\n3. AlertManager告警规则:")
    print(f"   规则组数: {len(monitoring.configs['alerts']['groups'])}")
    
    # 生成配置文件
    print("\n生成配置文件中...")
    monitoring.save_configs("./monitoring_configs")
    
    # 生成Docker Compose配置
    docker_compose = monitoring.generate_docker_compose()
    
    print("\nDocker Compose配置已生成，可部署完整监控体系:")
    print("  - Prometheus: http://localhost:9090")
    print("  - Grafana: http://localhost:3000 (admin/admin)")
    print("  - Kibana: http://localhost:5601")
    print("  - AlertManager: http://localhost:9093")
    
    return monitoring


if __name__ == "__main__":
    demo()