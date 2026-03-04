#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产级Agent成本监控面板
生成Token使用和成本分析可视化图表
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
matplotlib.rcParams["axes.unicode_minus"] = False


class CostDashboard:
    """成本监控数据仪表板"""

    def __init__(self, log_file: str = "agent_audit_logs.jsonl"):
        self.log_file = log_file
        self.data = self._load_audit_logs()

    def _load_audit_logs(self) -> List[Dict[str, Any]]:
        """加载审计日志数据"""
        logs = []

        if not os.path.exists(self.log_file):
            print(f"警告: 日志文件不存在 {self.log_file}")
            return logs

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            log_entry = json.loads(line)
                            logs.append(log_entry)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"加载日志文件失败: {e}")

        return logs

    def generate_dashboard(self, output_dir: str = "."):
        """生成完整的仪表板图表"""

        if not self.data:
            print("没有数据可生成图表")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成各个图表
        self._generate_token_usage_chart(output_dir)
        self._generate_cost_breakdown_chart(output_dir)
        self._generate_response_time_chart(output_dir)
        self._generate_daily_summary_chart(output_dir)

        print(f"仪表板图表已生成到目录: {output_dir}")

    def _generate_token_usage_chart(self, output_dir: str):
        """生成Token使用统计图表"""

        prompt_tokens = []
        completion_tokens = []
        timestamps = []

        for entry in self.data:
            token_usage = entry.get("token_usage", {})
            if token_usage:
                prompt_tokens.append(token_usage.get("prompt_tokens", 0))
                completion_tokens.append(token_usage.get("completion_tokens", 0))
                timestamps.append(entry.get("timestamp", ""))

        if not timestamps:
            return

        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 子图1: Token使用趋势
        x = range(len(timestamps))
        axes[0].plot(x, prompt_tokens, "b-", label="Prompt Tokens", linewidth=2)
        axes[0].plot(x, completion_tokens, "r-", label="Completion Tokens", linewidth=2)
        axes[0].fill_between(x, 0, prompt_tokens, alpha=0.3, color="blue")
        axes[0].fill_between(x, 0, completion_tokens, alpha=0.3, color="red")

        axes[0].set_xlabel("请求序号")
        axes[0].set_ylabel("Token数量")
        axes[0].set_title("Token使用趋势分析")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        # 子图2: 累计Token使用
        cumulative_prompt = np.cumsum(prompt_tokens)
        cumulative_completion = np.cumsum(completion_tokens)
        cumulative_total = cumulative_prompt + cumulative_completion

        axes[1].plot(x, cumulative_total, "g-", label="总Token", linewidth=3)
        axes[1].plot(x, cumulative_prompt, "b--", label="Prompt Token", linewidth=2)
        axes[1].plot(
            x, cumulative_completion, "r--", label="Completion Token", linewidth=2
        )

        axes[1].set_xlabel("请求序号")
        axes[1].set_ylabel("累计Token数量")
        axes[1].set_title("累计Token使用统计")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(output_dir, "token_usage_analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Token使用图表已生成: {output_file}")

    def _generate_cost_breakdown_chart(self, output_dir: str):
        """生成成本分解图表"""

        model_costs = {}

        for entry in self.data:
            token_usage = entry.get("token_usage", {})
            model = token_usage.get("model", "unknown")
            cost = token_usage.get("cost_usd", 0.0)

            if model not in model_costs:
                model_costs[model] = 0.0
            model_costs[model] += cost

        if not model_costs:
            return

        # 准备饼图数据
        models = list(model_costs.keys())
        costs = list(model_costs.values())
        total_cost = sum(costs)

        # 按成本排序
        sorted_indices = np.argsort(costs)[::-1]
        models = [models[i] for i in sorted_indices]
        costs = [costs[i] for i in sorted_indices]

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        # 子图1: 饼图
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        wedges, texts, autotexts = ax1.pie(
            costs,
            labels=models,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.3),
        )

        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        ax1.set_title(f"模型成本分解 (总成本: ${total_cost:.6f})")

        # 子图2: 条形图
        y_pos = np.arange(len(models))
        bars = ax2.barh(y_pos, costs, color=colors, edgecolor="black")

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(models)
        ax2.invert_yaxis()  # 成本最高的在顶部
        ax2.set_xlabel("成本 (USD)")
        ax2.set_title("各模型成本对比")

        # 在条形上添加数值标签
        for bar, cost in zip(bars, costs):
            width = bar.get_width()
            ax2.text(
                width + max(costs) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"${cost:.6f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        output_file = os.path.join(output_dir, "cost_breakdown_analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"成本分解图表已生成: {output_file}")

    def _generate_response_time_chart(self, output_dir: str):
        """生成响应时间分析图表"""

        response_times = []
        operations = []

        for entry in self.data:
            rt = entry.get("response_time_ms", 0)
            if rt > 0:
                response_times.append(rt)
                operations.append(entry.get("operation", "unknown"))

        if not response_times:
            return

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 子图1: 响应时间分布直方图
        n_bins = min(20, len(response_times))
        ax1.hist(
            response_times, bins=n_bins, color="skyblue", edgecolor="black", alpha=0.7
        )
        ax1.axvline(
            np.mean(response_times),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"平均: {np.mean(response_times):.1f}ms",
        )
        ax1.axvline(
            np.percentile(response_times, 95),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"P95: {np.percentile(response_times, 95):.1f}ms",
        )

        ax1.set_xlabel("响应时间 (ms)")
        ax1.set_ylabel("请求数量")
        ax1.set_title("响应时间分布")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 箱线图分析
        operation_types = list(set(operations))
        operation_data = []
        operation_labels = []

        for op in operation_types[:5]:  # 最多显示5种操作类型
            op_times = [rt for rt, o in zip(response_times, operations) if o == op]
            if op_times:
                operation_data.append(op_times)
                operation_labels.append(op)

        if operation_data:
            box = ax2.boxplot(
                operation_data, labels=operation_labels, patch_artist=True
            )

            # 设置箱线图颜色
            colors = ["lightblue", "lightgreen", "lightcoral", "lightsalmon", "plum"]
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)

            ax2.set_ylabel("响应时间 (ms)")
            ax2.set_title("不同操作的响应时间对比")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(output_dir, "response_time_analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"响应时间图表已生成: {output_file}")

    def _generate_daily_summary_chart(self, output_dir: str):
        """生成每日汇总图表"""

        daily_stats = {}

        for entry in self.data:
            timestamp = entry.get("timestamp", "")
            if timestamp:
                # 提取日期部分
                try:
                    date_str = timestamp.split("T")[0]
                except:
                    continue

                token_usage = entry.get("token_usage", {})
                total_tokens = token_usage.get("total_tokens", 0)
                cost_usd = token_usage.get("cost_usd", 0.0)
                response_time = entry.get("response_time_ms", 0)

                if date_str not in daily_stats:
                    daily_stats[date_str] = {
                        "total_requests": 0,
                        "total_tokens": 0,
                        "total_cost": 0.0,
                        "total_response_time": 0.0,
                    }

                stats = daily_stats[date_str]
                stats["total_requests"] += 1
                stats["total_tokens"] += total_tokens
                stats["total_cost"] += cost_usd
                stats["total_response_time"] += response_time

        if not daily_stats:
            return

        # 准备数据
        dates = sorted(daily_stats.keys())
        requests = [daily_stats[d]["total_requests"] for d in dates]
        tokens = [daily_stats[d]["total_tokens"] for d in dates]
        costs = [daily_stats[d]["total_cost"] for d in dates]
        avg_response_times = [
            daily_stats[d]["total_response_time"] / daily_stats[d]["total_requests"]
            if daily_stats[d]["total_requests"] > 0
            else 0
            for d in dates
        ]

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1: 每日请求量
        axes[0, 0].bar(dates, requests, color="steelblue", edgecolor="black")
        axes[0, 0].set_xlabel("日期")
        axes[0, 0].set_ylabel("请求数量")
        axes[0, 0].set_title("每日请求量统计")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 添加数值标签
        for i, (date, req) in enumerate(zip(dates, requests)):
            axes[0, 0].text(
                i,
                req + max(requests) * 0.02,
                str(req),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 子图2: 每日Token使用量
        axes[0, 1].plot(
            dates, tokens, "o-", color="darkorange", linewidth=2, markersize=8
        )
        axes[0, 1].fill_between(dates, 0, tokens, alpha=0.3, color="darkorange")
        axes[0, 1].set_xlabel("日期")
        axes[0, 1].set_ylabel("Token数量")
        axes[0, 1].set_title("每日Token使用量")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 子图3: 每日成本
        axes[1, 0].bar(dates, costs, color="forestgreen", edgecolor="black")
        axes[1, 0].set_xlabel("日期")
        axes[1, 0].set_ylabel("成本 (USD)")
        axes[1, 0].set_title("每日成本统计")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # 添加成本数值标签
        for i, (date, cost) in enumerate(zip(dates, costs)):
            axes[1, 0].text(
                i,
                cost + max(costs) * 0.02,
                f"${cost:.6f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 子图4: 平均响应时间
        axes[1, 1].plot(
            dates, avg_response_times, "s-", color="crimson", linewidth=2, markersize=8
        )
        axes[1, 1].set_xlabel("日期")
        axes[1, 1].set_ylabel("平均响应时间 (ms)")
        axes[1, 1].set_title("每日平均响应时间")
        axes[1, 1].tick_params(axis="x", rotation=45)

        # 添加响应时间数值标签
        for i, (date, rt) in enumerate(zip(dates, avg_response_times)):
            axes[1, 1].text(
                i,
                rt + max(avg_response_times) * 0.02,
                f"{rt:.1f}ms",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        output_file = os.path.join(output_dir, "daily_summary_analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"每日汇总图表已生成: {output_file}")

    def generate_html_report(self, output_file: str = "cost_dashboard.html"):
        """生成HTML格式的报告"""

        if not self.data:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>成本监控面板 - 无数据</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .warning { background-color: #fff3cd; border: 1px solid #ffc107; padding: 20px; border-radius: 5px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>成本监控面板</h1>
                    <div class="warning">
                        <h2>⚠️ 无可用数据</h2>
                        <p>未找到审计日志数据，请确保Agent已处理过请求。</p>
                    </div>
                </div>
            </body>
            </html>
            """
        else:
            # 计算统计数据
            total_requests = len(self.data)
            total_tokens = sum(
                entry.get("token_usage", {}).get("total_tokens", 0)
                for entry in self.data
            )
            total_cost = sum(
                entry.get("token_usage", {}).get("cost_usd", 0.0) for entry in self.data
            )
            avg_response_time = (
                sum(entry.get("response_time_ms", 0) for entry in self.data)
                / total_requests
                if total_requests > 0
                else 0
            )

            # 按模型统计成本
            model_costs = {}
            for entry in self.data:
                model = entry.get("token_usage", {}).get("model", "unknown")
                cost = entry.get("token_usage", {}).get("cost_usd", 0.0)
                model_costs[model] = model_costs.get(model, 0.0) + cost

            # 生成HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>成本监控面板 - {datetime.now().strftime("%Y-%m-%d")}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                    .stat-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
                    .stat-card h3 {{ margin-top: 0; color: #333; }}
                    .stat-number {{ font-size: 28px; font-weight: bold; color: #007bff; }}
                    .chart-placeholder {{ background-color: #e9ecef; padding: 40px; text-align: center; border-radius: 8px; margin-bottom: 20px; color: #6c757d; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
                    th {{ background-color: #f8f9fa; }}
                    .timestamp {{ font-size: 12px; color: #6c757d; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🎯 生产级Agent成本监控面板</h1>
                        <p>报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h3>总请求数</h3>
                            <div class="stat-number">{total_requests}</div>
                            <p>所有处理的请求数量</p>
                        </div>
                        
                        <div class="stat-card">
                            <h3>总Token使用量</h3>
                            <div class="stat-number">{total_tokens:,}</div>
                            <p>所有请求的Token总和</p>
                        </div>
                        
                        <div class="stat-card">
                            <h3>总成本</h3>
                            <div class="stat-number">${total_cost:.6f}</div>
                            <p>所有请求的成本总和</p>
                        </div>
                        
                        <div class="stat-card">
                            <h3>平均响应时间</h3>
                            <div class="stat-number">{avg_response_time:.1f}ms</div>
                            <p>请求的平均处理时间</p>
                        </div>
                    </div>
                    
                    <h2>📊 图表区域</h2>
                    <p>运行 <code>cost_dashboard.py</code> 脚本生成PNG图表：</p>
                    <ul>
                        <li><strong>token_usage_analysis.png</strong> - Token使用趋势分析</li>
                        <li><strong>cost_breakdown_analysis.png</strong> - 成本分解分析</li>
                        <li><strong>response_time_analysis.png</strong> - 响应时间分析</li>
                        <li><strong>daily_summary_analysis.png</strong> - 每日汇总分析</li>
                    </ul>
                    
                    <div class="chart-placeholder">
                        <h3>📈 图表预览</h3>
                        <p>请运行脚本生成可视化图表</p>
                        <p><code>python cost_dashboard.py</code></p>
                    </div>
                    
                    <h2>🔍 模型成本分解</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>模型</th>
                                <th>成本 (USD)</th>
                                <th>占比</th>
                            </tr>
                        </thead>
                        <tbody>
            """

            # 添加模型成本行
            for model, cost in sorted(
                model_costs.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (cost / total_cost * 100) if total_cost > 0 else 0
                html_content += f"""
                            <tr>
                                <td><strong>{model}</strong></td>
                                <td>${cost:.6f}</td>
                                <td>{percentage:.1f}%</td>
                            </tr>
                """

            html_content += """
                        </tbody>
                    </table>
                    
                    <h2>📋 最近请求记录</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>时间</th>
                                <th>操作</th>
                                <th>Token使用</th>
                                <th>响应时间</th>
                                <th>状态</th>
                            </tr>
                        </thead>
                        <tbody>
            """

            # 添加最近10条记录
            recent_entries = self.data[-10:] if len(self.data) > 10 else self.data
            for entry in recent_entries:
                timestamp = entry.get("timestamp", "")
                operation = entry.get("operation", "")
                token_usage = entry.get("token_usage", {})
                total_tokens = token_usage.get("total_tokens", 0)
                response_time = entry.get("response_time_ms", 0)
                success = entry.get("error") is None

                # 格式化时间
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    time_str = dt.strftime("%H:%M:%S")
                    date_str = dt.strftime("%m/%d")
                except:
                    time_str = timestamp[:19]
                    date_str = timestamp[:10]

                status_text = "✅ 成功" if success else "❌ 失败"
                status_color = "green" if success else "red"

                html_content += f"""
                            <tr>
                                <td><span class="timestamp">{date_str} {time_str}</span></td>
                                <td>{operation}</td>
                                <td>{total_tokens} tokens</td>
                                <td>{response_time:.1f}ms</td>
                                <td style="color:{status_color}">{status_text}</td>
                            </tr>
                """

            html_content += (
                """
                        </tbody>
                    </table>
                    
                    <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; text-align: center; color: #6c757d; font-size: 12px;">
                        <p>报告生成时间: """
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + """</p>
                        <p>生产级Agent成本监控面板 v1.0</p>
                    </div>
                </div>
            </body>
            </html>
            """
            )

        # 写入HTML文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"HTML报告已生成: {output_file}")
        return output_file


def main():
    """主函数"""
    print("=" * 60)
    print("生产级Agent成本监控面板")
    print("=" * 60)

    # 创建仪表板实例
    dashboard = CostDashboard()

    # 生成图表
    output_dir = "cost_dashboard_output"
    dashboard.generate_dashboard(output_dir)

    # 生成HTML报告
    html_file = os.path.join(output_dir, "cost_dashboard.html")
    dashboard.generate_html_report(html_file)

    print("\n" + "=" * 60)
    print("仪表板生成完成！")
    print(f"图表文件: {output_dir}/")
    print(f"HTML报告: {html_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
