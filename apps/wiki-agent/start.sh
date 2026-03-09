#!/bin/bash

# Wiki Agent 一键启动脚本
# 启动 Chrome + MCP Server

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Wiki Agent 启动脚本 ===${NC}"

# 检测操作系统
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    CHROME_PATH="google-chrome"
else
    echo -e "${RED}不支持的操作系统: $OSTYPE${NC}"
    exit 1
fi

echo -e "${BLUE}检测到操作系统: $OS${NC}"

# 检查 Chrome 是否已运行（带远程调试）
check_chrome() {
    if curl -s http://localhost:9222/json/version > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 启动 Chrome
start_chrome() {
    echo -e "${YELLOW}启动 Chrome (远程调试端口 9222)...${NC}"

    if [ ! -f "$CHROME_PATH" ] && [ "$OS" == "macos" ]; then
        echo -e "${RED}未找到 Chrome: $CHROME_PATH${NC}"
        echo -e "${YELLOW}请确保已安装 Google Chrome${NC}"
        exit 1
    fi

    # 在后台启动 Chrome
    if [ "$OS" == "macos" ]; then
        "$CHROME_PATH" \
            --remote-debugging-port=9222 \
            --no-first-run \
            --no-default-browser-check \
            --user-data-dir="$HOME/.wiki_agent/chrome_profile" \
            > /dev/null 2>&1 &
    else
        $CHROME_PATH \
            --remote-debugging-port=9222 \
            --no-first-run \
            --no-default-browser-check \
            --user-data-dir="$HOME/.wiki_agent/chrome_profile" \
            > /dev/null 2>&1 &
    fi

    CHROME_PID=$!
    echo -e "${GREEN}Chrome 已启动 (PID: $CHROME_PID)${NC}"

    # 等待 Chrome 启动
    echo -e "${YELLOW}等待 Chrome 启动...${NC}"
    for i in {1..10}; do
        if check_chrome; then
            echo -e "${GREEN}Chrome 远程调试已就绪${NC}"
            return 0
        fi
        sleep 1
    done

    echo -e "${RED}Chrome 启动超时${NC}"
    return 1
}

# 启动 MCP Server
start_mcp_server() {
    echo -e "${YELLOW}启动 MCP Server...${NC}"

    # 检查 uv 是否安装
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}未找到 uv，请先安装 uv: https://docs.astral.sh/uv/${NC}"
        exit 1
    fi

    # 检查并安装依赖
    echo -e "${YELLOW}检查依赖...${NC}"
    uv sync --quiet

    # 启动 MCP Server (使用 uv run 确保使用正确的 Python 环境)
    uv run python src/wiki_mcp_server.py &
    MCP_PID=$!

    echo -e "${GREEN}MCP Server 已启动 (PID: $MCP_PID)${NC}"

    # 等待 MCP Server 启动
    echo -e "${YELLOW}等待 MCP Server 启动...${NC}"
    for i in {1..10}; do
        if curl -s http://127.0.0.1:8001/sse > /dev/null 2>&1 || \
           curl -s http://127.0.0.1:8000/sse > /dev/null 2>&1; then
            echo -e "${GREEN}MCP Server 已就绪${NC}"
            return 0
        fi
        sleep 1
    done

    echo -e "${RED}MCP Server 启动超时${NC}"
    return 1
}

# 主流程
main() {
    # 创建工作目录
    mkdir -p "$HOME/.wiki_agent"

    # 1. 检查 Chrome
    if check_chrome; then
        echo -e "${GREEN}Chrome 已在运行 (远程调试端口 9222)${NC}"
    else
        start_chrome
    fi

    echo ""

    # 2. 检查 MCP Server
    if pgrep -f "wiki_mcp_server.py" > /dev/null; then
        echo -e "${GREEN}MCP Server 已在运行${NC}"
    else
        start_mcp_server
    fi

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Wiki Agent 启动完成!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}服务状态:${NC}"
    echo -e "  - Chrome 远程调试: http://localhost:9222"
    echo -e "  - MCP Server SSE:  http://127.0.0.1:8001/sse"
    echo ""
    echo -e "${BLUE}可用命令:${NC}"
    echo -e "  python src/cli.py search '关键词'"
    echo -e "  python src/cli.py read <page_id>"
    echo -e "  python src/cli.py --verbose read <page_id>  # 详细日志"
    echo ""
    echo -e "${YELLOW}按 Ctrl+C 停止服务${NC}"

    # 等待用户中断
    wait
}

# 清理函数
cleanup() {
    echo ""
    echo -e "${YELLOW}正在停止服务...${NC}"
    pkill -f "wiki_mcp_server.py" 2>/dev/null || true
    echo -e "${GREEN}服务已停止${NC}"
    exit 0
}

trap cleanup INT TERM

# 运行主程序
main
