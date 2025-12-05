#!/bin/bash

# MCP Server Startup Script for Z-Image-Turbo
# This script starts the MCP server with configurable transport modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help message
show_help() {
    cat << EOF
Z-Image-Turbo MCP Server Launcher

Usage: $0 [OPTIONS]

Options:
    --stdio             Run with stdio transport (for local clients like Claude Desktop)
    --http              Run with HTTP/SSE transport (for web clients)
    --host HOST         Set the host address for HTTP mode (default: 0.0.0.0)
    --port PORT         Set the port for HTTP mode (default: 8001)
    -h, --help          Show this help message

Examples:
    # Run with stdio (default, for Claude Desktop)
    $0 --stdio

    # Run with HTTP on default port 8001
    $0 --http

    # Run with HTTP on custom port
    $0 --http --port 8080

    # Run with HTTP on specific host
    $0 --http --host 127.0.0.1 --port 9000

EOF
}

# Default values
TRANSPORT=""
HOST=""
PORT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stdio)
            TRANSPORT="stdio"
            shift
            ;;
        --http)
            TRANSPORT="streamable-http"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Change to backend directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_info "Starting Z-Image-Turbo MCP Server..."

# Build command
CMD="python mcp_server.py"

if [ -n "$TRANSPORT" ]; then
    CMD="$CMD --transport $TRANSPORT"
fi

if [ -n "$HOST" ]; then
    CMD="$CMD --host $HOST"
fi

if [ -n "$PORT" ]; then
    CMD="$CMD --port $PORT"
fi

# Print configuration
print_info "Configuration:"
if [ -n "$TRANSPORT" ]; then
    echo "  Transport: $TRANSPORT"
else
    echo "  Transport: (from mcp_config.json)"
fi

if [ "$TRANSPORT" = "streamable-http" ]; then
    if [ -n "$HOST" ]; then
        echo "  Host: $HOST"
    else
        echo "  Host: (from mcp_config.json, default: 0.0.0.0)"
    fi

    if [ -n "$PORT" ]; then
        echo "  Port: $PORT"
    else
        echo "  Port: (from mcp_config.json, default: 8001)"
    fi
fi

echo ""
print_success "Launching MCP server..."
echo ""

# Run the server
exec $CMD
