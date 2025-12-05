# Z-Image-Turbo MCP Server - Complete Deployment Guide

A production-ready MCP (Model Context Protocol) server for AI image generation using the Z-Image-Turbo model. This guide covers everything you need to deploy and integrate with Claude Desktop, LM Studio, and other MCP clients.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Server Configuration](#server-configuration)
5. [Client Configuration](#client-configuration)
   - [Claude Desktop](#claude-desktop-setup)
   - [LM Studio](#lm-studio-setup)
   - [Other MCP Clients](#other-mcp-clients)
6. [Available Tools](#available-tools)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)
9. [Security Considerations](#security-considerations)
10. [Development](#development)

---

## Quick Start

```bash
# 1. Clone and setup
cd z-image-turbo
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Test the server
python backend/mcp_server.py --transport stdio
```

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **GPU VRAM** | 8 GB | 12+ GB |
| **System RAM** | 16 GB | 32 GB |
| **Disk Space** | 15 GB | 20 GB |
| **Python** | 3.10+ | 3.10+ |
| **CUDA** | 11.8+ | 12.0+ |

### Software Dependencies

- Python 3.10 or higher
- CUDA toolkit (for GPU acceleration)
- Git (for cloning diffusers from source)

---

## Installation

### Step 1: Create Virtual Environment

```bash
cd z-image-turbo
python -m venv venv

# Activate (choose your OS):
.\venv\Scripts\activate      # Windows PowerShell
source venv/bin/activate     # Linux/macOS
```

### Step 2: Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### Step 3: Configure the Model

Create or edit `config.json` in the project root:

```json
{
  "cache_dir": "./models",
  "model_id": "Tongyi-MAI/Z-Image-Turbo",
  "cpu_offload": false
}
```

| Option | Description |
|--------|-------------|
| `cache_dir` | Where to store the downloaded model (~12GB) |
| `model_id` | HuggingFace model identifier |
| `cpu_offload` | Set `true` if you have limited GPU memory |

### Step 4: Test the Installation

```bash
# Verify the server starts correctly (add --eager-load to test model loading)
python backend/mcp_server.py --transport stdio --eager-load
```

You should see:
```
Eager loading model at startup...
Loading Z-Image-Turbo model... (this may take 30-60 seconds on first run)
Model loaded on cuda (GPU memory: X.XX GB)
Model ready! Server is accepting requests.
```

---

## Server Configuration

### MCP Server Config (`backend/mcp_config.json`)

```json
{
  "transport": "stdio",
  "host": "0.0.0.0",
  "port": 8001,
  "eager_load": false,
  "model_ttl_minutes": 0,
  "max_concurrent_requests": 1,
  "log_level": "INFO"
}
```

### Configuration Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `transport` | string | `"stdio"` | `"stdio"` for local clients, `"streamable-http"` for web |
| `host` | string | `"0.0.0.0"` | Host address (HTTP mode only) |
| `port` | integer | `8001` | Port number (HTTP mode only) |
| `eager_load` | boolean | `false` | Load model at startup (use `--eager-load` to enable) |
| `model_ttl_minutes` | integer | `0` | Auto-unload after N minutes idle (0 = never) |
| `max_concurrent_requests` | integer | `1` | Max parallel requests (prevents OOM) |
| `log_level` | string | `"INFO"` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Recommended Configurations

**For Dedicated MCP Server (always ready):**
```json
{
  "transport": "stdio",
  "eager_load": true,
  "model_ttl_minutes": 0,
  "max_concurrent_requests": 1,
  "log_level": "INFO"
}
```

**For Shared GPU (save memory when idle):**
```json
{
  "transport": "stdio",
  "eager_load": false,
  "model_ttl_minutes": 10,
  "max_concurrent_requests": 1,
  "log_level": "INFO"
}
```

### Command Line Options

| Flag | Description | Example |
|------|-------------|---------|
| `--transport` | Override transport mode | `--transport stdio` |
| `--host` | Override host address | `--host 127.0.0.1` |
| `--port` | Override port number | `--port 8080` |
| `--eager-load` | Force model load at startup | `--eager-load` |
| `--lazy-load` | Force lazy loading | `--lazy-load` |

---

## Client Configuration

### Claude Desktop Setup

**Config file location:**
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

**Complete configuration:**

```json
{
    "mcpServers": {
      "z-image-turbo": {
        "command": "C:\\path\\to\\z-image-turbo\\venv\\Scripts\\python.exe",
        "args": [
          "C:\\path\\to\\z-image-turbo\\backend\\mcp_server.py",
          "--transport",
          "stdio",
          "--lazy-load"
        ],
        "env": {
          "PYTHONUNBUFFERED": "1"
        },
        "timeout": 300000
      }
    }
}
```

**⚠️ CRITICAL Settings Explained:**

| Setting | Value | Why It's Important |
|---------|-------|-------------------|
| `command` | Path to **venv** Python | Must use venv where dependencies are installed |
| `--eager-load` | CLI flag | Loads model at startup to avoid timeouts |
| `timeout` | `300000` | 5 minutes in ms - allows time for model loading |
| `PYTHONUNBUFFERED` | `"1"` | Ensures logs appear in real-time |

**Example for Windows:**
```json
{
  "mcpServers": {
    "z-image-turbo": {
      "command": "C:\\path\\to\\z-image-turbo\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\path\\to\\z-image-turbo\\backend\\mcp_server.py",
        "--transport",
        "stdio",
        "--lazy-load"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      },
      "timeout": 300000
    }
  }
}
```

**Example for macOS/Linux:**
```json
{
  "mcpServers": {
    "z-image-turbo": {
      "command": "/home/user/z-image-turbo/venv/bin/python",
      "args": [
        "/home/user/z-image-turbo/backend/mcp_server.py",
        "--transport",
        "stdio",
        "--lazy-load"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      },
      "timeout": 300000
    }
  }
}
```

### LM Studio Setup

**Complete configuration:**

```json
{
  "mcpServers": {
    "z-image-turbo": {
      "command": "C:\\path\\to\\z-image-turbo\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\path\\to\\z-image-turbo\\backend\\mcp_server.py",
        "--transport",
        "stdio",
        "--lazy-load"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      },
      "timeout": 300000
    }
  }
}
```

**Important for LM Studio:**
- Use **forward slashes** `/` even on Windows
- Point to the **venv Python executable**
- The `timeout` setting is critical

### Other MCP Clients

For any MCP client that supports stdio transport:

1. **Command**: Path to Python in your virtual environment
2. **Args**: `["path/to/mcp_server.py", "--transport", "stdio", "--eager-load"]`
3. **Timeout**: At least 300000ms (5 minutes)

---

## Available Tools

### 1. `generate_image`

Generate an image from a text prompt.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | ✅ | - | Text description of the image |
| `width` | integer | ❌ | 512 | Width in pixels (64-2048, divisible by 16) |
| `height` | integer | ❌ | 512 | Height in pixels (64-2048, divisible by 16) |
| `num_inference_steps` | integer | ❌ | 5 | Denoising steps (1-50) |
| `guidance_scale` | float | ❌ | 0.0 | CFG scale (0.0-20.0) |
| `seed` | integer | ❌ | random | For reproducible results |

**Example Request:**
```json
{
  "prompt": "A majestic dragon perched on a castle tower at sunset",
  "width": 1024,
  "height": 768,
  "num_inference_steps": 8,
  "guidance_scale": 0.0,
  "seed": 42
}
```

### 2. `get_model_info`

Get information about the loaded model, GPU, and configuration.

**Returns:**
```json
{
  "model_id": "Tongyi-MAI/Z-Image-Turbo",
  "device": "cuda",
  "is_loaded": true,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_memory_total_gb": 24.0,
  "gpu_memory_allocated_gb": 8.45,
  "default_settings": {
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 8,
    "guidance_scale": 0.0
  },
  "limits": {
    "min_dimension": 64,
    "max_dimension": 2048,
    "dimension_divisible_by": 16,
    "max_inference_steps": 50
  }
}
```

### 3. `update_model_config`

Update model configuration (requires reload).

**Parameters:**
- `cache_dir` (string, optional): Model cache directory
- `cpu_offload` (boolean, optional): Enable CPU offload

### 4. Resource: `image://examples`

Access curated example prompts and tips.

---

## Troubleshooting

### ❌ "No module named 'mcp'"

**Cause:** Claude Desktop/LM Studio is using system Python instead of venv.

**Solution:** Update your config to use the venv Python:
```json
"command": "C:/path/to/z-image-turbo/venv/Scripts/python.exe"
```

### ❌ Server disconnects immediately

**Cause:** Dependencies not installed or wrong Python path.

**Solution:**
1. Verify venv is activated: `.\venv\Scripts\activate`
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Test manually: `python backend/mcp_server.py --transport stdio`

### ❌ Timeout errors

**Cause:** Model loading takes 30-60+ seconds on first run.

**Solutions:**
1. Add `--eager-load` to args (loads at startup)
2. Set `"timeout": 300000` in client config
3. Ensure `eager_load: true` in `mcp_config.json`

### ❌ GPU out of memory

**Cause:** Not enough VRAM for the model (~8GB required).

**Solutions:**
1. Set `"cpu_offload": true` in `config.json`
2. Close other GPU applications
3. Use smaller image dimensions

### ❌ "Model not loaded" error

**Cause:** Model failed to load or was unloaded by TTL.

**Solutions:**
1. Check logs for loading errors
2. Verify disk space for model cache
3. Set `model_ttl_minutes: 0` to prevent auto-unload

### Viewing Logs

**Claude Desktop logs (Windows):**
```
%APPDATA%\Claude\logs\
```

**Server logs:** Written to stderr, visible in client logs.

---

## Performance Optimization

### Recommended Settings for Best Performance

| Setting | Value | Effect |
|---------|-------|--------|
| `eager_load` | `true` | No delay on first request |
| `model_ttl_minutes` | `0` | Model always ready |
| `max_concurrent_requests` | `1` | Prevents GPU OOM |
| `num_inference_steps` | `8` | Best speed/quality balance |

### Memory Management

- **~8GB VRAM** required when model is loaded
- Use `model_ttl_minutes: 10` to free memory when idle
- Enable `cpu_offload` for GPUs with less than 8GB VRAM

### Generation Speed by Dimension

| Resolution | Typical Time |
|------------|--------------|
| 512×512 | ~2 seconds |
| 768×768 | ~4 seconds |
| 1024×1024 | ~6 seconds |
| 2048×2048 | ~20 seconds |

---

## Security Considerations

- **Stdio Mode**: Only accessible locally (recommended)
- **HTTP Mode**: Only expose on trusted networks
- **No Content Filtering**: Prompts are processed as-is
- **Resource Usage**: Large AI model loads ~8GB VRAM

---

## Development

### Project Structure

```
z-image-turbo/
├── backend/
│   ├── mcp_server.py        # MCP server implementation
│   ├── mcp_config.json      # Server configuration
│   ├── run_mcp.sh           # Shell startup script
│   ├── run_mcp.ps1          # PowerShell startup script
│   ├── main.py              # FastAPI server (alternative)
│   └── requirements.txt     # Python dependencies
├── config.json              # Model configuration
├── MCP_README.md            # This file
└── README.md                # Main project README
```

### Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector python backend/mcp_server.py --transport stdio
```

### Adding New Tools

```python
@mcp.tool()
async def my_new_tool(param: str) -> str:
    """
    Tool description (shown to AI clients).

    Args:
        param: Parameter description

    Returns:
        Return value description
    """
    return f"Result: {param}"
```

---

## Support

- **Documentation**: https://modelcontextprotocol.io
- **Issues**: Open an issue on the GitHub repository
- **MCP Community**: https://discord.gg/anthropic

---

## Quick Reference Card

### Minimum Viable Config (Claude Desktop)

```json
{
  "mcpServers": {
    "z-image-turbo": {
      "command": "PATH/TO/venv/Scripts/python.exe",
      "args": ["PATH/TO/backend/mcp_server.py", "--transport", "stdio", "--eager-load"],
      "timeout": 300000
    }
  }
}
```

### Server Config Checklist

- [ ] `eager_load: true` (avoids timeouts)
- [ ] `model_ttl_minutes: 0` (keeps model loaded)
- [ ] `max_concurrent_requests: 1` (prevents OOM)

### Client Config Checklist

- [ ] Using **venv Python** path (not system Python)
- [ ] `--eager-load` in args
- [ ] `timeout: 300000` (5 minutes)
- [ ] Forward slashes `/` in paths (even on Windows)
