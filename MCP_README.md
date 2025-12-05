# Z-Image-Turbo MCP Server - Complete Deployment Guide

A production-ready MCP (Model Context Protocol) server for AI image generation using the Z-Image-Turbo model. This guide covers everything you need to deploy and integrate with Claude Desktop, LM Studio, and other MCP clients.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [GGUF Quantized Models](#gguf-quantized-models)
5. [Server Configuration](#server-configuration)
6. [Client Configuration](#client-configuration)
   - [Claude Desktop](#claude-desktop-setup)
   - [LM Studio](#lm-studio-setup)
   - [Other MCP Clients](#other-mcp-clients)
7. [Available Tools](#available-tools)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)
10. [Security Considerations](#security-considerations)
11. [Development](#development)

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

| Requirement | Minimum (Full Model) | Minimum (GGUF Q4) | Recommended |
|-------------|----------------------|-------------------|-------------|
| **GPU VRAM** | 8 GB | 6 GB | 12+ GB |
| **System RAM** | 16 GB | 16 GB | 32 GB |
| **Disk Space** | 15 GB | 5 GB | 20 GB |
| **Python** | 3.10+ | 3.10+ | 3.10+ |
| **CUDA** | 11.8+ | 11.8+ | 12.0+ |

> **Note:** GGUF quantized models significantly reduce VRAM and disk requirements. See [GGUF Quantized Models](#gguf-quantized-models) for details.

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
  "gguf_filename": null,
  "cpu_offload": false
}
```

| Option | Description |
|--------|-------------|
| `cache_dir` | Where to store the downloaded model (~12GB for full, ~4-8GB for GGUF) |
| `model_id` | HuggingFace model identifier (use `AaryanK/Z-Image-Turbo-GGUF` for GGUF) |
| `gguf_filename` | GGUF file to use (e.g., `z_image_turbo-Q4_K_M.gguf`). Set to `null` for full model |
| `cpu_offload` | Set `true` if you have limited GPU memory (auto-enabled for GGUF) |

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

## GGUF Quantized Models

GGUF (GPT-Generated Unified Format) allows you to run quantized versions of the Z-Image-Turbo model with significantly reduced VRAM and disk requirements.

### Available GGUF Models

Models are available from `AaryanK/Z-Image-Turbo-GGUF`:

| Model | Size | Bits | Quality | Use Case |
|-------|------|------|---------|----------|
| `z_image_turbo-Q3_K_S.gguf` | 3.79 GB | 3-bit | Good | Lowest VRAM, fastest |
| `z_image_turbo-Q4_K_M.gguf` | 4.98 GB | 4-bit | Great | **Recommended balance** |
| `z_image_turbo-Q5_K_M.gguf` | 5.52 GB | 5-bit | Very Good | Higher quality |
| `z_image_turbo-Q6_K.gguf` | 5.91 GB | 6-bit | Excellent | Near lossless |
| `z_image_turbo-Q8_0.gguf` | 7.22 GB | 8-bit | Near Perfect | Very high quality |
| `z_image_turbo-bf16.gguf` | 12.3 GB | 16-bit | Maximum | Full precision |

### Configuring GGUF via MCP

Use the `update_model_config` tool to switch to a GGUF model:

```json
{
  "model_id": "AaryanK/Z-Image-Turbo-GGUF",
  "gguf_filename": "z_image_turbo-Q4_K_M.gguf"
}
```

Or manually edit `config.json`:

```json
{
  "cache_dir": "./models",
  "model_id": "AaryanK/Z-Image-Turbo-GGUF",
  "gguf_filename": "z_image_turbo-Q4_K_M.gguf",
  "cpu_offload": false
}
```

### GGUF Technical Details

- **How it works:** GGUF contains only the quantized transformer weights. The text encoder and other components are loaded from the original `Tongyi-MAI/Z-Image-Turbo` model automatically.
- **CPU Offload:** Automatically enabled for GGUF models (recommended by diffusers library).
- **First load:** Downloads both the GGUF file and the original model components (~additional 2-3GB for text encoder).
- **Requires:** Latest diffusers from git (`pip install git+https://github.com/huggingface/diffusers.git`).

### Switching Back to Full Model

To return to the full-precision model:

```json
{
  "model_id": "Tongyi-MAI/Z-Image-Turbo",
  "gguf_filename": null
}
```

### GGUF Workflow via MCP Tools

AI assistants can manage GGUF models using these MCP tools:

1. **List available models**: `list_available_gguf_models` - See all GGUF variants with sizes
2. **Check downloads**: `list_downloaded_gguf_models` - See what's already cached locally
3. **Download a model**: `download_gguf_model` - Download a specific GGUF file
4. **Activate model**: `update_model_config` - Just set `gguf_filename` to switch
5. **Verify**: `get_model_info` - Check current GGUF status

**Example conversation:**
```
User: "Switch to a smaller model to save VRAM"
Assistant: [calls list_available_gguf_models]
Assistant: "I see Q4_K_M (4.98 GB) is recommended. Let me check if it's downloaded."
Assistant: [calls list_downloaded_gguf_models]
Assistant: "It's not downloaded yet. Downloading now..."
Assistant: [calls download_gguf_model with filename="z_image_turbo-Q4_K_M.gguf"]
Assistant: "Downloaded! Now activating..."
Assistant: [calls update_model_config with gguf_filename="z_image_turbo-Q4_K_M.gguf"]
Assistant: "Done! Now using Q4_K_M quantization (~5GB VRAM instead of ~8GB)"
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
  "model_id": "AaryanK/Z-Image-Turbo-GGUF",
  "cache_dir": "./models",
  "cpu_offload": false,
  "device": "cuda",
  "is_loaded": true,
  "cuda_available": true,
  "gguf": {
    "enabled": true,
    "filename": "z_image_turbo-Q4_K_M.gguf",
    "repo_id": "AaryanK/Z-Image-Turbo-GGUF",
    "note": "GGUF models use less VRAM with quantized weights"
  },
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_memory_total_gb": 24.0,
  "gpu_memory_allocated_gb": 5.2,
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

When setting `gguf_filename`, the `model_id` is automatically configured - you only need to specify the GGUF filename.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cache_dir` | string | Model cache directory |
| `cpu_offload` | boolean | Enable CPU offload for memory optimization |
| `gguf_filename` | string | GGUF file to use, or `"none"` for full model |

**Example - Switch to GGUF:**
```json
{
  "gguf_filename": "z_image_turbo-Q4_K_M.gguf"
}
```

**Example - Switch back to full model:**
```json
{
  "gguf_filename": "none"
}
```

### 4. `list_available_gguf_models`

List all available GGUF quantized models from HuggingFace.

**Returns:**
```json
{
  "models": [
    {
      "filename": "z_image_turbo-Q3_K_S.gguf",
      "size_gb": 3.79,
      "quantization": "Q3_K_S (3-bit)",
      "description": "3-bit small, fastest"
    },
    {
      "filename": "z_image_turbo-Q4_K_M.gguf",
      "size_gb": 4.98,
      "quantization": "Q4_K_M (4-bit)",
      "description": "4-bit medium, recommended"
    }
  ],
  "repo_id": "AaryanK/Z-Image-Turbo-GGUF",
  "current_model": "z_image_turbo-Q4_K_M.gguf",
  "tip": "Use download_gguf_model to download, then update_model_config to switch"
}
```

### 5. `list_downloaded_gguf_models`

List GGUF models already downloaded to your cache directory.

**Returns:**
```json
{
  "downloaded_models": [
    {
      "filename": "z_image_turbo-Q4_K_M.gguf",
      "path": "/path/to/cache/z_image_turbo-Q4_K_M.gguf",
      "size_gb": 4.98,
      "quantization": "Q4_K_M"
    }
  ],
  "count": 1,
  "cache_dir": "./models",
  "current_active": "z_image_turbo-Q4_K_M.gguf",
  "tip": "Use update_model_config with gguf_filename to switch models"
}
```

### 6. `download_gguf_model`

Download a specific GGUF model from HuggingFace.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filename` | string | ✅ | GGUF filename (e.g., `z_image_turbo-Q4_K_M.gguf`) |

**Example:**
```json
{
  "filename": "z_image_turbo-Q4_K_M.gguf"
}
```

**Returns:**
```json
{
  "status": "success",
  "filename": "z_image_turbo-Q4_K_M.gguf",
  "path": "/path/to/cache/z_image_turbo-Q4_K_M.gguf",
  "size_gb": 4.98,
  "message": "Downloaded z_image_turbo-Q4_K_M.gguf (4.98 GB)",
  "next_step": "Use update_model_config with gguf_filename='z_image_turbo-Q4_K_M.gguf' to activate"
}
```

### 7. Resource: `image://examples`

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

### ❌ GGUF loading fails with "No module named 'gguf'"

**Cause:** The `gguf` package is not installed.

**Solution:**
```bash
pip install gguf
```

### ❌ GGUF loading fails with tensor/shape errors

**Cause:** Using an older version of diffusers that doesn't support GGUF for ZImage.

**Solution:**
```bash
pip install git+https://github.com/huggingface/diffusers.git
```

### ❌ GGUF model takes long to load first time

**Cause:** First load downloads both the GGUF file AND original model components (text encoder).

**Solution:** This is normal - subsequent loads will be faster as files are cached.

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

**Full Model:**
- **~8GB VRAM** required when model is loaded
- Use `model_ttl_minutes: 10` to free memory when idle
- Enable `cpu_offload` for GPUs with less than 8GB VRAM

**GGUF Models:**
- **~4-6GB VRAM** for Q4/Q5 quantizations
- **~3.5GB VRAM** for Q3 quantizations
- CPU offload automatically enabled for optimal performance
- Best choice for GPUs with 6-8GB VRAM

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
