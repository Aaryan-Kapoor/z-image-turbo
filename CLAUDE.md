# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Z-Image-Turbo is a web interface and MCP server for the Tongyi-MAI Z-Image-Turbo text-to-image generation model. The project has two main deployment modes:

1. **Web Application**: React frontend + FastAPI backend for interactive image generation
2. **MCP Server**: Model Context Protocol server for AI assistant integration (Claude Desktop, LM Studio)

## Essential Commands

### Backend Setup & Running

```bash
# Create and activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Run FastAPI web server (port 8000)
cd backend
python main.py

# Run MCP server (stdio mode for Claude Desktop/LM Studio)
cd backend
python mcp_server.py --transport stdio

# Run MCP server with eager loading (loads model at startup)
cd backend
python mcp_server.py --transport stdio --eager-load

# Run MCP server in HTTP mode (port 8001)
cd backend
python mcp_server.py --transport streamable-http --port 8001

# Test MCP server with inspector
npx @modelcontextprotocol/inspector python backend/mcp_server.py --transport stdio
```

### Frontend Setup & Running

```bash
# Install dependencies
cd frontend
npm install

# Start dev server (port 5173)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Testing

The project uses the MCP Inspector for testing the MCP server:
```bash
npx @modelcontextprotocol/inspector python backend/mcp_server.py --transport stdio
```

## Architecture

### Two-Server Architecture

The project maintains two independent server implementations:

1. **`backend/main.py`** - FastAPI web server
   - REST API at `http://localhost:8000`
   - Endpoints: `/generate`, `/settings`, `/settings/model-path`, `/health`
   - Used by the React frontend for interactive use
   - CORS enabled for browser access

2. **`backend/mcp_server.py`** - MCP server
   - Runs on stdio (local) or HTTP/SSE transport (remote)
   - Exposes tools: `generate_image`, `get_model_info`, `update_model_config`
   - Used by AI assistants (Claude Desktop, LM Studio)
   - Includes model lifecycle management (lazy/eager loading, TTL)

**Important**: These are separate servers. Changes to one don't affect the other unless they share the common `config.json` file.

### Model Pipeline Management

Both servers use the Z-Image-Turbo diffusion model via Hugging Face's `diffusers` library:

- **Model class**: `ZImagePipeline` (requires diffusers installed from git source)
- **Model ID**: `Tongyi-MAI/Z-Image-Turbo` (6B parameters)
- **GPU Requirements**: ~8GB VRAM (can use CPU offload for less)
- **Lazy loading** (default): Model loads on first image generation request
- **Eager loading** (`--eager-load` flag): Model loads at server startup

The MCP server (`mcp_server.py`) has advanced lifecycle features:
- Thread-safe model loading/unloading with locks
- Model TTL (auto-unload after inactivity to free VRAM)
- Concurrent request limiting via semaphore
- Configurable via `backend/mcp_config.json`

The FastAPI server (`main.py`) uses simpler on-demand loading:
- Model loads on first request and stays loaded
- No TTL or automatic unloading

### Configuration System

**`config.json`** (project root) - Shared model configuration:
```json
{
  "cache_dir": "E:\\ImageModels",  // Where to store/load model weights
  "model_id": "Tongyi-MAI/Z-Image-Turbo",
  "cpu_offload": false  // Enable for GPUs with <8GB VRAM
}
```

**`backend/mcp_config.json`** - MCP server behavior:
```json
{
  "transport": "stdio",  // "stdio" or "streamable-http"
  "eager_load": false,   // Load model at startup vs on-demand
  "model_ttl_minutes": 0,  // Auto-unload after N minutes idle (0 = never)
  "max_concurrent_requests": 1,  // Concurrent generation limit
  "log_level": "INFO"
}
```

### Frontend Architecture

React + Vite single-page application:
- **Entry point**: `frontend/src/main.jsx`
- **Main component**: `frontend/src/App.jsx` (monolithic component, all UI in one file)
- **Styling**: CSS custom properties in `App.css` (dark glassmorphism theme)
- **API communication**: Direct `fetch()` calls to `http://localhost:8000`
- **State management**: React `useState` (no external state library)

Key UI features:
- Left sidebar: Model parameters (steps, guidance scale, dimensions, seed)
- Main canvas: Image display with hover overlay (download, fullscreen)
- Bottom bar: Prompt textarea with generate button
- Settings modal: Model cache directory configuration

## Key Development Patterns

### Model Loading Pattern

Both servers follow this pattern but with different complexity levels:

```python
# Global pipeline state
pipe = None

def get_pipeline():
    global pipe
    if pipe is None:
        # Load model from HuggingFace or cache
        pipe = ZImagePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # Use bfloat16 on GPU
            cache_dir=cache_dir
        )
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
    return pipe
```

In `mcp_server.py`, this is wrapped with:
- Thread locks for safety
- TTL-based unloading
- Semaphore for request limiting

### Image Generation Pattern

```python
generator = None
if seed != -1:
    generator = torch.Generator(device).manual_seed(seed)

image = pipeline(
    prompt=prompt,
    height=height,
    width=width,
    num_inference_steps=steps,
    guidance_scale=guidance_scale,
    generator=generator
).images[0]

# Convert PIL Image to base64 for JSON response
buffered = BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
```

### Validation Rules

Dimensions must be:
- Divisible by 16 (diffusion model requirement)
- Between 64-2048 pixels (enforced in MCP server)
- Frontend presets use standard ratios: 1:1, 3:4, 4:3, 16:9

### MCP Server Transport Modes

**Stdio Mode** (default, local only):
- Communication via stdin/stdout
- Used by Claude Desktop, LM Studio, MCP Inspector
- Configured in client apps with Python path + args

**HTTP/SSE Mode** (network accessible):
- Communication via HTTP with Server-Sent Events
- Used for web clients or remote access
- Requires host and port configuration

## Important Notes

### Dependencies Installation

The project requires `diffusers` from git source (not PyPI):
```bash
pip install git+https://github.com/huggingface/diffusers.git
```

This is because `ZImagePipeline` may not be in the stable PyPI release yet.

### Windows Path Handling

When configuring MCP clients (Claude Desktop, LM Studio), use:
- **Double backslashes** in JSON: `C:\\path\\to\\file`
- **Absolute paths** to venv Python: `C:\\path\\to\\venv\\Scripts\\python.exe`
- Never use system Python - dependencies are in venv

### Model Loading Timeouts

First model load takes 30-60 seconds:
- MCP clients need `timeout: 300000` (5 minutes in milliseconds)
- Use `--eager-load` flag to load at startup instead of on first request
- Subsequent generations are fast (~2-20 seconds depending on resolution)

### VRAM Management

Model uses ~8GB VRAM when loaded:
- Set `cpu_offload: true` in `config.json` for GPUs with less VRAM
- Use `model_ttl_minutes > 0` in MCP config to auto-unload when idle
- Only one generation at a time (`max_concurrent_requests: 1`) prevents OOM

### CORS and Localhost

FastAPI server has CORS enabled with `allow_origins=["*"]` for development. The frontend expects the backend on `localhost:8000`.

## File Structure Reference

```
z-image-turbo/
├── backend/
│   ├── main.py              # FastAPI web server (port 8000)
│   ├── mcp_server.py        # MCP server with lifecycle management
│   ├── mcp_config.json      # MCP server configuration
│   ├── requirements.txt     # Python dependencies
│   ├── run_mcp.sh           # Shell script to start MCP server
│   └── run_mcp.ps1          # PowerShell script to start MCP server
├── frontend/
│   ├── src/
│   │   ├── main.jsx         # React entry point
│   │   ├── App.jsx          # Main UI component (monolithic)
│   │   ├── App.css          # Styling with CSS custom properties
│   │   └── index.css        # Global styles
│   ├── vite.config.js       # Vite bundler configuration
│   └── package.json         # Node dependencies
├── config.json              # Shared model configuration
├── README.md                # User-facing documentation
├── MCP_README.md            # Detailed MCP deployment guide
└── venv/                    # Python virtual environment
```
