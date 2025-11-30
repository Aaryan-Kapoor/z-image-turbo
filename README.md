# Z-Image-Turbo

> A professional web interface for the Tongyi-MAI Z-Image-Turbo model — lightning-fast text-to-image generation with 6B parameters.

![Z-Image-Turbo Interface](assets/projectScreenshot.png)

![Z-Image-Turbo](https://img.shields.io/badge/Model-Z--Image--Turbo-blue) ![License](https://img.shields.io/badge/License-Apache%202.0-green)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- 8GB+ VRAM recommended (or use CPU offload)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aaryan-Kapoor/z-image-turbo.git
   cd z-image-turbo
   ```

2. **Backend Setup**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

**Terminal 1 - Start Backend:**
```bash
.\venv\Scripts\activate  # or source venv/bin/activate on Linux/Mac
cd backend
python main.py
```

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm run dev
```

Open **`http://localhost:5173`** in your browser and start generating!

---

## 🔌 MCP Server (Model Context Protocol)

Z-Image-Turbo now includes a powerful **MCP server** that exposes image generation capabilities through the standardized [Model Context Protocol](https://modelcontextprotocol.io). This allows AI assistants (like Claude), automation tools, and other MCP-compatible clients to generate images programmatically.

### Why Use the MCP Server?

- **AI Integration**: Let Claude or other AI assistants generate images directly during conversations
- **Automation**: Build automated workflows that include image generation
- **Remote Access**: Generate images from web clients or remote services (HTTP mode)
- **Standardized API**: Use the same protocol across different AI tools and platforms

### Quick Start with MCP

**1. Install MCP dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

**2. Run the MCP server:**

For local integration (Claude Desktop, MCP Inspector):
```bash
cd backend
./run_mcp.sh --stdio
```

For HTTP/web clients and remote access:
```bash
cd backend
./run_mcp.sh --http --port 8001
# Server available at http://localhost:8001/mcp
```

**3. Configuration:**
Edit `backend/mcp_config.json` to set default transport mode and port:
```json
{
  "transport": "stdio",
  "host": "0.0.0.0",
  "port": 8001
}
```

### Available MCP Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| **`generate_image`** | Generate images from text prompts | `prompt`, `width`, `height`, `num_inference_steps`, `guidance_scale`, `seed` |
| **`get_model_info`** | Get model status and configuration | None |
| **`update_model_config`** | Modify model settings dynamically | `cache_dir`, `cpu_offload` |
| **Resource: `image://examples`** | Access curated example prompts and tips | None |

### Usage Example

Once connected to Claude Desktop or another MCP client:

```
You: "Generate an image of a serene mountain landscape at sunset"

Claude: [Uses generate_image tool]
{
  "prompt": "A serene mountain landscape at sunset with vibrant orange and purple skies",
  "width": 1024,
  "height": 768,
  "num_inference_steps": 8
}

[Returns base64-encoded image]
```

### Transport Modes Comparison

| Feature | Stdio Mode | HTTP/SSE Mode |
|---------|-----------|---------------|
| **Use Case** | Local desktop integration | Web clients, remote access |
| **Best For** | Claude Desktop, MCP Inspector | Production APIs, multi-user |
| **Network** | Local only | Network accessible |
| **Setup** | Simpler | Requires port configuration |

### Claude Desktop Integration

Add to your Claude Desktop config file (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "z-image-turbo": {
      "command": "python",
      "args": ["/absolute/path/to/z-image-turbo/backend/mcp_server.py", "--transport", "stdio"]
    }
  }
}
```

**Important**: Replace `/absolute/path/to/z-image-turbo` with your actual installation path.

After restarting Claude Desktop, you can ask Claude to generate images and it will use the MCP server automatically!

### Testing Your MCP Server

Test with the official MCP Inspector:
```bash
npx @modelcontextprotocol/inspector python backend/mcp_server.py --transport stdio
```

This opens a web interface where you can test all available tools and inspect requests/responses.

📖 **Full MCP Documentation:** See [`backend/MCP_README.md`](backend/MCP_README.md) for detailed setup, Python client examples, troubleshooting, and complete API reference.

---

## ✨ Features

### Application
- **Premium Dark UI** — Glassmorphism design with intuitive controls
- **Smart Presets** — Quick aspect ratios (1:1, 3:4, 16:9) and resolutions (480p-1080p)
- **Fine Control** — Sliders for dimensions, inference steps, guidance scale, and seed
- **Real-time Progress** — Live generation tracking
- **Flexible Deployment** — Custom model cache directory, CPU offload option

### MCP Server Integration
- **🔌 Dual Transport Modes** — Support for both stdio (local) and HTTP/SSE (remote) connections
- **🤖 AI Assistant Compatible** — Seamless integration with Claude Desktop and other MCP clients
- **🛠️ Rich Tool Set** — Image generation, model info, configuration management, and example prompts
- **⚙️ Configurable** — Customizable host, port, and transport settings via `mcp_config.json`
- **🔒 Production Ready** — Stateless HTTP mode for scalable deployments

### Model (Z-Image-Turbo)
- **⚡ Lightning Fast** — Optimized for **8-step generation**, achieving sub-second latency on enterprise GPUs.
- **🏗️ S3-DiT Architecture** — Built on **Scalable Single-Stream Diffusion Transformer** technology.
- **🧠 Advanced Encoders** — Uses **Qwen 4B** for powerful language understanding and **Flux VAE** for image decoding.
- **🎓 DMDR Training** — Trained using **Fusing DMD with Reinforcement Learning** for superior semantic alignment.
- **🌐 Bilingual Mastery** — Exceptional rendering of text in both **English and Chinese**.
- **🎨 Versatile & Uncensored** — From photorealism to anime, handling complex concepts without censorship.
- **📐 High Fidelity** — Native support for resolutions up to **2MP** (e.g., 1024x1536, 1440x1440).
- **💾 Efficient** — 6B parameters, comfortably fitting in 16GB VRAM (consumer-friendly).

---

## 🔬 Technical Architecture

Z-Image-Turbo represents a significant leap in efficient generative AI:

*   **Base Architecture**: S3-DiT (Scalable Single-Stream DiT)
*   **Text Encoder**: Qwen 4B (Large Language Model based conditioning)
*   **VAE**: Flux Autoencoder
*   **Training Method**: Distilled from Z-Image using DMDR (DMD + RL)
*   **Inference**: 8 NFEs (Number of Function Evaluations) default
*   **Precision**: Optimized for bfloat16 / fp8

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, PyTorch, Diffusers, Transformers
- **Frontend:** React, Vite, Lucide React
- **MCP Server:** FastMCP, Starlette (supports stdio and HTTP/SSE transports)
- **Model:** Tongyi-MAI/Z-Image-Turbo (6B parameters)

---

## ⚙️ Configuration

Access settings via the gear icon in the sidebar:
- **Model Cache Directory** — Specify where to download/store the model
- **CPU Offload** — Enable for GPUs with limited VRAM

---

## 📝 License

This project is open-source under the Apache 2.0 License.

---

## 🙏 Credits

- **Model:** [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by Alibaba Group
- **UI Framework:** React + Vite
- **Backend:** FastAPI + Diffusers

