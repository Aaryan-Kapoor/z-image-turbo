"""
MCP Server for Z-Image-Turbo
Provides tools for AI image generation using the Z-Image-Turbo model.
"""

import json
import os
import sys
import base64
from io import BytesIO
from typing import Optional, Literal
import asyncio
import logging

# Configure logging to stderr to avoid corrupting stdio communication
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
    import torch
    from diffusers import DiffusionPipeline
    from PIL import Image
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    logger.error("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


# Load configuration
def load_config():
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {
            "cache_dir": "./models",
            "model_id": "Tongyi-MAI/Z-Image-Turbo",
            "cpu_offload": False
        }


# Load MCP server configuration
def load_mcp_config():
    """Load MCP server configuration"""
    config_path = os.path.join(os.path.dirname(__file__), "mcp_config.json")
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"MCP config file not found at {config_path}, using defaults")
        return {
            "transport": "stdio",
            "host": "0.0.0.0",
            "port": 8001
        }


# Global variables for model caching
pipe = None
config = load_config()


def get_pipeline():
    """Lazy load and cache the diffusion pipeline"""
    global pipe
    if pipe is None:
        logger.info("Loading Z-Image-Turbo model...")
        model_id = config.get("model_id", "Tongyi-MAI/Z-Image-Turbo")
        cache_dir = config.get("cache_dir", "./models")
        cpu_offload = config.get("cpu_offload", False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=cache_dir
        )

        if cpu_offload and device == "cuda":
            logger.info("Enabling CPU offload for memory optimization")
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to(device)

        logger.info(f"Model loaded successfully on {device}")

    return pipe


# Create FastMCP server
mcp = FastMCP("z-image-turbo")


@mcp.tool()
async def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
    seed: Optional[int] = None
) -> str:
    """
    Generate an image from a text prompt using the Z-Image-Turbo model.

    Args:
        prompt: Text description of the image to generate
        width: Image width in pixels (default: 1024)
        height: Image height in pixels (default: 1024)
        num_inference_steps: Number of denoising steps (default: 8, range: 1-50)
        guidance_scale: Classifier-free guidance scale (default: 0.0, range: 0.0-20.0)
        seed: Random seed for reproducibility (optional)

    Returns:
        Base64-encoded PNG image data
    """
    logger.info(f"Generating image with prompt: '{prompt}'")

    try:
        # Get the pipeline
        pipeline = get_pipeline()

        # Set up generator with seed if provided
        generator = None
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
            logger.info(f"Using seed: {seed}")

        # Generate image
        result = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        # Extract the image
        image = result.images[0]

        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        logger.info("Image generated successfully")
        return img_base64

    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool()
async def get_model_info() -> dict:
    """
    Get information about the currently loaded Z-Image-Turbo model.

    Returns:
        Dictionary containing model configuration and status
    """
    global pipe, config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_loaded = pipe is not None

    return {
        "model_id": config.get("model_id", "Tongyi-MAI/Z-Image-Turbo"),
        "cache_dir": config.get("cache_dir", "./models"),
        "cpu_offload": config.get("cpu_offload", False),
        "device": device,
        "is_loaded": is_loaded,
        "cuda_available": torch.cuda.is_available(),
        "default_width": 1024,
        "default_height": 1024,
        "default_steps": 8,
        "default_guidance_scale": 0.0
    }


@mcp.tool()
async def update_model_config(
    cache_dir: Optional[str] = None,
    cpu_offload: Optional[bool] = None
) -> dict:
    """
    Update the model configuration settings.
    Note: Requires reloading the model to take effect.

    Args:
        cache_dir: Path to the model cache directory (optional)
        cpu_offload: Enable CPU offload for memory optimization (optional)

    Returns:
        Updated configuration dictionary
    """
    global config, pipe

    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")

    # Update config values
    if cache_dir is not None:
        config["cache_dir"] = cache_dir

    if cpu_offload is not None:
        config["cpu_offload"] = cpu_offload

    # Save updated config
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration updated: {config}")

        # Reset pipeline to force reload with new config
        if pipe is not None:
            logger.info("Model will be reloaded with new settings on next generation")
            pipe = None

        return {
            "status": "success",
            "message": "Configuration updated. Model will reload with new settings.",
            "config": config
        }
    except Exception as e:
        error_msg = f"Error updating configuration: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.resource("image://examples")
async def get_example_prompts() -> str:
    """
    Get example prompts for image generation.
    """
    examples = [
        "A serene mountain landscape at sunset with vibrant orange and purple skies",
        "A futuristic city with flying cars and neon lights",
        "A cute robot playing with a kitten in a garden",
        "An astronaut riding a horse on Mars",
        "A magical forest with glowing mushrooms and fairy lights",
        "A steampunk airship flying through clouds",
        "A cozy coffee shop on a rainy day, warm lighting",
        "A majestic dragon perched on a castle tower"
    ]

    return json.dumps({
        "example_prompts": examples,
        "tips": [
            "Be specific and descriptive in your prompts",
            "Include details about lighting, mood, and style",
            "The model works best with 8 inference steps (default)",
            "Use guidance_scale=0.0 for fastest generation"
        ]
    }, indent=2)


def main():
    """Main entry point for the MCP server"""
    import argparse

    parser = argparse.ArgumentParser(description="Z-Image-Turbo MCP Server")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "streamable-http"],
        help="Transport mode (stdio or streamable-http)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for HTTP transport (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port for HTTP transport (default: 8001)"
    )

    args = parser.parse_args()

    # Load config and apply overrides
    mcp_config = load_mcp_config()

    transport = args.transport or mcp_config.get("transport", "stdio")
    host = args.host or mcp_config.get("host", "0.0.0.0")
    port = args.port or mcp_config.get("port", 8001)

    logger.info(f"Starting Z-Image-Turbo MCP server with transport: {transport}")

    if transport == "stdio":
        logger.info("Running in stdio mode for local client integration")
        mcp.run(transport="stdio")
    elif transport == "streamable-http":
        logger.info(f"Running HTTP server on {host}:{port}")
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        from starlette.responses import JSONResponse
        from starlette.middleware.cors import CORSMiddleware
        from starlette.requests import Request

        # Health check endpoint
        async def health_check(request):
            return JSONResponse({
                "status": "healthy",
                "service": "z-image-turbo-mcp",
                "transport": "streamable-http"
            })

        # API endpoint to list tools (for Test UI)
        async def list_tools(request):
            tools = await mcp.list_tools()
            return JSONResponse({
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema
                    } for tool in tools
                ]
            })

        # API endpoint to call tools (for Test UI)
        async def call_tool_api(request: Request):
            try:
                data = await request.json()
                name = data.get("name")
                arguments = data.get("arguments", {})
                
                if not name:
                    return JSONResponse({"error": "Tool name required"}, status_code=400)

                result = await mcp.call_tool(name, arguments)
                
                def serialize(obj):
                    # Handle Pydantic v2
                    if hasattr(obj, "model_dump"):
                        return serialize(obj.model_dump())
                    # Handle Pydantic v1
                    if hasattr(obj, "dict") and callable(obj.dict):
                        return serialize(obj.dict())
                    # Handle Dataclasses
                    if hasattr(obj, "__dataclass_fields__"):
                        from dataclasses import asdict
                        return serialize(asdict(obj))
                    # Handle Lists/Tuples
                    if isinstance(obj, (list, tuple)):
                        return [serialize(item) for item in obj]
                    # Handle Dictionaries
                    if isinstance(obj, dict):
                        return {k: serialize(v) for k, v in obj.items()}
                    # Handle Enums
                    if hasattr(obj, "value") and hasattr(obj, "name"):
                        return obj.value
                    # Fallback
                    return obj

                serialized_result = serialize(result)
                
                # Return just the content array, no extra wrapping
                return JSONResponse({"content": serialized_result})

            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}")
                import traceback
                traceback.print_exc()
                return JSONResponse({"error": str(e)}, status_code=500)

        # Create Starlette app with MCP mounted
        app = Starlette(
            routes=[
                Route("/health", health_check),
                Route("/api/tools/list", list_tools, methods=["POST", "GET"]),
                Route("/api/tools/call", call_tool_api, methods=["POST"]),
                Mount("/mcp", app=mcp.sse_app()),
            ]
        )

        # Add CORS middleware to allow browser access
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        uvicorn.run(app, host=host, port=port)
    else:
        logger.error(f"Unknown transport: {transport}")
        sys.exit(1)


if __name__ == "__main__":
    main()
