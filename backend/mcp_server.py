"""
MCP Server for Z-Image-Turbo
Provides tools for AI image generation using the Z-Image-Turbo model.

Production-ready features:
- Configurable eager/lazy model loading
- Model TTL (auto-unload after inactivity)
- Concurrent request limiting
- Configurable logging
"""

import json
import os
import sys
import base64
import time
import threading
from io import BytesIO
from typing import Optional
import asyncio
import logging

# Initial logging setup (will be reconfigured based on config)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ImageContent
    import torch
    from diffusers import DiffusionPipeline, ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig
    from PIL import Image
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    logger.error("Please install dependencies: pip install -r backend/requirements.txt")
    logger.error("For GGUF support: pip install git+https://github.com/huggingface/diffusers.git")
    sys.exit(1)

# HuggingFace Hub for model management
try:
    from huggingface_hub import hf_hub_download, list_repo_files, scan_cache_dir
    HF_HUB_AVAILABLE = True
except ImportError:
    logger.warning("huggingface_hub not installed. Model listing/downloading will not work.")
    HF_HUB_AVAILABLE = False
    hf_hub_download = None
    list_repo_files = None
    scan_cache_dir = None

# GGUF model metadata
GGUF_REPO_ID = "AaryanK/Z-Image-Turbo-GGUF"
GGUF_QUANT_INFO = {
    "Q3_K_S": {"bits": 3, "size_gb": 3.79, "desc": "3-bit small, fastest"},
    "Q3_K_M": {"bits": 3, "size_gb": 4.12, "desc": "3-bit medium"},
    "Q3_K": {"bits": 3, "size_gb": 4.12, "desc": "3-bit standard"},
    "Q4_0": {"bits": 4, "size_gb": 4.59, "desc": "4-bit legacy"},
    "Q4_1": {"bits": 4, "size_gb": 4.85, "desc": "4-bit improved"},
    "Q4_K_S": {"bits": 4, "size_gb": 4.66, "desc": "4-bit small K-quant"},
    "Q4_K_M": {"bits": 4, "size_gb": 4.98, "desc": "4-bit medium, recommended"},
    "Q5_0": {"bits": 5, "size_gb": 5.26, "desc": "5-bit legacy"},
    "Q5_1": {"bits": 5, "size_gb": 5.53, "desc": "5-bit improved"},
    "Q5_K_S": {"bits": 5, "size_gb": 5.19, "desc": "5-bit small K-quant"},
    "Q5_K_M": {"bits": 5, "size_gb": 5.52, "desc": "5-bit medium, good quality"},
    "Q6_K": {"bits": 6, "size_gb": 5.91, "desc": "6-bit, near lossless"},
    "Q8_0": {"bits": 8, "size_gb": 7.22, "desc": "8-bit, very high quality"},
    "bf16": {"bits": 16, "size_gb": 12.3, "desc": "BFloat16, maximum quality"},
    "f16": {"bits": 16, "size_gb": 12.3, "desc": "Float16, maximum quality"},
}


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
            "gguf_filename": None,
            "cpu_offload": False
        }


def load_mcp_config():
    """Load MCP server configuration"""
    config_path = os.path.join(os.path.dirname(__file__), "mcp_config.json")
    defaults = {
        "transport": "stdio",
        "host": "0.0.0.0",
        "port": 8001,
        "eager_load": False,
        "model_ttl_minutes": 0,
        "max_concurrent_requests": 1,
        "log_level": "INFO"
    }
    try:
        with open(config_path, "r") as f:
            loaded = json.load(f)
            # Merge with defaults (loaded values take precedence)
            return {**defaults, **loaded}
    except FileNotFoundError:
        logger.warning(f"MCP config file not found at {config_path}, using defaults")
        return defaults


# Global state
pipe = None
config = load_config()
mcp_config = load_mcp_config()
last_used_time = None
model_lock = threading.Lock()
request_semaphore = None  # Will be initialized in main()

def _load_pipeline_internal():
    """Internal function to load the pipeline (must be called with lock held)"""
    global pipe, last_used_time

    logger.info("Loading Z-Image-Turbo model... (this may take 30-60 seconds on first run)")
    model_id = config.get("model_id", "Tongyi-MAI/Z-Image-Turbo")
    gguf_filename = config.get("gguf_filename")
    cache_dir = config.get("cache_dir", "./models")
    cpu_offload = config.get("cpu_offload", False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # GGUF Loading Logic
    is_gguf = gguf_filename or model_id.endswith(".gguf")
    if is_gguf:
        logger.info(f"Loading GGUF model: {model_id} ({gguf_filename if gguf_filename else 'local'})")

        # Build GGUF URL for HuggingFace
        gguf_url = f"https://huggingface.co/{model_id}/blob/main/{gguf_filename}"
        logger.info(f"Loading GGUF transformer from: {gguf_url}")

        # Step 1: Load the quantized transformer from GGUF
        transformer = ZImageTransformer2DModel.from_single_file(
            gguf_url,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        logger.info("GGUF transformer loaded successfully")

        # Step 2: Load the full pipeline from original model, inject quantized transformer
        original_model_id = "Tongyi-MAI/Z-Image-Turbo"
        logger.info(f"Loading pipeline components from {original_model_id}...")
        new_pipe = ZImagePipeline.from_pretrained(
            original_model_id,
            transformer=transformer,
            torch_dtype=dtype,
            cache_dir=cache_dir
        )
        logger.info("Pipeline assembled with GGUF transformer")
    else:
        # Standard Loading
        new_pipe = ZImagePipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=cache_dir
        )

    # For GGUF models, always use enable_model_cpu_offload (recommended)
    if is_gguf and device == "cuda":
        logger.info("Enabling CPU offload for GGUF model (recommended)")
        new_pipe.enable_model_cpu_offload()
    elif cpu_offload and device == "cuda":
        logger.info("Enabling CPU offload for memory optimization")
        new_pipe.enable_model_cpu_offload()
    else:
        new_pipe = new_pipe.to(device)

    # Log GPU memory usage if available
    if device == "cuda":
        try:
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"Model loaded on {device} (GPU memory: {memory_gb:.2f} GB)")
        except Exception:
            logger.info(f"Model loaded successfully on {device}")
    else:
        logger.info(f"Model loaded successfully on {device}")

    return new_pipe


def load_pipeline():
    """Load the diffusion pipeline into memory (thread-safe)"""
    global pipe, last_used_time
    
    with model_lock:
        if pipe is not None:
            last_used_time = time.time()
            return pipe
        
        pipe = _load_pipeline_internal()
        last_used_time = time.time()
        return pipe


def unload_pipeline():
    """Unload the model from memory to free resources"""
    global pipe, last_used_time
    
    with model_lock:
        if pipe is not None:
            logger.info("Unloading model to free memory...")
            del pipe
            pipe = None
            last_used_time = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cache cleared")
            
            logger.info("Model unloaded successfully")


def get_pipeline():
    """Get the pipeline, loading it if necessary (supports lazy loading)"""
    global pipe, last_used_time
    
    with model_lock:
        if pipe is None:
            logger.info("Lazy loading model on first request...")
            pipe = _load_pipeline_internal()
        
        last_used_time = time.time()
        return pipe



def start_ttl_monitor():
    """Start background thread to monitor model TTL and unload when idle"""
    ttl_minutes = mcp_config.get("model_ttl_minutes", 0)
    
    if ttl_minutes <= 0:
        logger.debug("Model TTL disabled (model will stay loaded indefinitely)")
        return
    
    ttl_seconds = ttl_minutes * 60
    logger.info(f"Model TTL enabled: will unload after {ttl_minutes} minutes of inactivity")
    
    def monitor():
        global pipe, last_used_time
        while True:
            time.sleep(60)  # Check every minute
            
            with model_lock:
                if pipe is not None and last_used_time is not None:
                    idle_time = time.time() - last_used_time
                    if idle_time >= ttl_seconds:
                        logger.info(f"Model idle for {idle_time/60:.1f} minutes, unloading...")
                        unload_pipeline()
    
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()


# Create FastMCP server
mcp = FastMCP("z-image-turbo")


@mcp.tool()
async def generate_image(
    prompt: str,
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 5,
    guidance_scale: float = 0.0,
    seed: Optional[int] = None
) -> ImageContent:
    """
    Generate an image from a text prompt using the Z-Image-Turbo model.

    Args:
        prompt: Text description of the image to generate
        width: Image width in pixels (default: 512, must be divisible by 16)
        height: Image height in pixels (default: 512, must be divisible by 16)
        num_inference_steps: Number of denoising steps (default: 5, range: 1-50)
        guidance_scale: Classifier-free guidance scale (default: 0.0, range: 0.0-20.0)
        seed: Random seed for reproducibility (optional)

    Returns:
        Generated image in MCP ImageContent format
    """
    global request_semaphore
    
    logger.info(f"Generating image with prompt: '{prompt}'")

    # Input validation
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError(f"Width ({width}) and height ({height}) must be divisible by 16")
    
    if width < 64 or height < 64:
        raise ValueError(f"Minimum dimension is 64 pixels (got {width}x{height})")
    
    if width > 2048 or height > 2048:
        raise ValueError(f"Maximum dimension is 2048 pixels (got {width}x{height})")
    
    if num_inference_steps < 1 or num_inference_steps > 50:
        raise ValueError(f"num_inference_steps must be between 1-50 (got {num_inference_steps})")
    
    if guidance_scale < 0.0 or guidance_scale > 20.0:
        raise ValueError(f"guidance_scale must be between 0.0-20.0 (got {guidance_scale})")

    # Use semaphore to limit concurrent requests (prevents GPU OOM)
    if request_semaphore is not None:
        async with request_semaphore:
            return await _generate_image_internal(prompt, width, height, num_inference_steps, guidance_scale, seed)
    else:
        return await _generate_image_internal(prompt, width, height, num_inference_steps, guidance_scale, seed)


async def _generate_image_internal(prompt, width, height, num_inference_steps, guidance_scale, seed):
    """Internal image generation logic"""
    try:
        # Get the pipeline (will lazy load if needed)
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
        
        # Return as MCP ImageContent for proper rendering in clients
        return ImageContent(
            type="image",
            data=img_base64,
            mimeType="image/png"
        )

    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool()
async def get_model_info() -> dict:
    """
    Get information about the currently loaded Z-Image-Turbo model.

    Returns:
        Dictionary containing model configuration, status, and system info
    """
    global pipe, config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_loaded = pipe is not None

    # Determine GGUF status
    model_id = config.get("model_id", "Tongyi-MAI/Z-Image-Turbo")
    gguf_filename = config.get("gguf_filename")
    is_gguf = bool(gguf_filename) or model_id.endswith(".gguf")

    result = {
        "model_id": model_id,
        "cache_dir": config.get("cache_dir", "./models"),
        "cpu_offload": config.get("cpu_offload", False),
        "device": device,
        "is_loaded": is_loaded,
        "cuda_available": torch.cuda.is_available(),
        # GGUF-specific info
        "gguf": {
            "enabled": is_gguf,
            "filename": gguf_filename,
            "repo_id": "AaryanK/Z-Image-Turbo-GGUF" if is_gguf else None,
            "note": "GGUF models use less VRAM with quantized weights" if is_gguf else None
        },
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

    # Add GPU info if available
    if torch.cuda.is_available():
        try:
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if is_loaded:
                result["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        except Exception:
            # GPU info is optional; ignore errors if unavailable
            pass

    return result


@mcp.tool()
async def update_model_config(
    cache_dir: Optional[str] = None,
    cpu_offload: Optional[bool] = None,
    gguf_filename: Optional[str] = None
) -> dict:
    """
    Update the model configuration settings.
    Note: Requires reloading the model to take effect.

    When setting gguf_filename, the model_id is automatically configured:
    - Set gguf_filename to a .gguf file to use GGUF quantization
    - Set gguf_filename to empty string or "none" to switch back to full model

    Args:
        cache_dir: Path to the model cache directory (optional)
        cpu_offload: Enable CPU offload for memory optimization (optional)
        gguf_filename: GGUF filename to use (e.g., "z_image_turbo-Q4_K_M.gguf"), or empty/"none" for full model

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

    # Handle gguf_filename - automatically set model_id
    if gguf_filename is not None:
        # Normalize empty values
        if gguf_filename.lower() in ("", "none", "null", "false"):
            # Switch to full model
            config["gguf_filename"] = None
            config["model_id"] = "Tongyi-MAI/Z-Image-Turbo"
            logger.info("Switching to full precision model")
        else:
            # Switch to GGUF model
            config["gguf_filename"] = gguf_filename
            config["model_id"] = GGUF_REPO_ID
            logger.info(f"Switching to GGUF model: {gguf_filename}")

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


@mcp.tool()
async def list_available_gguf_models() -> dict:
    """
    List all available GGUF quantized models from AaryanK/Z-Image-Turbo-GGUF.

    Returns a list of models with their sizes, quantization levels, and descriptions.
    Use this to see what GGUF models you can download and use.

    Returns:
        Dictionary containing available models with metadata
    """
    if not HF_HUB_AVAILABLE:
        raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")

    try:
        files = list_repo_files(GGUF_REPO_ID)
    except Exception as e:
        logger.warning(f"Error listing repo files: {e}, using cached list")
        # Fallback to hardcoded list
        return {
            "models": [
                {"filename": "z_image_turbo-Q3_K_S.gguf", "size_gb": 3.79, "quantization": "Q3_K_S (3-bit)", "description": "3-bit small, fastest"},
                {"filename": "z_image_turbo-Q4_K_M.gguf", "size_gb": 4.98, "quantization": "Q4_K_M (4-bit)", "description": "4-bit medium, recommended"},
                {"filename": "z_image_turbo-Q5_K_M.gguf", "size_gb": 5.52, "quantization": "Q5_K_M (5-bit)", "description": "5-bit medium, good quality"},
                {"filename": "z_image_turbo-Q6_K.gguf", "size_gb": 5.91, "quantization": "Q6_K (6-bit)", "description": "6-bit, near lossless"},
                {"filename": "z_image_turbo-Q8_0.gguf", "size_gb": 7.22, "quantization": "Q8_0 (8-bit)", "description": "8-bit, very high quality"},
                {"filename": "z_image_turbo-bf16.gguf", "size_gb": 12.3, "quantization": "bf16 (16-bit)", "description": "BFloat16, maximum quality"},
            ],
            "repo_id": GGUF_REPO_ID,
            "note": "Using cached model list (could not reach HuggingFace)"
        }

    # Build model list with metadata
    gguf_models = []
    for file in files:
        if file.endswith(".gguf"):
            # Find quantization type (match longest first)
            quant_type = None
            for q in sorted(GGUF_QUANT_INFO.keys(), key=len, reverse=True):
                if q in file or q.lower() in file.lower():
                    quant_type = q
                    break

            if quant_type and quant_type in GGUF_QUANT_INFO:
                info = GGUF_QUANT_INFO[quant_type]
                gguf_models.append({
                    "filename": file,
                    "size_gb": info["size_gb"],
                    "quantization": f"{quant_type} ({info['bits']}-bit)",
                    "description": info["desc"]
                })
            else:
                gguf_models.append({
                    "filename": file,
                    "size_gb": 0.0,
                    "quantization": "Unknown",
                    "description": "Unknown quantization"
                })

    # Sort by size
    gguf_models.sort(key=lambda x: (x["size_gb"], x["filename"]))

    return {
        "models": gguf_models,
        "repo_id": GGUF_REPO_ID,
        "current_model": config.get("gguf_filename"),
        "tip": "Use download_gguf_model to download, then update_model_config to switch"
    }


@mcp.tool()
async def list_downloaded_gguf_models() -> dict:
    """
    List GGUF models that have already been downloaded to the cache directory.

    This shows which models are available locally and ready to use without downloading.

    Returns:
        Dictionary containing downloaded models and their paths
    """
    cache_dir = config.get("cache_dir", "./models")

    downloaded = []

    # Check the cache directory for GGUF files
    if cache_dir and os.path.exists(cache_dir):
        # Walk through cache looking for GGUF files
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                if file.endswith(".gguf"):
                    full_path = os.path.join(root, file)
                    size_bytes = os.path.getsize(full_path)
                    size_gb = round(size_bytes / (1024**3), 2)

                    # Extract quantization info
                    quant_type = None
                    for q in sorted(GGUF_QUANT_INFO.keys(), key=len, reverse=True):
                        if q in file or q.lower() in file.lower():
                            quant_type = q
                            break

                    downloaded.append({
                        "filename": file,
                        "path": full_path,
                        "size_gb": size_gb,
                        "quantization": quant_type or "Unknown"
                    })

    # Also check HuggingFace cache if available
    if HF_HUB_AVAILABLE and scan_cache_dir:
        try:
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if GGUF_REPO_ID.replace("/", "--") in str(repo.repo_id) or "Z-Image-Turbo-GGUF" in str(repo.repo_id):
                    for revision in repo.revisions:
                        for file in revision.files:
                            if file.file_name.endswith(".gguf"):
                                size_gb = round(file.size_on_disk / (1024**3), 2)
                                # Avoid duplicates
                                if not any(d["filename"] == file.file_name for d in downloaded):
                                    quant_type = None
                                    for q in sorted(GGUF_QUANT_INFO.keys(), key=len, reverse=True):
                                        if q in file.file_name or q.lower() in file.file_name.lower():
                                            quant_type = q
                                            break
                                    downloaded.append({
                                        "filename": file.file_name,
                                        "path": str(file.file_path),
                                        "size_gb": size_gb,
                                        "quantization": quant_type or "Unknown"
                                    })
        except Exception as e:
            logger.warning(f"Error scanning HF cache: {e}")

    # Get current active model
    current = config.get("gguf_filename")

    return {
        "downloaded_models": downloaded,
        "count": len(downloaded),
        "cache_dir": cache_dir,
        "current_active": current,
        "tip": "Use update_model_config with gguf_filename to switch models"
    }


@mcp.tool()
async def download_gguf_model(filename: str) -> dict:
    """
    Download a specific GGUF model from AaryanK/Z-Image-Turbo-GGUF.

    Downloads the model to the configured cache directory. Use list_available_gguf_models
    to see what models are available for download.

    Args:
        filename: The GGUF filename to download (e.g., "z_image_turbo-Q4_K_M.gguf")

    Returns:
        Dictionary with download status and file path
    """
    if not HF_HUB_AVAILABLE:
        raise RuntimeError("huggingface_hub not installed. Run: pip install huggingface_hub")

    if not filename.endswith(".gguf"):
        raise ValueError(f"Invalid filename: {filename}. Must be a .gguf file")

    cache_dir = config.get("cache_dir", "./models")

    # Create cache dir if needed
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    logger.info(f"Downloading {filename} from {GGUF_REPO_ID}...")

    try:
        # Download the file
        local_path = hf_hub_download(
            repo_id=GGUF_REPO_ID,
            filename=filename,
            cache_dir=cache_dir,
            force_download=False
        )

        # Get file size
        size_bytes = os.path.getsize(local_path)
        size_gb = round(size_bytes / (1024**3), 2)

        logger.info(f"Download completed: {local_path}")

        return {
            "status": "success",
            "filename": filename,
            "path": local_path,
            "size_gb": size_gb,
            "message": f"Downloaded {filename} ({size_gb} GB)",
            "next_step": f"Use update_model_config with gguf_filename='{filename}' to activate"
        }

    except Exception as e:
        error_msg = f"Error downloading {filename}: {str(e)}"
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
    global request_semaphore, mcp_config
    
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
        help="Host address for HTTP transport (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port for HTTP transport (default: 8001)"
    )
    parser.add_argument(
        "--eager-load",
        action="store_true",
        dest="eager_load",
        help="Load model at startup (overrides config)"
    )
    parser.add_argument(
        "--lazy-load",
        action="store_true", 
        dest="lazy_load",
        help="Load model on first request (overrides config)"
    )

    args = parser.parse_args()

    # Reload config (in case it was modified)
    mcp_config = load_mcp_config()
    
    # Configure logging level from config
    log_level = mcp_config.get("log_level", "INFO").upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    logger.info(f"Log level set to {log_level}")

    # Apply command line overrides
    transport = args.transport or mcp_config.get("transport", "stdio")
    host = args.host or mcp_config.get("host", "0.0.0.0")
    port = args.port or mcp_config.get("port", 8001)
    
    # Determine eager loading setting (CLI overrides config)
    if args.eager_load:
        eager_load = True
    elif args.lazy_load:
        eager_load = False
    else:
        eager_load = mcp_config.get("eager_load", True)

    # Initialize request semaphore for concurrency control
    max_concurrent = mcp_config.get("max_concurrent_requests", 1)
    request_semaphore = asyncio.Semaphore(max_concurrent)
    logger.info(f"Max concurrent requests: {max_concurrent}")

    # Start TTL monitor if configured
    start_ttl_monitor()

    # Eager load the model if configured
    if eager_load:
        logger.info("Eager loading model at startup...")
        try:
            load_pipeline()
            logger.info("Model ready! Server is accepting requests.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Server will attempt to load model on first request.")
    else:
        logger.info("Lazy loading enabled - model will load on first request")
        ttl = mcp_config.get("model_ttl_minutes", 0)
        if ttl > 0:
            logger.info(f"Model TTL: {ttl} minutes")

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
