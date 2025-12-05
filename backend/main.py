from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import threading
import torch
import io
import base64
import os
import json
from fastapi.middleware.cors import CORSMiddleware

try:
    from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig
except ImportError as e:
    print(f"Failed to import from diffusers: {e}")
    print("Please install diffusers from source: pip install git+https://github.com/huggingface/diffusers.git")
    ZImagePipeline = None
    ZImageTransformer2DModel = None
    GGUFQuantizationConfig = None

try:
    from huggingface_hub import hf_hub_download, list_repo_files, hf_hub_url
    HF_HUB_AVAILABLE = True
except ImportError:
    print("huggingface_hub not installed. Model downloading will not work.")
    HF_HUB_AVAILABLE = False
    hf_hub_download = None
    list_repo_files = None
    hf_hub_url = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global download progress tracking
download_progress = {}
download_lock = threading.Lock()

CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    return {
        "cache_dir": None,
        "model_id": "Tongyi-MAI/Z-Image-Turbo",
        "gguf_filename": None,
        "cpu_offload": False
    }

def save_config(config):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")

import psutil
import time

# Global configuration
model_config = load_config()
if "cpu_offload" not in model_config:
    model_config["cpu_offload"] = False

# Global variable for the pipeline
pipe = None

def get_pipeline():
    global pipe
    if pipe is None:
        if ZImagePipeline is None:
            raise HTTPException(status_code=500, detail="ZImagePipeline class not available. Install diffusers from source.")

        print(f"Loading model {model_config['model_id']}...")

        if model_config['cache_dir']:
            print(f"Using cache directory: {model_config['cache_dir']}")

        try:
            # Check for CUDA
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            # Check for GGUF configuration
            gguf_file = model_config.get("gguf_filename")
            model_id = model_config.get("model_id", "Tongyi-MAI/Z-Image-Turbo")

            # Logic to handle GGUF loading
            if gguf_file or model_id.endswith(".gguf"):
                print(f"Detected GGUF configuration. Loading transformer from GGUF...")

                if ZImageTransformer2DModel is None or GGUFQuantizationConfig is None:
                    raise HTTPException(
                        status_code=500,
                        detail="GGUF support requires latest diffusers. Run: pip install git+https://github.com/huggingface/diffusers.git"
                    )

                # Build GGUF URL for HuggingFace
                gguf_url = f"https://huggingface.co/{model_id}/blob/main/{gguf_file}"
                print(f"Loading GGUF transformer from: {gguf_url}")

                # Step 1: Load the quantized transformer from GGUF
                transformer = ZImageTransformer2DModel.from_single_file(
                    gguf_url,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                    torch_dtype=dtype,
                )
                print("GGUF transformer loaded successfully")

                # Step 2: Load the full pipeline from original model, inject quantized transformer
                original_model_id = "Tongyi-MAI/Z-Image-Turbo"
                print(f"Loading pipeline components from {original_model_id}...")
                pipe = ZImagePipeline.from_pretrained(
                    original_model_id,
                    transformer=transformer,
                    torch_dtype=dtype,
                    cache_dir=model_config.get('cache_dir')
                )
                print("Pipeline assembled with GGUF transformer")
            else:
                # Standard loading
                pipe = ZImagePipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=False,
                    cache_dir=model_config['cache_dir']
                )

            # For GGUF models, always use enable_model_cpu_offload (recommended)
            # For standard models, use cpu_offload setting or move to device
            is_gguf = gguf_file or model_id.endswith(".gguf")
            if is_gguf and device == "cuda":
                print("Enabling CPU offload for GGUF model (recommended)")
                pipe.enable_model_cpu_offload()
            elif model_config.get("cpu_offload", False) and device == "cuda":
                print("Enabling CPU Offload")
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)

            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    return pipe

class SettingsRequest(BaseModel):
    cache_dir: str
    cpu_offload: bool = False
    model_id: str = None
    gguf_filename: str = None

@app.post("/settings/model-path")
async def set_model_path(req: SettingsRequest):
    global pipe
    try:
        if req.cache_dir and not os.path.exists(req.cache_dir):
            os.makedirs(req.cache_dir, exist_ok=True)

        model_config["cache_dir"] = req.cache_dir
        model_config["cpu_offload"] = req.cpu_offload

        # Update model ID and filename if provided
        if req.model_id is not None:
            model_config["model_id"] = req.model_id
        if req.gguf_filename is not None:
            model_config["gguf_filename"] = req.gguf_filename if req.gguf_filename else None

        save_config(model_config)
        # Force reload of the pipeline
        pipe = None
        return {"status": "success", "message": "Settings saved. Model will reload on next generation."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/settings")
async def get_settings():
    return model_config

class GenerateRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    steps: int = 8
    guidance_scale: float = 0.0
    seed: int = -1

@app.post("/generate")
def generate_image(req: GenerateRequest):
    # Validate dimensions
    if req.height % 16 != 0 or req.width % 16 != 0:
        raise HTTPException(status_code=400, detail="Height and Width must be divisible by 16.")

    try:
        pipeline = get_pipeline()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = None
        if req.seed != -1:
            generator = torch.Generator(device).manual_seed(req.seed)
        
        # Run inference
        print(f"Generating with prompt: {req.prompt}")
        
        image = pipeline(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            generator=generator,
        ).images[0]
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"image": f"data:image/png;base64,{img_str}"}
    except Exception as e:
        print(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/available")
async def list_available_models():
    """List available GGUF models from AaryanK/Z-Image-Turbo-GGUF repository"""
    if not HF_HUB_AVAILABLE:
        raise HTTPException(status_code=500, detail="huggingface_hub not installed")

    repo_id = "AaryanK/Z-Image-Turbo-GGUF"

    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        print(f"Error listing repo files: {e}")
        # Return hardcoded list as fallback
        return {
            "models": [
                {"filename": "z_image_turbo-Q4_K_M.gguf", "size_gb": 5.2, "quantization": "Q4_K_M (4-bit)", "description": "Balanced 4-bit, good quality/size trade-off"},
                {"filename": "z_image_turbo-Q5_K_M.gguf", "size_gb": 6.4, "quantization": "Q5_K_M (5-bit)", "description": "Higher quality 5-bit quantization"},
                {"filename": "z_image_turbo-Q6_K.gguf", "size_gb": 7.6, "quantization": "Q6_K (6-bit)", "description": "High quality 6-bit, near full precision"},
                {"filename": "z_image_turbo-Q8_0.gguf", "size_gb": 9.8, "quantization": "Q8_0 (8-bit)", "description": "Very high quality 8-bit quantization"},
            ],
            "repo_id": repo_id,
            "note": "Using cached model list"
        }

    # Filter for GGUF files and add metadata
    gguf_models = []
    for file in files:
        if file.endswith(".gguf"):
            quant = "Unknown"
            size_gb = 0.0
            desc = ""

            if "Q4_K_M" in file:
                quant = "Q4_K_M (4-bit)"
                size_gb = 5.2
                desc = "Balanced 4-bit, good quality/size trade-off"
            elif "Q5_K_M" in file:
                quant = "Q5_K_M (5-bit)"
                size_gb = 6.4
                desc = "Higher quality 5-bit quantization"
            elif "Q6_K" in file:
                quant = "Q6_K (6-bit)"
                size_gb = 7.6
                desc = "High quality 6-bit, near full precision"
            elif "Q8_0" in file:
                quant = "Q8_0 (8-bit)"
                size_gb = 9.8
                desc = "Very high quality 8-bit quantization"
            elif "f16" in file.lower():
                quant = "FP16"
                size_gb = 12.0
                desc = "Full half-precision, maximum quality"

            gguf_models.append({
                "filename": file,
                "size_gb": size_gb,
                "quantization": quant,
                "description": desc
            })

    return {"models": gguf_models, "repo_id": repo_id}

class DownloadRequest(BaseModel):
    filename: str

@app.post("/models/download")
async def download_model(req: DownloadRequest):
    """Download a specific GGUF model from AaryanK repository"""
    if not HF_HUB_AVAILABLE:
        raise HTTPException(status_code=500, detail="huggingface_hub not installed. Run: pip install huggingface_hub")

    filename = req.filename
    repo_id = "AaryanK/Z-Image-Turbo-GGUF"
    cache_dir = model_config.get("cache_dir") or "./models"

    print(f"Starting download of {filename} to {cache_dir}")

    try:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot create cache directory: {str(e)}")

    # Initialize progress tracking
    with download_lock:
        download_progress[filename] = {
            "status": "starting",
            "progress": 0,
            "message": "Initializing download..."
        }

    # Start download in background
    def download_task():
        try:
            with download_lock:
                download_progress[filename] = {
                    "status": "downloading",
                    "progress": 0,
                    "message": "Downloading from HuggingFace..."
                }

            print(f"Downloading {filename} from {repo_id}...")

            # Download the file
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                force_download=False
            )

            print(f"Download completed: {local_path}")

            with download_lock:
                download_progress[filename] = {
                    "status": "completed",
                    "progress": 100,
                    "message": "Download completed",
                    "path": local_path
                }

        except Exception as e:
            print(f"Download error: {str(e)}")
            with download_lock:
                download_progress[filename] = {
                    "status": "error",
                    "progress": 0,
                    "message": str(e)
                }

    thread = threading.Thread(target=download_task)
    thread.start()

    return {
        "status": "started",
        "message": "Download started in background",
        "filename": filename
    }

@app.get("/models/download-progress/{filename:path}")
async def get_download_progress(filename: str):
    """Get download progress for a specific model"""
    with download_lock:
        if filename in download_progress:
            return download_progress[filename]
        else:
            return {
                "status": "not_found",
                "progress": 0,
                "message": "No download found for this file"
            }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
