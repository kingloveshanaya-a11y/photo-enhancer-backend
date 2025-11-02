from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image, ImageEnhance
import torch
import io
import os
import numpy as np
import traceback
import cv2
import urllib.request
from realesrgan import RealESRGANer

router = APIRouter()

# === CONFIG ===
WEIGHTS_DIR = "weights"
MODELS = {
    "photo": {
        "file": "realesr-general-x4v3.pth",
        "url": "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/realesr-general-x4v3.pth"
    },
    "anime": {
        "file": "realesr-animevideov3.pth",
        "url": "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/realesr-animevideov3.pth"
    }
}

# === GLOBAL MODEL INSTANCES ===
upsamplers = {"photo": None, "anime": None}


# --- Helper: auto-detect image type ---
def detect_image_type(img: np.ndarray) -> str:
    """Auto-detect anime vs photo based on edge and saturation levels."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    sat = np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1])
    # Heuristic: anime tends to have sharper edges or higher color saturation
    if edge_density > 0.08 or sat > 100:
        return "anime"
    return "photo"


# --- Model management ---
def ensure_model_exists(model_type: str):
    """Ensure a RealESRGAN model is downloaded."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_info = MODELS[model_type]
    model_path = os.path.join(WEIGHTS_DIR, model_info["file"])
    if not os.path.exists(model_path):
        print(f"⚡ Downloading {model_type} model weights...")
        urllib.request.urlretrieve(model_info["url"], model_path)
        print(f"✅ Downloaded {model_path}")
    return model_path


def init_model(model_type: str):
    """Initialize a RealESRGAN model safely (auto-detect architecture)."""
    global upsamplers
    if upsamplers.get(model_type):
        return upsamplers[model_type]

    model_path = ensure_model_exists(model_type)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device == "cuda"

    print(f"🧠 Initializing RealESRGAN '{model_type}' model on {device}...")

    # ✅ Let RealESRGANer choose the correct model architecture
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=None,   # <-- auto architecture selection fix
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=device,
    )

    upsamplers[model_type] = upsampler
    print(f"🚀 RealESRGAN '{model_type}' model initialized successfully.")
    return upsampler


# === STARTUP EVENT ===
@router.on_event("startup")
def startup_event():
    """Preload both models at startup for smoother first use."""
    for model_type in MODELS.keys():
        try:
            init_model(model_type)
        except Exception as e:
            print(f"⚠️ Could not preload {model_type} model: {e}")


# === MAIN ENHANCE ENDPOINT ===
@router.post("/enhance")
async def enhance_photo(
    file: UploadFile = File(...),
    type: str = Query("auto", description="Image type: 'photo', 'anime', or 'auto'"),
    scale: int = Query(4, ge=2, le=8, description="Upscale factor (2, 4, or 8)"),
    denoise: float = Query(0.5, ge=0.0, le=1.0, description="Denoise level (0=weak, 1=strong)")
):
    """
    Enhance an image intelligently using RealESRGAN.
    Automatically detects photo/anime and applies adaptive color and tone.
    """
    try:
        img_bytes = await file.read()
        input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_np = np.array(input_image)

        # Auto or manual type selection
        model_type = type.lower().strip()
        if model_type == "auto":
            model_type = detect_image_type(input_np)
            print(f"🧭 Auto-detected image type: {model_type}")

        if model_type not in MODELS:
            raise HTTPException(status_code=400, detail="Invalid type. Use 'photo', 'anime', or 'auto'.")

        upsampler = upsamplers.get(model_type) or init_model(model_type)

        # --- Enhancement ---
        result = upsampler.enhance(input_np, outscale=scale, denoise=denoise)
        output_np = result[0] if isinstance(result, tuple) else result
        output_image = Image.fromarray(output_np)

        # --- Post-processing style tuning ---
        if model_type == "photo":
            # Natural, soft, realistic beauty
            output_image = ImageEnhance.Color(output_image).enhance(1.05)
            output_image = ImageEnhance.Contrast(output_image).enhance(1.02)
            output_image = ImageEnhance.Sharpness(output_image).enhance(0.9)
        elif model_type == "anime":
            # Vibrant, sharp, high-contrast anime look
            output_image = ImageEnhance.Contrast(output_image).enhance(1.15)
            output_image = ImageEnhance.Color(output_image).enhance(1.25)
            output_image = ImageEnhance.Sharpness(output_image).enhance(1.1)

        # --- Return result ---
        buf = io.BytesIO()
        output_image.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        print(f"✨ {model_type.capitalize()} enhancement completed (x{scale}).")
        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        print("❌ Enhancement error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
