from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image, ImageEnhance
import torch
import io
import os
import numpy as np
import traceback
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

router = APIRouter()

# === CONFIG ===
WEIGHTS_DIR = "weights"
MODELS = {
    "photo": "realesr-general-x4v3.pth",     # for real/human photos
    "anime": "realesr-animevideov3.pth"      # for anime/cartoons
}

# === GLOBAL MODEL INSTANCES ===
upsamplers = {"photo": None, "anime": None}


# --- Helper: detect image type ---
def detect_image_type(img: np.ndarray) -> str:
    """Auto-detect whether image is anime or photo based on color and edge density."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    sat = np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1])
    # Heuristic: anime has stronger edges or high saturation
    if edge_density > 0.08 or sat > 100:
        return "anime"
    return "photo"


# --- Model initialization ---
def ensure_model_exists(model_name: str):
    """Ensure that a specific model file exists locally."""
    model_path = os.path.join(WEIGHTS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Model weights not found: {model_path}\n"
            f"Please download '{model_name}' from the official Real-ESRGAN release "
            f"and place it in the '{WEIGHTS_DIR}' folder."
        )
    return model_path


def init_model(model_type: str):
    """Initialize a specific RealESRGAN model."""
    global upsamplers

    if upsamplers.get(model_type):
        return upsamplers[model_type]

    model_filename = MODELS.get(model_type)
    if not model_filename:
        raise ValueError(f"Invalid model type '{model_type}'. Use 'photo' or 'anime'.")

    model_path = ensure_model_exists(model_filename)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device == "cuda"

    print(f"🧠 Initializing RealESRGAN '{model_type}' model on {device}...")

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=4,
    )

    upsampler = RealESRGANer(
        scale=4, model_path=model_path, model=model,
        tile=512, tile_pad=10, pre_pad=0,
        half=use_half, device=device,
    )

    upsamplers[model_type] = upsampler
    print(f"🚀 RealESRGAN '{model_type}' model initialized successfully.")
    return upsampler


# === STARTUP EVENT ===
@router.on_event("startup")
def startup_event():
    """Initialize both models at startup (optional)."""
    for model_type in MODELS.keys():
        try:
            init_model(model_type)
        except Exception as e:
            print(f"⚠️ Failed to load {model_type} model: {e}")


# === MAIN ENDPOINT ===
@router.post("/enhance")
async def enhance_photo(
    file: UploadFile = File(...),
    type: str = Query("auto", description="Image type: 'photo', 'anime', or 'auto'"),
    scale: int = Query(4, ge=2, le=8, description="Upscale factor (2, 4, or 8)"),
    denoise: float = Query(0.5, ge=0.0, le=1.0, description="Denoise level (0=weak, 1=strong)"),
):
    """
    Enhance an image intelligently using RealESRGAN.
    Auto-detects between photo/anime and applies the appropriate model and tone balance.
    """
    try:
        img_bytes = await file.read()
        input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_np = np.array(input_image)

        # Determine model type
        model_type = type.lower().strip()
        if model_type == "auto":
            model_type = detect_image_type(input_np)
            print(f"🧭 Auto-detected image type: {model_type}")

        if model_type not in MODELS:
            raise HTTPException(status_code=400, detail="Invalid type. Use 'photo', 'anime', or 'auto'.")

        upsampler = upsamplers.get(model_type) or init_model(model_type)

        # Perform enhancement
        result = upsampler.enhance(input_np, outscale=scale, denoise=denoise)
        output_np = result[0] if isinstance(result, tuple) else result
        output_image = Image.fromarray(output_np)

        # --- Post-processing for style ---
        if model_type == "photo":
            # Subtle enhancements for human realism and beauty
            output_image = ImageEnhance.Color(output_image).enhance(1.05)
            output_image = ImageEnhance.Sharpness(output_image).enhance(0.9)
            output_image = ImageEnhance.Contrast(output_image).enhance(1.02)
        elif model_type == "anime":
            # Sharper edges and richer colors for anime
            output_image = ImageEnhance.Contrast(output_image).enhance(1.15)
            output_image = ImageEnhance.Color(output_image).enhance(1.2)
            output_image = ImageEnhance.Sharpness(output_image).enhance(1.1)

        # Return result
        buf = io.BytesIO()
        output_image.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        print(f"✨ {model_type.capitalize()} enhancement completed successfully at x{scale} scale.")
        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        print("❌ Enhancement error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
