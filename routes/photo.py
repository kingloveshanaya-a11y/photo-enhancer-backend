from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import io
import os
import numpy as np
import traceback
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
upsamplers = {
    "photo": None,
    "anime": None
}

def ensure_model_exists(model_name: str):
    """Ensure that a specific model file exists locally."""
    model_path = os.path.join(WEIGHTS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. "
            f"Please download '{model_name}' from the official Real-ESRGAN release "
            f"and place it in the '{WEIGHTS_DIR}' folder."
        )
    return model_path

def init_model(model_type: str):
    """Initialize a specific RealESRGAN model."""
    global upsamplers

    if upsamplers.get(model_type) is not None:
        return upsamplers[model_type]

    model_filename = MODELS.get(model_type)
    if not model_filename:
        raise ValueError(f"Invalid model type '{model_type}'. Use 'photo' or 'anime'.")

    model_path = ensure_model_exists(model_filename)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device == "cuda"

    print(f"🧠 Initializing RealESRGAN '{model_type}' model on device: {device}")

    # Define network architecture (same for both)
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
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
    """Initialize both models at app startup (optional)."""
    for model_type in MODELS.keys():
        try:
            init_model(model_type)
        except Exception as e:
            print(f"⚠️ Failed to load {model_type} model at startup: {e}")

# === MAIN ENDPOINT ===
@router.post("/enhance")
async def enhance_photo(
    file: UploadFile = File(...),
    type: str = Query("photo", description="Type of image: 'photo' (default) or 'anime'"),
    denoise: float = Query(
        0.5, ge=0.0, le=1.0,
        description="Denoise strength for v3 model (0=weak, 1=strong)"
    )
):
    """Enhance a photo or anime image using RealESRGAN."""
    try:
        # Read uploaded image
        img_bytes = await file.read()
        input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_np = np.array(input_image)

        # Choose model
        model_type = type.lower().strip()
        if model_type not in MODELS:
            raise HTTPException(status_code=400, detail="Invalid type. Use 'photo' or 'anime'.")

        upsampler = upsamplers.get(model_type) or init_model(model_type)

        # Enhance image
        result = upsampler.enhance(input_np, outscale=4, denoise=denoise)
        output_np = result[0] if isinstance(result, tuple) else result

        # Convert to JPEG
        output_image = Image.fromarray(output_np)
        buf = io.BytesIO()
        output_image.save(buf, format="JPEG")
        buf.seek(0)

        print(f"✨ {model_type.capitalize()} image enhancement successful!")
        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        print("❌ Error during image enhancement:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
