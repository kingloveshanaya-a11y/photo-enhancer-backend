from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import io
import os
import numpy as np
import requests
import threading
import traceback
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

router = APIRouter()

# === CONFIG ===
WEIGHTS_DIR = "weights"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4.pth")
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4.pth"

# === GLOBALS ===
upsampler = None
download_thread = None


def download_weights():
    """Download model weights if missing."""
    if os.path.exists(MODEL_PATH):
        print("✅ RealESRGAN weights already present.")
        return
    print("💖 Downloading RealESRGAN weights... Please wait 💖")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    try:
        with requests.get(MODEL_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ RealESRGAN weights downloaded successfully!")
    except Exception as e:
        print(f"⚠️ Failed to download weights: {e}")


def init_model():
    """Initialize the RealESRGAN model."""
    global upsampler
    if upsampler is not None:
        return upsampler

    if not os.path.exists(MODEL_PATH):
        download_weights()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = (device == "cuda")

    print(f"🧠 Initializing RealESRGAN model on device: {device}")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=MODEL_PATH,
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=device,
    )

    print("🚀 RealESRGAN model initialized successfully.")
    return upsampler


# === Lazy async init ===
def background_init():
    """Non-blocking background model preload."""
    try:
        init_model()
    except Exception as e:
        print(f"⚠️ Background model init failed: {e}")


# Start preloading in background immediately at import
download_thread = threading.Thread(target=background_init, daemon=True)
download_thread.start()


@router.post("/enhance")
async def enhance_photo(file: UploadFile = File(...)):
    """Enhance a photo using RealESRGAN."""
    try:
        img_bytes = await file.read()
        input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_np = np.array(input_image)

        # Ensure model is ready
        global upsampler
        if upsampler is None:
            upsampler = init_model()

        # Perform enhancement
        result = upsampler.enhance(input_np, outscale=4)

        # Handle RealESRGAN's return type (2-tuple or 3-tuple)
        if isinstance(result, tuple):
            output_np = result[0] if len(result) == 2 else result[1]
        else:
            output_np = result

        # Convert numpy → image → bytes
        output_image = Image.fromarray(output_np)
        buf = io.BytesIO()
        output_image.save(buf, format="JPEG")
        buf.seek(0)

        print("✨ Image enhancement successful!")
        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        print("❌ Error during image enhancement:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
