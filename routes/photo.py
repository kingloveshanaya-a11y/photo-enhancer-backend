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
MODEL_FILENAME = "RealESRGAN_x4plus.pth"
MODEL_PATH = os.path.join(WEIGHTS_DIR, MODEL_FILENAME)

# === GLOBAL MODEL INSTANCE ===
upsampler: RealESRGANer | None = None

def init_model():
    """Initialize the RealESRGAN model."""
    global upsampler
    if upsampler is not None:
        return upsampler

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}. Please download them first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device == "cuda"

    print(f"🧠 Initializing RealESRGAN model on device: {device}")

    # Correct architecture for RealESRGAN_x4plus.pth
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,   # Must be 23 blocks
        num_grow_ch=32, # Must be 32 growth channels
        scale=4,
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


# Optional: Initialize model at startup to avoid delay on first request
@app.on_event("startup")
def startup_event():
    global upsampler
    try:
        upsampler = init_model()
    except Exception as e:
        print("❌ Failed to initialize RealESRGAN model at startup:")
        traceback.print_exc()


@router.post("/enhance")
async def enhance_photo(
    file: UploadFile = File(...),
    denoise: float = Query(
        0.5, ge=0.0, le=1.0,
        description="Denoise strength for v3 model (0=weak, 1=strong)"
    )
):
    """
    Enhance a photo using RealESRGAN with optional denoise strength.
    Returns the enhanced image in JPEG format.
    """
    try:
        # Read image from upload
        img_bytes = await file.read()
        input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_np = np.array(input_image)

        # Ensure model is initialized
        global upsampler
        if upsampler is None:
            upsampler = init_model()

        # Perform enhancement
        result = upsampler.enhance(input_np, outscale=4, denoise=denoise)
        output_np = result[0] if isinstance(result, tuple) else result

        # Convert back to PIL Image
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
