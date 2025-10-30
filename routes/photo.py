from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import io
import os
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import traceback

router = APIRouter()

# Paths
WEIGHTS_DIR = "weights"
MODEL_PATH = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4.pth")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"RealESRGAN weights not found at {MODEL_PATH}.")

device = "cuda" if torch.cuda.is_available() else "cpu"
use_half = True if device == "cuda" else False

# Load model once at startup
model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
)

# Tile-based enhancement for full-resolution
upsampler = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model,
    tile=512,        # smaller tiles reduce GPU memory load
    tile_pad=10,     # overlap to avoid seams
    pre_pad=0,
    half=use_half,
    device=device
)

@router.post("/enhance")
async def enhance_photo(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_np = np.array(input_image)

        # Enhance image using tiles (full resolution)
        result = upsampler.enhance(input_np, outscale=4)

        # Handle both 2-tuple or 3-tuple return
        if isinstance(result, tuple):
            if len(result) == 2:
                output_np, _ = result
            elif len(result) == 3:
                _, output_np, _ = result
            else:
                raise ValueError("Unexpected return from upsampler.enhance()")
        else:
            output_np = result

        output_image = Image.fromarray(output_np)

        buf = io.BytesIO()
        output_image.save(buf, format="JPEG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        print("Error during image enhancement:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
