from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageEnhance
import torch, io, os, numpy as np, traceback, uuid, asyncio
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

router = APIRouter()

# === Configuration ===
WEIGHTS_DIR = "weights"
PROGRESS = {}  # job_id → progress %
RESULTS = {}   # job_id → image buffer or error string
CLEANUP_DELAY = 300  # 5 minutes

MODELS = {
    "photo":  {"file": "realesr-general-x4v3.pth", "nb": 23},
    "anime":  {"file": "realesr-animevideov3.pth", "nb": 6},
    "x4plus": {"file": "RealESRGAN_x4plus.pth",    "nb": 23},
    "x4":     {"file": "RealESRGAN_x4.pth",        "nb": 23},
}

upsamplers = {m: None for m in MODELS}


# === Utility helpers ===
def ensure_model_exists(model_type: str) -> str:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_path = os.path.join(WEIGHTS_DIR, MODELS[model_type]["file"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Missing model file: {model_path}\n"
            f"➡️  Please copy it into your 'weights' folder."
        )
    return model_path


def init_model(model_type: str):
    """Lazily initialize a model if not already loaded."""
    if upsamplers.get(model_type):
        return upsamplers[model_type]

    model_path = ensure_model_exists(model_type)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device == "cuda"
    print(f"🧠 Initializing RealESRGAN '{model_type}' model on {device}...")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=MODELS[model_type]["nb"],
        num_grow_ch=32,
        scale=4
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=device
    )

    upsamplers[model_type] = upsampler
    print(f"✅ Model '{model_type}' initialized successfully.")
    return upsampler


def detect_image_type(image_np: np.ndarray) -> str:
    """Rudimentary photo vs. anime detection."""
    mean_sat = np.mean(image_np.std(axis=2))
    edges = np.mean(np.abs(np.gradient(image_np.mean(axis=2))))
    if mean_sat < 25 and edges > 2.0:
        return "anime"
    return "photo"


async def update_progress(job_id: str, value: int):
    PROGRESS[job_id] = max(0, min(100, value))


async def cleanup_job(job_id: str):
    """Remove job data from memory after CLEANUP_DELAY seconds."""
    await asyncio.sleep(CLEANUP_DELAY)
    PROGRESS.pop(job_id, None)
    RESULTS.pop(job_id, None)
    print(f"🧹 Cleaned up job {job_id}.")


# === Preload models on startup ===
@router.on_event("startup")
def preload_models():
    try:
        init_model("photo")
        init_model("anime")
        print("🚀 Preloaded 'photo' and 'anime' models.")
    except Exception as e:
        print(f"⚠️ Model preload warning: {e}")


# === Core enhancement task ===
async def process_enhancement(job_id: str, image_data: bytes, model: str, scale: int, denoise: float):
    try:
        await update_progress(job_id, 5)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_np = np.array(img)
        await update_progress(job_id, 20)

        # Choose model
        if not model or model not in MODELS:
            model = detect_image_type(input_np)
            print(f"🔍 Auto-detected model: {model}")
        else:
            print(f"🎯 Using user-selected model: {model}")

        upsampler = upsamplers.get(model) or init_model(model)
        await update_progress(job_id, 45)

        # Upscale
        result = upsampler.enhance(input_np, outscale=scale, denoise=denoise)
        output_np = result[0] if isinstance(result, tuple) else result
        await update_progress(job_id, 75)

        # Gentle post-enhancement
        out_img = Image.fromarray(output_np)
        out_img = ImageEnhance.Color(out_img).enhance(1.05)
        out_img = ImageEnhance.Contrast(out_img).enhance(1.02)
        out_img = ImageEnhance.Sharpness(out_img).enhance(0.9)

        buf = io.BytesIO()
        out_img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        RESULTS[job_id] = buf
        await update_progress(job_id, 100)
        print(f"✨ Job {job_id}: done (x{scale}, {model}).")

        asyncio.create_task(cleanup_job(job_id))
    except Exception as e:
        traceback.print_exc()
        PROGRESS[job_id] = -1
        RESULTS[job_id] = str(e)


# === Routes ===
@router.post("/enhance/start")
async def start_enhancement(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    scale: int = Query(4, ge=2, le=8),
    model: str = Query(None, description="Force model: photo | anime | x4plus | x4"),
    denoise: float = Query(0.5, ge=0.0, le=1.0)
):
    """Start enhancement job asynchronously."""
    try:
        job_id = str(uuid.uuid4())
        PROGRESS[job_id] = 0
        RESULTS[job_id] = None
        img_bytes = await file.read()

        background_tasks.add_task(process_enhancement, job_id, img_bytes, model, scale, denoise)
        return JSONResponse({"job_id": job_id, "status": "started"})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enhance/progress/{job_id}")
async def get_progress(job_id: str):
    """Check enhancement progress."""
    progress = PROGRESS.get(job_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if progress == -1:
        return {"job_id": job_id, "status": "error", "error": RESULTS.get(job_id)}
    return {"job_id": job_id, "progress": progress}


@router.get("/enhance/result/{job_id}")
async def get_result(job_id: str):
    """Fetch completed enhanced image."""
    if job_id not in RESULTS:
        raise HTTPException(status_code=404, detail="Job not found")
    if PROGRESS.get(job_id) != 100:
        raise HTTPException(status_code=400, detail="Job not complete")

    buf = RESULTS[job_id]
    if isinstance(buf, str):
        raise HTTPException(status_code=500, detail=buf)
    return StreamingResponse(buf, media_type="image/jpeg")
