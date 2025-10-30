import os
import subprocess
import importlib.util

# --- Dynamic dependency installer for Real-ESRGAN ---
def ensure_realesrgan_installed():
    """Dynamically install Real-ESRGAN if not already present."""
    try:
        import realesrgan
        print("✅ Real-ESRGAN already installed")
    except ImportError:
        print("💖 Installing Real-ESRGAN dynamically... please wait 💖")
        subprocess.run(["pip", "install", "basicsr==1.4.2", "gfpgan==1.3.8"], check=True)
        subprocess.run([
            "pip", "install",
            "git+https://github.com/xinntao/Real-ESRGAN.git@fa4c8a03ae3dbc9ea6ed471a6ab5da94ac15c2ea"
        ], check=True)
        print("✅ Real-ESRGAN installed successfully!")

# Ensure it’s available before routes import it
ensure_realesrgan_installed()

# --- FastAPI setup ---
from fastapi import FastAPI
from routes import photo, video, subtitles, remove_bg, crop

app = FastAPI(title="AI Enhancer App")

# Include routes
app.include_router(photo.router, prefix="/photo", tags=["Photo"])
app.include_router(video.router, prefix="/video", tags=["Video"])
app.include_router(subtitles.router, prefix="/subtitles", tags=["Subtitles"])
app.include_router(remove_bg.router, prefix="/remove-bg", tags=["Background Removal"])
app.include_router(crop.router, prefix="/crop", tags=["Crop"])

@app.get("/")
async def root():
    return {"message": "AI Enhancer Backend is running 💖"}

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
