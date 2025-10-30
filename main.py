import os
from fastapi import FastAPI
from routes import photo, video, subtitles, remove_bg, crop

# 🪄 Force-install Real-ESRGAN + BasicSR at runtime (for Render free tier)
# This avoids build errors during deploy.
try:
    os.system("pip install git+https://github.com/xinntao/BasicSR.git@master --no-cache-dir || true")
    os.system("pip install git+https://github.com/xinntao/Real-ESRGAN.git@fa4c8a03ae3dbc9ea6ed471a6ab5da94ac15c2ea --no-cache-dir || true")
except Exception as e:
    print("⚠️ Warning: Failed to install Real-ESRGAN packages:", e)

app = FastAPI(title="AI Enhancer App")

# Include route modules
app.include_router(photo.router, prefix="/photo", tags=["Photo"])
app.include_router(video.router, prefix="/video", tags=["Video"])
app.include_router(subtitles.router, prefix="/subtitles", tags=["Subtitles"])
app.include_router(remove_bg.router, prefix="/remove-bg", tags=["Background Removal"])
app.include_router(crop.router, prefix="/crop", tags=["Crop"])

@app.get("/")
async def root():
    return {"message": "AI Enhancer Backend is running 💖"}

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}
