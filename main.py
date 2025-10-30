import subprocess

# ---- Install Real-ESRGAN dynamically (Render-safe) ----
try:
    import realesrgan
except ImportError:
    print("💖 Installing Real-ESRGAN (Render-safe fork)... please wait 💖")
    subprocess.run(
        [
            "pip",
            "install",
            "git+https://github.com/ai-forever/Real-ESRGAN.git@master"
        ],
        check=False  # ✅ Do NOT crash the app if pip fails
    )

    # Try again after install, fallback if needed
    try:
        import realesrgan
    except ImportError:
        print("⚠️ Fallback: installing basicsr + facexlib manually...")
        subprocess.run(
            ["pip", "install", "basicsr==1.4.2", "facexlib==0.2.5"],
            check=False
        )

# ---- Now import FastAPI and your routes ----
from fastapi import FastAPI
from routes import photo, video, subtitles, remove_bg, crop

# Create FastAPI app
app = FastAPI(title="AI Enhancer App")

# Include route modules
app.include_router(photo.router, prefix="/photo", tags=["Photo"])
app.include_router(video.router, prefix="/video", tags=["Video"])
app.include_router(subtitles.router, prefix="/subtitles", tags=["Subtitles"])
app.include_router(remove_bg.router, prefix="/remove-bg", tags=["Background Removal"])
app.include_router(crop.router, prefix="/crop", tags=["Crop"])

# Root route
@app.get("/")
async def root():
    return {"message": "AI Enhancer Backend is running 💖"}

# Health check route
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
