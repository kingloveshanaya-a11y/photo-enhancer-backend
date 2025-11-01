from fastapi import FastAPI
from routes import photo, video, subtitles, remove_bg, crop

app = FastAPI(title="AI Enhancer App")

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
