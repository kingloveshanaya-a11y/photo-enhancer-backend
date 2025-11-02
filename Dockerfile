# 🐍 Use lightweight Python base image
FROM python:3.11-slim

# Avoid interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 🧩 Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 curl wget \
    build-essential gfortran libopenblas-dev liblapack-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# 🧰 Upgrade pip & build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ⚙️ Install numpy first to prevent BLAS issues
RUN pip install --no-cache-dir numpy==1.26.4

# 🧠 Install CPU-only PyTorch and torchvision
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 🧩 Install Real-ESRGAN and its dependencies
RUN pip install --no-cache-dir basicsr==1.4.2 gfpgan==1.3.8 realesrgan==0.3.0

# 📦 Install any remaining project dependencies
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# 📥 Pre-download Real-ESRGAN model weights for both photo + anime
RUN mkdir -p weights && \
    wget -O weights/realesr-general-x4v3.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth && \
    wget -O weights/realesr-animevideov3.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth

# 🧠 Optimize for low-memory (Render Free Tier)
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# 🔥 Expose Uvicorn port
EXPOSE 10000

# 🚀 Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
