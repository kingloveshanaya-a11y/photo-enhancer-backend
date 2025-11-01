# 🐍 Use lightweight Python image
FROM python:3.11-slim

# Avoid interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 🧩 Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 curl wget \
    build-essential gfortran libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# 🧰 Upgrade pip and core build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ⚙️ Install numpy first to avoid conflicts
RUN pip install --no-cache-dir numpy==1.26.4

# 🧠 Install PyTorch CPU-only version
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# 🧩 Install Real-ESRGAN + supporting libs
RUN pip install --no-cache-dir basicsr==1.4.2 gfpgan==1.3.8 realesrgan==0.3.0

# 📦 Install remaining dependencies if requirements.txt exists
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# 📥 Pre-download Real-ESRGAN model weights (for CPU version)
RUN mkdir -p weights && \
    wget -O weights/realesr-general-x4v3.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth

# 🧠 Optimize for low memory environments (like Render free tier)
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# 🔥 Expose port for Uvicorn
EXPOSE 10000

# 🚀 Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
