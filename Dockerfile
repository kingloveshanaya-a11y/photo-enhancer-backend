# Use Python 3.11 (stable for Real-ESRGAN)
FROM python:3.11-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies + build tools for numpy/scipy
RUN apt-get update && apt-get install -y \
    git ffmpeg libgl1 libglib2.0-0 \
    build-essential gfortran libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip and install wheel/setuptools
RUN pip install --upgrade pip setuptools wheel

# Install numpy first (needed by basicsr and torch)
RUN pip install numpy==2.0.1

# Install compatible PyTorch + torchvision (CPU version)
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install Real-ESRGAN dependencies
RUN pip install basicsr==1.4.2 gfpgan==1.3.8

# Install other dependencies from requirements.txt
RUN pip install -r requirements.txt

# Install Real-ESRGAN (GitHub)
RUN pip install git+https://github.com/ai-forever/Real-ESRGAN.git

# Expose Render’s dynamic port (Render sets $PORT)
EXPOSE 10000

# Default command for Render
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
