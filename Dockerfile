# Use lightweight Python image
FROM python:3.11-slim

# Prevent prompts and set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install only essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 \
    build-essential gfortran libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip and core build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first (safe version)
RUN pip install --no-cache-dir numpy==1.26.4

# Install PyTorch CPU version (lightweight)
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install key dependencies (Real-ESRGAN + support libs)
RUN pip install --no-cache-dir basicsr==1.4.2 gfpgan==1.3.8 realesrgan

# Install the rest
RUN pip install --no-cache-dir -r requirements.txt

# Reduce memory load by disabling parallel compilation
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Expose Render port
EXPOSE 10000

# Start FastAPI (optimized for small containers)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
