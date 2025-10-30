# Use Python 3.11 (compatible with Real-ESRGAN)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install Real-ESRGAN from GitHub (specific commit for stability)
RUN pip install git+https://github.com/xinntao/Real-ESRGAN.git@fa4c8a03ae3dbc9ea6ed471a6ab5da94ac15c2ea

# Copy app code
COPY . .

# Expose port (Render uses this port)
EXPOSE 10000

# Start command for Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
