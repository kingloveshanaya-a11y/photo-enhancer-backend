# Use Python 3.11 (stable for Real-ESRGAN)
FROM python:3.11-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install Real-ESRGAN dependencies first
RUN pip install basicsr==1.4.2 gfpgan==1.3.8

# Install all other dependencies
RUN pip install -r requirements.txt

# Optional: latest Real-ESRGAN
RUN pip install git+https://github.com/ai-forever/Real-ESRGAN.git

# Expose the Render port
EXPOSE 10000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
