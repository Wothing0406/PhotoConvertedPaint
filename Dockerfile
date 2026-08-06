# Use NVIDIA CUDA 12.1 runtime on Ubuntu 22.04 as base image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid tzdata interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV U2NET_HOME=/root/.u2net

# Install Python 3.10 and system dependencies for OpenCV/CUDA
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python/pip command
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
# Install packages (pip will install cuda-enabled onnxruntime-gpu successfully on this image)
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/src /app/output /app/test_images /root/.u2net

# Copy the rest of the application code
COPY src/ /app/src/
COPY static/ /app/static/
COPY app.py /app/

# Expose port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
