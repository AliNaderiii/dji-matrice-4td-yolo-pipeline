FROM python:3.10-slim

# Author: Ali Naderi | DJI Matrice 4TD Pipeline
# Fixed for Debian trixie + pip hash issue

WORKDIR /app

# System deps for OpenCV and ONNX - Debian trixie compatible
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first - fixes hash mismatch for torch 2.5.1+cu121
RUN pip install --upgrade pip

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY configs/ configs/
COPY src/ src/
COPY scripts/ scripts/
COPY notebooks/ notebooks/
COPY docs/ docs/
COPY demo/ demo/
COPY README.md .
COPY .gitignore .

# Create dirs
RUN mkdir -p datasets runs dji_submission demo

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["bash"]

# Usage:
# docker build -t dji-matrice-pipeline -f docker/Dockerfile .
# docker run -it --gpus all -v $(pwd)/datasets:/app/datasets -v $(pwd)/runs:/app/runs dji-matrice-pipeline
# For MQTT broker only (no need to build app):
# docker compose -f docker/docker-compose.yml up -d emqx
