# ── Stage 1: Build React frontend ─────────────────────────────────────────
# Build context: repo root (/workspace/storage/NeuriveAI)
FROM node:20-alpine AS frontend-build

WORKDIR /frontend
COPY web/frontend/package.json ./
RUN npm install

COPY web/frontend/ ./
RUN npm run build
# Output lands at /frontend/dist  (vite.config outDir is ../backend/static,
# but inside this stage that resolves to /frontend/dist — we copy it explicitly below)


# ── Stage 2: Python runtime with CUDA ─────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Layer 1: Pipeline dependencies (heavy — cached unless requirements.txt changes)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Layer 2: Web dependencies (fast)
COPY web/backend/requirements-web.txt /app/web/backend/requirements-web.txt
RUN pip install --no-cache-dir -r /app/web/backend/requirements-web.txt

# Layer 3: Application source
COPY *.py /app/
COPY web/backend/main.py /app/web/backend/main.py

# Layer 4: Built frontend from Stage 1
RUN mkdir -p /app/web/backend/static
COPY --from=frontend-build /frontend/dist /app/web/backend/static/

# Uploads directory
RUN mkdir -p /app/web/uploads

EXPOSE 8000

ENV PIPELINE_DIR=/app
ENV UPLOADS_DIR=/app/web/uploads
ENV CUDA_VISIBLE_DEVICES=2

CMD ["python", "/app/web/backend/main.py"]
