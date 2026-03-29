# ── Base: CUDA 12.1 + Python 3.10 ────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System deps ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3     /usr/bin/python

# ── Python deps ───────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r /tmp/requirements.txt

# ── App ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY helmet_tracker.py  .
COPY hot_detector.py    .
COPY iou_detector.py    .
COPY velocity_detector.py .
COPY impact_detector.py .
COPY track_video.py     .

# models/ and video files are mounted at runtime (see docker-compose.yml)

CMD ["python", "track_video.py"]
