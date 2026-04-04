# syntax=docker/dockerfile:1
# ── Stage 1: Build React frontend ─────────────────────────────────────────
# Build context: repo root (/workspace/storage/NeuriveAI)
FROM node:20-alpine AS frontend-build

WORKDIR /frontend
COPY web/frontend/package.json ./

# Cache mount keeps the npm registry cache across builds.
# First build downloads everything; subsequent builds reuse the cache.
RUN --mount=type=cache,target=/root/.npm \
    npm install

COPY web/frontend/ ./
RUN npm run build
# Output: /frontend/dist/


# ── Stage 2: Python web backend (slim — no GPU, no ML deps) ───────────────
FROM python:3.10-slim

WORKDIR /app

COPY web/backend/requirements-web.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-web.txt

COPY web/backend/main.py ./web/backend/main.py

# Copy built frontend from Stage 1
COPY --from=frontend-build /frontend/dist ./web/backend/static/

RUN mkdir -p /app/web/uploads

EXPOSE 8000

# MODEL_SERVER_URL: host.containers.internal resolves to the Podman host on Linux.
# Override with the actual host IP if needed.
ENV MODEL_SERVER_URL=http://host.containers.internal:8001
ENV UPLOADS_DIR=/app/web/uploads
# PIPELINE_DIR used only for demo video path resolution
ENV PIPELINE_DIR=/app/pipeline

CMD ["python", "web/backend/main.py"]
