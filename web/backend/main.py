"""
NeuriveAI Web Backend — FastAPI proxy layer (no GPU, no pipeline).
Serves the React frontend and proxies pipeline requests to the model server.

Model server must be running on the host at MODEL_SERVER_URL (default :8001).
"""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import aiofiles
import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Paths & config ─────────────────────────────────────────────────────────
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", str(Path(__file__).parent / "uploads")))
STATIC_DIR  = Path(__file__).parent / "static"
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001")

# Demo videos live in the pipeline repo root, bind-mounted at /app/pipeline in Docker
PIPELINE_MOUNT = Path(os.getenv("PIPELINE_DIR", str(Path(__file__).parent.parent.parent.resolve())))
DEMO_STEM = "grok-video-4fa879d0-7052-400c-9aed-efb888d9579e"

# HUD overlay shown on impact freeze — off by default until polished.
# Set env var SHOW_IMPACT_HUD=true or flip this constant to re-enable.
SHOW_IMPACT_HUD: bool = os.getenv("SHOW_IMPACT_HUD", "false").lower() == "true"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="NeuriveAI Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Video Range streaming helper ───────────────────────────────────────────
def _video_stream_response(video_path: Path, request: Request) -> StreamingResponse:
    """Serve a video file with HTTP Range support (required for browser seeking)."""
    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        byte1, byte2 = 0, file_size - 1
        m = re.search(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            byte1 = int(m.group(1))
            if m.group(2):
                byte2 = int(m.group(2))

        length = byte2 - byte1 + 1

        def iterfile():
            with open(video_path, "rb") as f:
                f.seek(byte1)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            "Content-Range": f"bytes {byte1}-{byte2}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iterfile(), status_code=206, headers=headers)
    else:
        return FileResponse(
            str(video_path),
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes"},
        )


def _resolve_video_path(job_id: str, stem_suffix: str) -> Path:
    """
    Resolve the on-disk path for a video file.
    - demo job: videos live in PIPELINE_MOUNT (bind-mounted repo root)
    - real jobs: videos live in UPLOADS_DIR/{job_id}/
    """
    if job_id == "demo":
        name = f"{DEMO_STEM}{stem_suffix}.mp4"
        return PIPELINE_MOUNT / name
    else:
        name = f"original{stem_suffix}.mp4"
        return UPLOADS_DIR / job_id / name


# ── API: Config ───────────────────────────────────────────────────────────
@app.get("/api/config")
def get_config():
    return {"showImpactHud": SHOW_IMPACT_HUD}


# ── API: Upload ────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True)

    dest = job_dir / "original.mp4"
    async with aiofiles.open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            await f.write(chunk)

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{MODEL_SERVER_URL}/process",
                json={"job_id": job_id},
                timeout=10,
            )
            r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Model server error: {e}")

    return {"job_id": job_id, "filename": file.filename}


# ── API: SSE Status (proxied from model server) ────────────────────────────
@app.get("/api/status/{job_id}")
async def proxy_status(job_id: str):
    async def stream():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
                async with client.stream(
                    "GET", f"{MODEL_SERVER_URL}/status/{job_id}"
                ) as resp:
                    async for chunk in resp.aiter_text():
                        yield chunk
        except httpx.HTTPError as e:
            import json as _json
            yield f"data: {_json.dumps({'stage': 'error', 'progress': 0, 'message': str(e)})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── API: Results (proxied) ─────────────────────────────────────────────────
@app.get("/api/results/{job_id}")
async def proxy_results(job_id: str):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{MODEL_SERVER_URL}/results/{job_id}", timeout=10)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Model server unreachable: {e}")


# ── API: Demo — run pipeline (proxied) ────────────────────────────────────
@app.post("/api/demo/run")
async def proxy_demo_run():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{MODEL_SERVER_URL}/demo/run", timeout=10)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Model server unreachable: {e}")


# ── API: Video streaming (served directly from bind-mounted filesystem) ────
@app.get("/api/video/{job_id}/annotated")
async def video_annotated(job_id: str, request: Request):
    path = _resolve_video_path(job_id, "_annotated")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Annotated video not found: {path.name}")
    return _video_stream_response(path, request)


@app.get("/api/video/{job_id}/original")
async def video_original(job_id: str, request: Request):
    path = _resolve_video_path(job_id, "")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {path.name}")
    return _video_stream_response(path, request)


# ── Serve React SPA ────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse(
        "<h1>Frontend not built. Run: podman compose -f web/compose.yaml up --build</h1>"
    )

_assets_dir = STATIC_DIR / "assets"
if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404)
    # Serve static files from the dist directory (e.g. brain_regions.json, favicons)
    static_file = STATIC_DIR / full_path
    if static_file.exists() and static_file.is_file():
        return FileResponse(str(static_file))
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    raise HTTPException(status_code=404)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"[Web Backend] Starting on :{port}  MODEL_SERVER_URL={MODEL_SERVER_URL}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
