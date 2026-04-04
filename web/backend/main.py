"""
NeuriveAI Web Backend — FastAPI app
Serves the React frontend and provides the pipeline API.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

import aiofiles
import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ──────────────────────────────────────────────────────────────────
# Allow override via env var for container deployments (PIPELINE_DIR=/app)
PIPELINE_DIR = Path(
    os.getenv("PIPELINE_DIR", str(Path(__file__).parent.parent.parent.resolve()))
)
UPLOADS_DIR = Path(
    os.getenv("UPLOADS_DIR", str(Path(__file__).parent / "uploads"))
)
STATIC_DIR = Path(__file__).parent / "static"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Demo artifacts ─────────────────────────────────────────────────────────
DEMO_STEM = "grok-video-4fa879d0-7052-400c-9aed-efb888d9579e"

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="NeuriveAI Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Job store ──────────────────────────────────────────────────────────────
# {job_id: {stage, progress, log, job_dir, video_stem, error}}
jobs: dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=2)


# ── Pipeline runner ────────────────────────────────────────────────────────
def _run_pipeline(job_id: str, video_path: Path) -> None:
    """Blocking — runs in ThreadPoolExecutor."""
    job = jobs[job_id]

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "2"}

    cmd = [
        sys.executable,
        str(PIPELINE_DIR / "track_video.py"),
        "--video", str(video_path),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PIPELINE_DIR),
            env=env,
        )

        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            job["log"].append(line)

            if "[Pass 2]" in line:
                job["stage"] = "pass2"
                job["progress"] = 55
            elif "[Render]" in line:
                job["stage"] = "rendering"
                job["progress"] = 80
            elif "Saved to:" in line:
                job["progress"] = 90
            elif "[Done]" in line:
                job["stage"] = "complete"
                job["progress"] = 100
            elif "[Pass 1 complete]" in line:
                job["progress"] = 50
            elif "[IMPACT DETECTED]" in line:
                job["progress"] = min(job["progress"] + 3, 48)
            elif "[Pass 1]" in line:
                job["stage"] = "pass1"
                job["progress"] = 10

        proc.wait()

        if proc.returncode != 0:
            job["stage"] = "error"
            job["error"] = f"Pipeline exited with code {proc.returncode}"
        else:
            job["stage"] = "complete"
            job["progress"] = 100

    except Exception as exc:
        job["stage"] = "error"
        job["error"] = str(exc)
        job["log"].append(f"[ERROR] {exc}")


# ── Video range streaming ──────────────────────────────────────────────────
def _video_stream_response(video_path: Path, request: Request) -> StreamingResponse:
    """Stream a video file with HTTP Range request support for browser seeking."""
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


def _get_video_meta(path: Path) -> tuple[float, int]:
    """Read FPS and frame count from a video file."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


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

    jobs[job_id] = {
        "stage": "pass1",
        "progress": 5,
        "log": [f"Received: {file.filename}"],
        "job_dir": job_dir,
        "video_stem": "original",
        "error": None,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_pipeline, job_id, dest)

    return {"job_id": job_id, "filename": file.filename}


# ── API: SSE Status ────────────────────────────────────────────────────────
@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            job = jobs.get(job_id)
            if not job:
                break

            payload = json.dumps({
                "stage": job["stage"],
                "progress": job["progress"],
                "message": job["log"][-1] if job["log"] else "",
            })
            yield f"data: {payload}\n\n"

            if job["stage"] in ("complete", "error"):
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── API: Results ───────────────────────────────────────────────────────────
@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["stage"] not in ("complete",):
        raise HTTPException(status_code=400, detail="Job not complete")

    job_dir = job["job_dir"]
    stem = job["video_stem"]
    report_path = job_dir / f"{stem}.impact_report.json"
    video_path = job_dir / f"{stem}.mp4"

    if not report_path.exists():
        raise HTTPException(status_code=500, detail="Report file missing")

    async with aiofiles.open(report_path) as f:
        report = json.loads(await f.read())

    fps, total_frames = _get_video_meta(video_path)

    return {"fps": fps, "total_frames": total_frames, "report": report}


# ── API: Video stream ──────────────────────────────────────────────────────
@app.get("/api/video/{job_id}/annotated")
async def video_annotated(job_id: str, request: Request):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    stem = job["video_stem"]
    path = job["job_dir"] / f"{stem}_annotated.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not found")
    return _video_stream_response(path, request)


@app.get("/api/video/{job_id}/original")
async def video_original(job_id: str, request: Request):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    stem = job["video_stem"]
    path = job["job_dir"] / f"{stem}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return _video_stream_response(path, request)


# ── API: Demo mode ─────────────────────────────────────────────────────────
@app.get("/api/demo")
async def load_demo():
    report_path = PIPELINE_DIR / f"{DEMO_STEM}.impact_report.json"
    video_path = PIPELINE_DIR / f"{DEMO_STEM}.mp4"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Demo report not found")

    async with aiofiles.open(report_path) as f:
        report = json.loads(await f.read())

    fps, total_frames = _get_video_meta(video_path)

    jobs["demo"] = {
        "stage": "complete",
        "progress": 100,
        "log": ["Demo mode — loaded pre-existing test artifacts"],
        "job_dir": PIPELINE_DIR,
        "video_stem": DEMO_STEM,
        "error": None,
    }

    return {
        "job_id": "demo",
        "fps": fps,
        "total_frames": total_frames,
        "report": report,
    }


# ── Serve React SPA ────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>Frontend not built yet. Run: cd web/frontend && npm run build</h1>")

# Mount static assets (JS/CSS chunks) — only if the built frontend is present
_assets_dir = STATIC_DIR / "assets"
if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

# Catch-all for React Router (if added later)
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # Don't catch API routes
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404)
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    raise HTTPException(status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
