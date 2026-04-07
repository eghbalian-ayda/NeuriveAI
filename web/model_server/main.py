"""
NeuriveAI Model Server — runs on HOST with GPU access.
Wraps the track_video.py pipeline and exposes a lightweight HTTP API.

Start with:
    CUDA_VISIBLE_DEVICES=2 python web/model_server/main.py
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

import aiofiles
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(
    os.getenv("PIPELINE_DIR", str(Path(__file__).parent.parent.parent.resolve()))
)
UPLOADS_DIR = Path(
    os.getenv("UPLOADS_DIR", str(Path(__file__).parent.parent / "uploads"))
)
DEMO_STEM = "grok-video-4fa879d0-7052-400c-9aed-efb888d9579e"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="NeuriveAI Model Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Job store ──────────────────────────────────────────────────────────────
# {job_id: {stage, progress, log: list[str], job_dir: Path, video_stem: str, error}}
jobs: dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=2)


# ── Pipeline runner ────────────────────────────────────────────────────────
def _run_pipeline(job_id: str) -> None:
    """Blocking — runs in ThreadPoolExecutor."""
    job = jobs[job_id]

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "2"}

    cmd = [
        sys.executable,
        str(PIPELINE_DIR / "track_video.py"),
        "--video", str(job["job_dir"] / f"{job['video_stem']}.mp4"),
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


def _get_video_meta(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


# ── Models ─────────────────────────────────────────────────────────────────
class ProcessRequest(BaseModel):
    job_id: str


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "gpu": os.getenv("CUDA_VISIBLE_DEVICES", "unset")}


@app.post("/process")
async def process_video(req: ProcessRequest):
    job_id = req.job_id
    job_dir = UPLOADS_DIR / job_id
    video_path = job_dir / "original.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    jobs[job_id] = {
        "stage": "pass1",
        "progress": 5,
        "log": [f"Starting pipeline for job {job_id}"],
        "job_dir": job_dir,
        "video_stem": "original",
        "error": None,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_pipeline, job_id)

    return {"job_id": job_id}


@app.get("/status/{job_id}")
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


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["stage"] != "complete":
        raise HTTPException(status_code=400, detail=f"Job not complete (stage: {job['stage']})")

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


@app.post("/demo/run")
async def run_demo():
    """Queue the demo video through the full pipeline (same as a real upload)."""
    video_path = PIPELINE_DIR / f"{DEMO_STEM}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Demo video not found: {video_path}")

    # If already running, return the existing job rather than starting a second pipeline
    existing = jobs.get("demo")
    if existing and existing["stage"] not in ("complete", "error"):
        return {"job_id": "demo"}

    jobs["demo"] = {
        "stage": "pass1",
        "progress": 5,
        "log": [f"Starting demo pipeline for {DEMO_STEM}"],
        "job_dir": PIPELINE_DIR,
        "video_stem": DEMO_STEM,
        "error": None,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_pipeline, "demo")

    return {"job_id": "demo"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MODEL_SERVER_PORT", "8001"))
    print(f"[Model Server] Starting on :{port}  PIPELINE_DIR={PIPELINE_DIR}  UPLOADS_DIR={UPLOADS_DIR}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
