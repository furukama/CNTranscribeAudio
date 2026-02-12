from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from app.pipeline import ChineseTranscriptionPipeline


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# Silence noisy Transformers generation advisory warnings (e.g. invalid/ignored flags).
logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)

app = FastAPI(title="CN Transcribe Audio", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

pipeline = ChineseTranscriptionPipeline()


class TranscribeRequest(BaseModel):
    youtube_url: HttpUrl
    translation_backend: Literal["hf", "mlx"] = "hf"


jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _update_job(job_id: str, **kwargs) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)
            jobs[job_id]["updated_at"] = time.time()


def _run_job(job_id: str, youtube_url: str, translation_backend: str) -> None:
    def progress_cb(percent: int, message: str) -> None:
        _update_job(job_id, progress=percent, message=message)

    try:
        result = pipeline.run(
            youtube_url,
            translation_backend=translation_backend,
            progress_cb=progress_cb,
        )
        _update_job(job_id, status="completed", progress=100, message="Complete", result=result)
    except Exception as exc:
        _update_job(job_id, status="failed", message=str(exc), error=f"Transcription failed: {exc}")


@app.post("/api/transcribe/start")
def transcribe_start(payload: TranscribeRequest) -> dict[str, str]:
    job_id = uuid.uuid4().hex
    now = time.time()
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued",
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, str(payload.youtube_url), payload.translation_backend),
        daemon=True,
    )
    thread.start()
    _update_job(job_id, status="running", message="Starting", progress=1)
    return {"job_id": job_id}


@app.get("/api/transcribe/{job_id}")
def transcribe_status(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


@app.post("/api/transcribe")
def transcribe(payload: TranscribeRequest) -> dict:
    try:
        return pipeline.run(
            str(payload.youtube_url),
            translation_backend=payload.translation_backend,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
