#!/usr/bin/env python3
"""Hosted delivery validation API for the JSONL checker.

This wraps validation/validate_delivery.py in a small async job API so the
GitHub Pages app can run validations without needing local shell access.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
VALIDATOR = ROOT / "validation" / "validate_delivery.py"
JOB_ROOT = Path(os.environ.get("VALIDATION_JOB_ROOT", "/tmp/jsonl_checker_validation_jobs"))
MAX_WORKERS = int(os.environ.get("VALIDATION_API_MAX_JOBS", "2"))
DEFAULT_VALIDATION_WORKERS = int(os.environ.get("VALIDATION_WORKERS", "10"))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "VALIDATION_ALLOWED_ORIGINS",
        "https://mila-avag.github.io,http://localhost:8000,http://127.0.0.1:8000,null",
    ).split(",")
    if origin.strip()
]


app = FastAPI(title="JSONL Checker Delivery Validation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = Lock()


class ValidationRequest(BaseModel):
    fileName: str = Field(default="delivery.jsonl")
    content: str
    workers: int = Field(default=DEFAULT_VALIDATION_WORKERS, ge=1, le=32)


class UploadInitRequest(BaseModel):
    fileName: str = Field(default="delivery.jsonl")
    workers: int = Field(default=DEFAULT_VALIDATION_WORKERS, ge=1, le=32)
    totalChunks: int = Field(ge=1, le=200)
    encoding: str = Field(default="gzip")


def safe_stem(name: str) -> str:
    stem = Path(name or "delivery").stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:120] or "delivery"


def summarize_report(report_path: Path, total_tasks: int | None) -> dict[str, Any] | None:
    if not report_path.exists():
        return None
    issues = json.loads(report_path.read_text())
    affected = {issue.get("task_id") for issue in issues if issue.get("task_id")}
    buckets = Counter(issue.get("pair") or issue.get("category") or "Unknown" for issue in issues)
    type_buckets = Counter(
        f"{issue.get('pair') or issue.get('category')}/{issue.get('cat') or issue.get('type')}"
        for issue in issues
    )
    total = total_tasks if total_tasks is not None else 0
    return {
        "total": total,
        "passing": max(total - len(affected), 0) if total else 0,
        "affected_tasks": len(affected),
        "total_issues": len(issues),
        "issue_buckets": dict(buckets),
        "issue_type_buckets": dict(type_buckets),
    }


def count_jsonl_tasks(path: Path) -> int:
    count = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def update_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        jobs[job_id].update(updates)


def enqueue_validation(file_name: str, content: str, workers: int) -> dict[str, Any]:
    if not VALIDATOR.exists():
        raise HTTPException(status_code=500, detail=f"Validator not found: {VALIDATOR}")
    if not content.strip():
        raise HTTPException(status_code=400, detail="JSONL content is empty")

    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    stem = safe_stem(file_name)
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    delivery_path = job_dir / f"{stem}.jsonl"
    report_path = job_dir / f"validation_report_{stem}.json"
    work_dir = job_dir / "work"
    delivery_path.write_text(content, encoding="utf-8")

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "file_name": file_name,
            "delivery_path": str(delivery_path),
            "work_dir": str(work_dir),
            "report_path": str(report_path),
            "workers": workers,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "stdout_tail": "",
            "stderr_tail": "",
            "summary": None,
        }

    executor.submit(run_validation_job, job_id)
    return {"job_id": job_id, "status": "queued"}


def upload_dir(upload_id: str) -> Path:
    return JOB_ROOT / "uploads" / upload_id


def run_validation_job(job_id: str) -> None:
    with jobs_lock:
        job = jobs[job_id]
        delivery_path = Path(job["delivery_path"])
        work_dir = Path(job["work_dir"])
        report_path = Path(job["report_path"])
        workers = int(job["workers"])

    update_job(job_id, status="running", started_at=time.time())
    total_tasks = count_jsonl_tasks(delivery_path)

    cmd = [
        sys.executable,
        str(VALIDATOR),
        "--delivery",
        str(delivery_path),
        "--work-dir",
        str(work_dir),
        "--output",
        str(report_path),
        "--workers",
        str(workers),
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=int(os.environ.get("VALIDATION_TIMEOUT_SECONDS", "3600")),
        )
        summary = summarize_report(report_path, total_tasks)
        update_job(
            job_id,
            status="done",
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            stdout_tail=proc.stdout[-4000:],
            stderr_tail=proc.stderr[-4000:],
            summary=summary,
            finished_at=time.time(),
        )
    except subprocess.TimeoutExpired as exc:
        update_job(
            job_id,
            status="error",
            exit_code=None,
            stdout=exc.stdout or "",
            stderr=f"Validation timed out after {exc.timeout} seconds",
            stdout_tail=(exc.stdout or "")[-4000:],
            stderr_tail=f"Validation timed out after {exc.timeout} seconds",
            summary=None,
            finished_at=time.time(),
        )
    except Exception as exc:
        update_job(
            job_id,
            status="error",
            exit_code=None,
            stdout="",
            stderr=str(exc),
            stdout_tail="",
            stderr_tail=str(exc),
            summary=None,
            finished_at=time.time(),
        )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    deps = {name: shutil.which(name) for name in ("curl", "unzip")}
    missing = [name for name, path in deps.items() if not path]
    return {
        "status": "ok" if not missing else "missing_dependencies",
        "dependencies": deps,
        "missing": missing,
        "validator_exists": VALIDATOR.exists(),
    }


@app.post("/api/validations")
def create_validation(req: ValidationRequest) -> dict[str, Any]:
    return enqueue_validation(req.fileName, req.content, req.workers)


@app.post("/api/validations/upload")
async def create_validation_upload(
    request: Request,
    x_file_name: str = Header(default="delivery.jsonl"),
    x_workers: int = Header(default=DEFAULT_VALIDATION_WORKERS),
    content_encoding: str | None = Header(default=None),
) -> dict[str, Any]:
    body = await request.body()
    if content_encoding == "gzip" or request.headers.get("content-type") == "application/gzip":
        try:
            body = gzip.decompress(body)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid gzip payload: {exc}") from exc
    try:
        content = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Payload is not UTF-8: {exc}") from exc
    file_name = urllib.parse.unquote(x_file_name or "delivery.jsonl")
    workers = max(1, min(int(x_workers or DEFAULT_VALIDATION_WORKERS), 32))
    return enqueue_validation(file_name, content, workers)


@app.post("/api/validations/chunked/init")
def init_chunked_validation(req: UploadInitRequest) -> dict[str, Any]:
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex
    directory = upload_dir(upload_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "file_name": req.fileName,
                "workers": max(1, min(req.workers, 32)),
                "total_chunks": req.totalChunks,
                "encoding": req.encoding,
                "created_at": time.time(),
            }
        ),
        encoding="utf-8",
    )
    return {"upload_id": upload_id, "total_chunks": req.totalChunks}


@app.post("/api/validations/chunked/{upload_id}/{chunk_index:int}")
async def upload_validation_chunk(upload_id: str, chunk_index: int, request: Request) -> dict[str, Any]:
    directory = upload_dir(upload_id)
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Upload not found")
    meta = json.loads(meta_path.read_text())
    total_chunks = int(meta["total_chunks"])
    if chunk_index < 0 or chunk_index >= total_chunks:
        raise HTTPException(status_code=400, detail="Chunk index out of range")
    body = await request.body()
    (directory / f"chunk_{chunk_index:05d}.bin").write_bytes(body)
    return {"upload_id": upload_id, "chunk_index": chunk_index, "received_bytes": len(body)}


@app.post("/api/validations/chunked/{upload_id}/complete")
def complete_chunked_validation(upload_id: str) -> dict[str, Any]:
    directory = upload_dir(upload_id)
    meta_path = directory / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Upload not found")
    meta = json.loads(meta_path.read_text())
    total_chunks = int(meta["total_chunks"])
    chunks = []
    for idx in range(total_chunks):
        chunk_path = directory / f"chunk_{idx:05d}.bin"
        if not chunk_path.exists():
            raise HTTPException(status_code=400, detail=f"Missing chunk {idx}")
        chunks.append(chunk_path.read_bytes())
    payload = b"".join(chunks)
    if meta.get("encoding") == "gzip":
        try:
            payload = gzip.decompress(payload)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid gzip payload: {exc}") from exc
    try:
        content = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Payload is not UTF-8: {exc}") from exc
    result = enqueue_validation(meta["file_name"], content, int(meta["workers"]))
    shutil.rmtree(directory, ignore_errors=True)
    return result


@app.get("/api/validations/{job_id}")
def get_validation(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Validation job not found")
        return dict(job)


@app.get("/api/validations/{job_id}/report")
def get_validation_report(job_id: str) -> FileResponse:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Validation job not found")
        report_path = Path(job["report_path"])
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report is not ready")
    return FileResponse(report_path, media_type="application/json", filename=report_path.name)
