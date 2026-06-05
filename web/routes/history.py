import logging
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _project_root(request: Request) -> Path:
    return request.app.state.project_root


# ── List ──────────────────────────────────────────────────────


@router.get("/history")
async def list_history(request: Request, task: str = None):
    job_store = request.app.state.job_store
    jobs = await job_store.list_jobs()
    if task:
        jobs = [j for j in jobs if j.get("task_name") == task]
    jobs.sort(key=lambda j: j.get("start_time", 0), reverse=True)
    return jobs


# ── Detail ────────────────────────────────────────────────────


@router.get("/history/{job_id}")
async def get_history(job_id: str, request: Request):
    job_store = request.app.state.job_store
    meta = await job_store.read_metadata(job_id)
    if meta is None:
        raise HTTPException(404, f"Job '{job_id}' not found")

    log_text = await job_store.get_log(job_id)
    return {
        "metadata": meta,
        "log": log_text,
    }


# ── Delete ────────────────────────────────────────────────────


@router.delete("/history/{job_id}")
async def delete_history(job_id: str, request: Request):
    job_store = request.app.state.job_store
    ok = await job_store.delete_job(job_id)
    if not ok:
        raise HTTPException(404, f"Job '{job_id}' not found")
    logger.info("Job '%s' history deleted", job_id)
    return {"detail": "ok"}
