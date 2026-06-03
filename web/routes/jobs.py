import logging
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _project_root(request: Request) -> Path:
    return request.app.state.project_root


# ── Job detail ────────────────────────────────────────────────


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    scheduler = request.app.state.scheduler
    job = scheduler.get(job_id)
    if job is not None:
        return scheduler._serialize_job(job)

    # Fall back to disk (job from previous session)
    job_store = request.app.state.job_store
    meta = await job_store.read_metadata(job_id)
    if meta is None:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return meta


# ── Stop ──────────────────────────────────────────────────────


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str, request: Request):
    scheduler = request.app.state.scheduler
    ok = await scheduler.stop(job_id)
    if not ok:
        raise HTTPException(404, f"Job '{job_id}' not found or not running")
    logger.info("Job '%s' stop requested", job_id)
    return {"detail": "ok"}
