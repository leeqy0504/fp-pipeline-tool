import json
import logging
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class RunRequest(BaseModel):
    preset: str
    config_path: str | None = None


def _project_root(request: Request) -> Path:
    return request.app.state.project_root


def _tasks_dir(request: Request) -> Path:
    return _project_root(request) / "tasks"


def _output_dir(request: Request) -> Path:
    return _project_root(request) / "output"


# ── Task listing ──────────────────────────────────────────────


@router.get("/tasks")
async def list_tasks(request: Request):
    tasks_dir = _tasks_dir(request)
    if not tasks_dir.exists():
        return []

    scheduler = request.app.state.scheduler
    job_store = request.app.state.job_store
    output_dir = _output_dir(request)

    # Build fast lookups from scheduler
    all_jobs = await scheduler.list_all()
    running_map = {}
    latest_map = {}
    for j in all_jobs:
        key = j.task_name
        if j.status.value == "running":
            running_map[key] = j.job_id
        if key not in latest_map or j.start_time > latest_map[key][0]:
            latest_map[key] = (j.start_time, j.job_id)

    # On-disk fallback for latest_job_id (jobs from previous sessions)
    disk_jobs = await job_store.list_jobs()
    disk_latest = {}
    for dj in disk_jobs:
        key = dj.get("task_name")
        if not key:
            continue
        st = dj.get("start_time", 0)
        if key not in disk_latest or st > disk_latest[key][0]:
            disk_latest[key] = (st, dj["job_id"])

    tasks = []
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name

        manifest = None
        manifest_path = output_dir / task_name / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        running_job_id = running_map.get(task_name)

        # Prefer in-memory latest, fall back to disk
        if task_name in latest_map:
            latest_job_id = latest_map[task_name][1]
        elif task_name in disk_latest:
            latest_job_id = disk_latest[task_name][1]
        else:
            latest_job_id = None

        tasks.append({
            "task_name": task_name,
            "manifest": manifest,
            "latest_job_id": latest_job_id,
            "running_job_id": running_job_id,
        })

    return tasks


# ── Task detail ───────────────────────────────────────────────


@router.get("/tasks/{task_name}")
async def get_task(task_name: str, request: Request):
    tasks_dir = _tasks_dir(request)
    task_path = tasks_dir / task_name
    if not task_path.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")

    scheduler = request.app.state.scheduler
    job_store = request.app.state.job_store
    output_dir = _output_dir(request)

    manifest = None
    manifest_path = output_dir / task_name / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # running job
    running = await scheduler.list_running()
    running_job_id = None
    for j in running:
        if j.task_name == task_name:
            running_job_id = j.job_id
            break

    # latest job: in-memory first, then disk
    all_jobs = await scheduler.list_all()
    task_jobs = [j for j in all_jobs if j.task_name == task_name]
    if task_jobs:
        latest = max(task_jobs, key=lambda j: j.start_time)
        latest_job_id = latest.job_id
    else:
        disk_jobs = await job_store.list_jobs()
        disk_task_jobs = [j for j in disk_jobs if j.get("task_name") == task_name]
        if disk_task_jobs:
            latest = max(disk_task_jobs, key=lambda j: j.get("start_time", 0))
            latest_job_id = latest["job_id"]
        else:
            latest_job_id = None

    return {
        "task_name": task_name,
        "manifest": manifest,
        "latest_job_id": latest_job_id,
        "running_job_id": running_job_id,
    }


# ── Run ───────────────────────────────────────────────────────


@router.post("/tasks/{task_name}/run")
async def run_task(task_name: str, body: RunRequest, request: Request):
    scheduler = request.app.state.scheduler
    try:
        job_id = await scheduler.submit(task_name, body.preset, body.config_path)
        return {"job_id": job_id}
    except Exception as e:
        from web.scheduler import ConflictError
        if isinstance(e, ConflictError):
            raise HTTPException(409, detail=str(e))
        raise HTTPException(500, detail=str(e))


# ── Reset ─────────────────────────────────────────────────────


@router.post("/tasks/{task_name}/reset")
async def reset_task(task_name: str, request: Request):
    output_dir = _output_dir(request)
    manifest_path = output_dir / task_name / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, f"No manifest found for task '{task_name}'")

    # Safety check: don't reset if a job is running for this task
    scheduler = request.app.state.scheduler
    running = await scheduler.list_running()
    for j in running:
        if j.task_name == task_name:
            raise HTTPException(409, f"Task '{task_name}' has a running job ({j.job_id}). Stop it first.")

    manifest_path.unlink()
    logger.info("Manifest deleted for task '%s'", task_name)
    return {"detail": "ok"}
