import json
import logging
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from web.routes import _project_root

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class RunRequest(BaseModel):
    preset: str
    config_path: str | None = None
    stage_selection: dict | None = None


class StageSettingsRequest(BaseModel):
    preset: str
    enabled: list[str]


class CloneRequest(BaseModel):
    task_name: str | None = None

def _tasks_dir(request: Request) -> Path:
    return _project_root(request) / "tasks"


def _output_dir(request: Request) -> Path:
    return _project_root(request) / "output"


def _sanitize_task_name(name: str) -> str:
    name = (name or "").replace("\x00", "").strip()
    name = name.replace("/", "_").replace("\\", "_")
    if not name or ".." in name:
        return ""
    return name


def _settings_path(task_path: Path) -> Path:
    return task_path / "pipeline_settings.json"


def _task_config_path(task_path: Path) -> Path:
    return task_path / "pipeline_config.yaml"


def _load_stage_settings(task_path: Path) -> dict | None:
    path = _settings_path(task_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read pipeline settings: %s", path, exc_info=True)
        return None


def _stage_labels() -> dict[str, str]:
    return {
        "masks": "分割",
        "hunyuangen": "三维生成",
        "scale": "尺度标定",
        "package": "打包",
        "foundationpose": "FoundationPose",
        "detection_dataset": "检测数据集",
    }


def _default_clone_name(source_name: str, tasks_dir: Path) -> str:
    base = re.sub(r"-copy(?:-\d+)?$", "", source_name)
    candidate = f"{base}-copy"
    idx = 2
    while (tasks_dir / candidate).exists():
        candidate = f"{base}-copy-{idx}"
        idx += 1
    return candidate


def _resolve_stage_settings(preset: str, enabled: list[str]) -> dict:
    from pipeline.pipeline import PipelineOrchestrator

    stages = PipelineOrchestrator().resolve_preset(preset)
    stage_set = set(stages)
    unknown = [stage for stage in enabled if stage not in stage_set]
    if unknown:
        raise HTTPException(400, f"Unknown stage(s): {', '.join(unknown)}")
    enabled_ordered = [stage for stage in stages if stage in set(enabled)]
    skipped = [stage for stage in stages if stage not in set(enabled)]
    return {
        "preset": preset,
        "stages": stages,
        "enabled": enabled_ordered,
        "skipped": skipped,
        "labels": _stage_labels(),
    }


def _write_task_config_snapshot(project_root: Path, source_path: Path, dest_path: Path, new_name: str) -> None:
    import yaml

    source_config = _task_config_path(source_path)
    if not source_config.exists():
        source_config = project_root / "configs" / "foundationpose.yaml"
    if not source_config.exists():
        return

    config = yaml.safe_load(source_config.read_text(encoding="utf-8")) or {}
    config["task"] = new_name
    input_config = config.setdefault("input", {})
    input_config["rgbd_dir"] = f"./tasks/{new_name}/"
    input_config["multi_views_dir"] = f"./tasks/{new_name}/views/"

    _task_config_path(dest_path).write_text(
        yaml.dump(config, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )


# ── Shared task-list builder ──────────────────────────────────


async def _build_tasks_list(scheduler, job_store, project_root: Path) -> list[dict]:
    """Return a sorted list of task dicts with manifest + job lookups.

    Used by both the API endpoint and the page routes in app.py.
    """
    tasks_dir = project_root / "tasks"
    output_dir = project_root / "output"

    if not tasks_dir.exists():
        return []

    all_jobs = await scheduler.list_all()
    running_map = {}
    latest_map = {}
    for j in all_jobs:
        if j.status.value == "running":
            running_map[j.task_name] = j.job_id
        if j.task_name not in latest_map or j.start_time > latest_map[j.task_name][0]:
            latest_map[j.task_name] = (j.start_time, j.job_id)

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
            "stage_settings": _load_stage_settings(task_dir),
        })

    return tasks


# ── Task listing ──────────────────────────────────────────────


@router.get("/tasks")
async def list_tasks(request: Request):
    return await _build_tasks_list(
        request.app.state.scheduler,
        request.app.state.job_store,
        _project_root(request),
    )


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
        "stage_settings": _load_stage_settings(task_path),
    }


@router.get("/tasks/{task_name}/stage-settings")
async def get_stage_settings(task_name: str, request: Request):
    task_path = _tasks_dir(request) / task_name
    if not task_path.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")
    saved = _load_stage_settings(task_path)
    if saved:
        return saved
    from pipeline.pipeline import PipelineOrchestrator
    stages = PipelineOrchestrator().resolve_preset("foundationpose")
    return _resolve_stage_settings("foundationpose", stages)


@router.post("/tasks/{task_name}/stage-settings")
async def save_stage_settings(task_name: str, body: StageSettingsRequest, request: Request):
    task_path = _tasks_dir(request) / task_name
    if not task_path.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")
    settings = _resolve_stage_settings(body.preset, body.enabled)
    _settings_path(task_path).write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Pipeline settings saved for task '%s': enabled=%s skipped=%s",
                task_name, settings["enabled"], settings["skipped"])
    return settings


# ── Run ───────────────────────────────────────────────────────


@router.post("/tasks/{task_name}/run")
async def run_task(task_name: str, body: RunRequest, request: Request):
    scheduler = request.app.state.scheduler
    try:
        task_path = _tasks_dir(request) / task_name
        if not task_path.is_dir():
            raise HTTPException(404, f"Task '{task_name}' not found")
        config_path = body.config_path
        local_config = _task_config_path(task_path)
        if config_path is None and local_config.exists():
            config_path = str(local_config)
        stage_selection = body.stage_selection or _load_stage_settings(task_path)
        if stage_selection:
            stage_selection = _resolve_stage_settings(
                stage_selection.get("preset", body.preset),
                stage_selection.get("enabled", []),
            )
        job_id = await scheduler.submit(
            task_name,
            body.preset,
            config_path,
            stage_selection=stage_selection,
        )
        return {"job_id": job_id}
    except HTTPException:
        raise
    except Exception as e:
        from web.scheduler import ConflictError
        if isinstance(e, ConflictError):
            raise HTTPException(409, detail=str(e))
        raise HTTPException(500, detail=str(e))


# ── Delete ────────────────────────────────────────────────────


@router.delete("/tasks/{task_name}")
async def delete_task(task_name: str, request: Request):
    task_path = _tasks_dir(request) / task_name
    if not task_path.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")

    scheduler = request.app.state.scheduler
    running = await scheduler.list_running()
    for j in running:
        if j.task_name == task_name:
            raise HTTPException(409, f"Task '{task_name}' has a running job ({j.job_id}). Stop it first.")

    shutil.rmtree(task_path)
    output_path = _output_dir(request) / task_name
    if output_path.exists():
        shutil.rmtree(output_path)

    job_store = request.app.state.job_store
    jobs = await job_store.list_jobs()
    for job in jobs:
        if job.get("task_name") == task_name:
            await job_store.delete_job(job["job_id"])

    logger.info("Task '%s' deleted", task_name)
    return {"detail": "ok"}


# ── Clone ─────────────────────────────────────────────────────


@router.post("/tasks/{task_name}/clone")
async def clone_task(task_name: str, body: CloneRequest, request: Request):
    tasks_dir = _tasks_dir(request)
    source_path = tasks_dir / task_name
    if not source_path.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")

    requested = _sanitize_task_name(body.task_name or "")
    new_name = requested or _default_clone_name(task_name, tasks_dir)
    if not new_name:
        raise HTTPException(400, "Invalid task name")
    dest_path = tasks_dir / new_name
    if dest_path.exists():
        raise HTTPException(409, f"Task '{new_name}' already exists")

    shutil.copytree(source_path, dest_path)
    _write_task_config_snapshot(_project_root(request), source_path, dest_path, new_name)

    info_path = dest_path / "dataset_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["cloned_from"] = task_name
            info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")
        except Exception:
            logger.warning("Failed to annotate cloned dataset_info for task '%s'", new_name, exc_info=True)

    logger.info("Task '%s' cloned to '%s'", task_name, new_name)
    return {"task_name": new_name, "source_task": task_name}


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
