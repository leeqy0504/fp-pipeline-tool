import json
import logging
import re
import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
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


class ReviewStateRequest(BaseModel):
    frame: str
    state: str


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


def _task_yaml_path(task_path: Path) -> Path:
    return task_path / "task.yaml"


def _read_task_yaml(task_path: Path) -> dict:
    import yaml

    path = _task_yaml_path(task_path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("Failed to read task.yaml: %s", path, exc_info=True)
        return {}


def _task_pipeline(task_path: Path) -> str:
    saved = _load_stage_settings(task_path)
    if saved and saved.get("preset"):
        return str(saved["preset"])
    data = _read_task_yaml(task_path)
    return str(data.get("pipeline") or data.get("preset") or "pose6d")


def _ensure_task_yaml(project_root: Path, task_name: str) -> Path:
    import yaml

    task_path = project_root / "tasks" / task_name
    path = _task_yaml_path(task_path)
    if path.exists():
        return path

    info_path = task_path / "dataset_info.json"
    info = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read dataset_info for task '%s'", task_name, exc_info=True)

    sam2_points = info.get("sam2_points", {})
    real_size = info.get("real_size", {})
    data = {
        "task_id": task_name,
        "pipeline": "pose6d",
        "runtime": "server",
        "class_id": 0,
        "input": {
            "rgbd_dir": f"./tasks/{task_name}/",
            "multi_views_dir": f"./tasks/{task_name}/views/",
            "first_frame": 0,
        },
        "sam2": {
            "points": sam2_points.get("points", []),
            "labels": sam2_points.get("labels", []),
        },
        "real_size": {
            "longest_edge": real_size.get("longest_edge", 1.0),
        },
        "output_dir": "output/",
    }
    path.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _load_stage_settings(task_path: Path) -> dict | None:
    path = _settings_path(task_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read pipeline settings: %s", path, exc_info=True)
        return None


def _latest_manifest_path(output_dir: Path, task_name: str) -> Path | None:
    task_output = output_dir / task_name
    run_root = task_output / "runs"
    candidates: list[Path] = []
    if run_root.exists():
        candidates.extend(run_root.glob("*/manifest.json"))
    legacy = task_output / "manifest.json"
    if legacy.exists():
        candidates.append(legacy)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_latest_manifest(output_dir: Path, task_name: str) -> dict | None:
    path = _latest_manifest_path(output_dir, task_name)
    if not path:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read manifest: %s", path, exc_info=True)
        return None


def _safe_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read JSON artifact: %s", path, exc_info=True)
        return None


def _rel_to_project(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _file_item(project_root: Path, path: Path, label: str | None = None) -> dict:
    rel_path = _rel_to_project(project_root, path)
    return {
        "name": path.name,
        "label": label or path.name,
        "path": rel_path,
        "url": f"/api/tasks/artifact-file?path={quote(rel_path)}",
        "size": path.stat().st_size if path.exists() else None,
    }


def _stage_dir(manifest: dict | None, stage_name: str) -> Path | None:
    if not manifest:
        return None
    info = (manifest.get("stages") or {}).get(stage_name)
    output = info.get("output_dir") if info else None
    return Path(output) if output else None


def _latest_run_dir(manifest: dict | None) -> Path | None:
    if not manifest:
        return None
    run_dir = (manifest.get("metadata") or {}).get("run_dir")
    return Path(run_dir) if run_dir else None


def _sample_files(paths: list[Path], limit: int = 24) -> list[Path]:
    return sorted(paths)[:limit]


def _review_status_path(qa_dir: Path) -> Path:
    return qa_dir / "review_status.json"


def _load_review_status(qa_dir: Path | None) -> dict:
    if not qa_dir:
        return {}
    path = _review_status_path(qa_dir)
    if not path.exists():
        return {}
    return _safe_json(path) or {}


def _save_review_status(qa_dir: Path, data: dict) -> None:
    path = _review_status_path(qa_dir)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _annotation_frame_items(project_root: Path, task_dir: Path, qa_report: dict | None,
                            review_status: dict, export_dir: Path | None) -> list[dict]:
    if not qa_report:
        return []
    frames = review_status.get("frames", {})
    rgb_dir = task_dir / "rgb"
    masks_dir = Path(qa_report["source_masks"])
    annotations_by_frame = {}
    annotations = _safe_json(export_dir / "annotations.json") if export_dir else None
    for ann in (annotations or {}).get("annotations", []):
        annotations_by_frame[ann.get("frame")] = ann

    items = []
    for row in qa_report.get("frames", []):
        frame = row["frame"]
        review_info = frames.get(frame, {})
        state = review_info.get("state", row.get("state", "accepted"))
        image_path = rgb_dir / frame
        mask_path = masks_dir / frame
        preview_path = export_dir / "preview" / f"{Path(frame).stem}.svg" if export_dir else None
        ann = annotations_by_frame.get(frame)
        items.append({
            "frame": frame,
            "state": state,
            "qa_state": row.get("state"),
            "manual": bool(review_info.get("manual")),
            "reason": row.get("flags", []),
            "bbox_xyxy": row.get("bbox_xyxy"),
            "area": row.get("area"),
            "image": _file_item(project_root, image_path) if image_path.exists() else None,
            "mask": _file_item(project_root, mask_path) if mask_path.exists() else None,
            "preview": _file_item(project_root, preview_path) if preview_path and preview_path.exists() else None,
            "exported": ann is not None,
        })
    return items


def _rebuild_detection_dataset_from_review(project_root: Path, task_name: str) -> dict:
    from pipeline.config import load_config
    from pipeline.stages.annotation_dataset import DetectionDatasetExportStage
    from pipeline.stages.context import DataContext, RunContext, StageContext

    task_dir = project_root / "tasks" / task_name
    manifest = _load_latest_manifest(project_root / "output", task_name)
    qa_dir = _stage_dir(manifest, "mask_qa")
    export_dir = _stage_dir(manifest, "detection_dataset_export")
    if not qa_dir:
        raise HTTPException(400, "mask_qa output not found")
    if not export_dir:
        run_dir = _latest_run_dir(manifest)
        if not run_dir:
            raise HTTPException(400, "run_dir not found")
        export_dir = run_dir / "stages" / "detection_dataset_export"

    config_path = _task_yaml_path(task_dir)
    if not config_path.exists():
        raise HTTPException(400, "task.yaml not found")
    config = load_config(str(config_path), project_root=project_root)
    if manifest:
        config.run_id = manifest.get("run_id")
    if not Path(config.input.rgbd_dir).is_absolute():
        config.input.rgbd_dir = str(project_root / config.input.rgbd_dir)
    if not Path(config.input.multi_views_dir).is_absolute():
        config.input.multi_views_dir = str(project_root / config.input.multi_views_dir)
    if not Path(config.output_dir).is_absolute():
        config.output_dir = str(project_root / config.output_dir)

    context = StageContext(
        run=RunContext(run_id=config.run_id, task_name=task_name),
        data=DataContext(
            task_dir=Path(config.input.rgbd_dir),
            run_dir=_latest_run_dir(manifest) or export_dir.parent.parent,
            output_dir=export_dir,
            inputs={"mask_qa": qa_dir},
        ),
        stage_name="detection_dataset_export",
    )
    DetectionDatasetExportStage().run(config, export_dir, context=context)
    return _build_task_artifacts(project_root, task_name)


def _build_task_artifacts(project_root: Path, task_name: str) -> dict:
    task_dir = project_root / "tasks" / task_name
    output_dir = project_root / "output"
    manifest = _load_latest_manifest(output_dir, task_name)
    pipeline = _task_pipeline(task_dir)

    rgb_files = _sample_files(list((task_dir / "rgb").glob("*.png")) + list((task_dir / "rgb").glob("*.jpg")), 7)
    depth_files = _sample_files(list((task_dir / "depth").glob("*.png")), 12)
    view_files = _sample_files(
        list((task_dir / "views").glob("*.png"))
        + list((task_dir / "views").glob("*.jpg"))
        + list((task_dir / "views").glob("*.jpeg")),
        12,
    )

    artifacts = {
        "task_name": task_name,
        "pipeline": pipeline,
        "manifest": manifest,
        "dataset": {
            "rgb": [_file_item(project_root, p) for p in rgb_files],
            "depth": [_file_item(project_root, p) for p in depth_files],
            "views": [_file_item(project_root, p) for p in view_files],
        },
        "annotation": None,
        "pose6d": None,
    }

    if pipeline == "annotation_dataset":
        prompt_dir = _stage_dir(manifest, "prompt_mask")
        video_dir = _stage_dir(manifest, "sam2_video_propagation")
        qa_dir = _stage_dir(manifest, "mask_qa")
        review_dir = _stage_dir(manifest, "review_pack")
        export_dir = _stage_dir(manifest, "detection_dataset_export")

        qa_report = _safe_json(qa_dir / "qa_report.json") if qa_dir else None
        review_status = _load_review_status(qa_dir)
        annotations = _safe_json(export_dir / "annotations.json") if export_dir else None
        masks_dir = video_dir / "masks" if video_dir else None
        mask_files = _sample_files(list(masks_dir.glob("*.png")) if masks_dir and masks_dir.exists() else [], 12)
        exported_images = _sample_files(list((export_dir / "images").glob("*.png")) if export_dir else [], 12)
        exported_labels = sorted((export_dir / "labels").glob("*.txt"))[:30] if export_dir and (export_dir / "labels").exists() else []
        contact_sheet = export_dir / "contact_sheet.svg" if export_dir else None
        artifacts["annotation"] = {
            "prompt_masks": [_file_item(project_root, p) for p in _sample_files(list(prompt_dir.glob("*.png")) if prompt_dir else [], 6)],
            "propagated_masks": [_file_item(project_root, p) for p in mask_files],
            "qa_summary": (qa_report or {}).get("summary"),
            "qa_frames": [
                row for row in (qa_report or {}).get("frames", [])
                if row.get("state") != "accepted"
            ][:120],
            "review_status": review_status,
            "frames": _annotation_frame_items(project_root, task_dir, qa_report, review_status, export_dir),
            "contact_sheet": _file_item(project_root, contact_sheet, "contact_sheet.svg") if contact_sheet and contact_sheet.exists() else None,
            "review_pack": _file_item(project_root, review_dir / "index.html", "review_pack.html") if review_dir and (review_dir / "index.html").exists() else None,
            "dataset_yaml": _file_item(project_root, export_dir / "dataset.yaml", "dataset.yaml") if export_dir and (export_dir / "dataset.yaml").exists() else None,
            "annotations": (annotations or {}),
            "exported_images": [_file_item(project_root, p) for p in exported_images],
            "exported_labels": [_file_item(project_root, p) for p in exported_labels],
        }
    else:
        package_dir = _stage_dir(manifest, "package")
        fp_dir = _stage_dir(manifest, "foundationpose")
        det_dir = _stage_dir(manifest, "detection_dataset")
        fp_visuals = []
        if fp_dir and fp_dir.exists():
            for pattern in ("*.png", "**/*.png", "*.jpg", "**/*.jpg"):
                fp_visuals.extend(fp_dir.glob(pattern))
        pose_files = sorted((fp_dir / "ob_in_cam").glob("*.txt"))[:20] if fp_dir and (fp_dir / "ob_in_cam").exists() else []
        package_rgb = _sample_files(list((package_dir / "rgb").glob("*.png")) if package_dir else [], 20)
        package_masks = _sample_files(list((package_dir / "masks").glob("*.png")) if package_dir else [], 20)
        det_annotations = _safe_json(det_dir / "annotations.json") if det_dir else None
        artifacts["pose6d"] = {
            "package_rgb": [_file_item(project_root, p) for p in package_rgb],
            "package_masks": [_file_item(project_root, p) for p in package_masks],
            "fp_visuals": [_file_item(project_root, p) for p in _sample_files(list(set(fp_visuals)), 30)],
            "pose_files": [_file_item(project_root, p) for p in pose_files],
            "detection_dataset": det_annotations,
            "dataset_yaml": _file_item(project_root, det_dir / "dataset.yaml", "dataset.yaml") if det_dir and (det_dir / "dataset.yaml").exists() else None,
        }

    return artifacts


def _stage_labels() -> dict[str, str]:
    return {
        "masks": "分割",
        "prompt_mask": "首帧分割",
        "sam2_video_propagation": "视频传播",
        "mask_qa": "Mask 质检",
        "review_pack": "预览包",
        "detection_dataset_export": "检测数据集导出",
        "hunyuangen": "三维生成",
        "scale": "尺度标定",
        "package": "打包",
        "foundationpose": "6D 位姿",
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


def _resolve_stage_settings(preset: str, enabled: list[str], stages: list[str] | None = None) -> dict:
    if stages is None:
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


def _resolve_run_stage_settings(project_root: Path, preset: str, selection: dict | None) -> dict | None:
    """Normalize run-time stage selection for a preset.

    The top-level run preset is authoritative. Older saved settings or stale
    browser state may carry a different ``selection.preset``; using that value
    can silently run the wrong pipeline from the task detail page.
    """
    if not selection:
        return None
    from pipeline.pipeline import PipelineOrchestrator

    stages = PipelineOrchestrator(project_root=project_root).resolve_preset(preset)
    if "enabled" in selection and selection.get("enabled") is not None:
        return _resolve_stage_settings(preset, selection.get("enabled", []), stages=stages)

    skipped = set(selection.get("skipped", []))
    enabled = [stage for stage in stages if stage not in skipped]
    return _resolve_stage_settings(preset, enabled, stages=stages)


def _default_stage_settings(task_path: Path) -> dict:
    from pipeline.pipeline import PipelineOrchestrator

    preset = _task_pipeline(task_path)
    stages = PipelineOrchestrator(project_root=task_path.parent.parent).resolve_preset(preset)
    return _resolve_stage_settings(preset, stages)


def _sync_task_pipeline(task_path: Path, preset: str) -> None:
    import yaml

    config = _read_task_yaml(task_path)
    config["pipeline"] = preset
    config.pop("preset", None)
    _task_yaml_path(task_path).write_text(
        yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _write_task_config_snapshot(project_root: Path, source_path: Path, dest_path: Path, new_name: str) -> None:
    import yaml

    source_config = _task_yaml_path(source_path)
    if not source_config.exists():
        source_config = _ensure_task_yaml(project_root, source_path.name)

    config = yaml.safe_load(source_config.read_text(encoding="utf-8")) or {}
    config["task_id"] = new_name
    config.pop("task", None)
    config.setdefault("pipeline", "pose6d")
    config.setdefault("runtime", "server")
    input_config = config.setdefault("input", {})
    input_config["rgbd_dir"] = f"./tasks/{new_name}/"
    input_config["multi_views_dir"] = f"./tasks/{new_name}/views/"

    _task_yaml_path(dest_path).write_text(
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

        manifest = _load_latest_manifest(output_dir, task_name)

        running_job_id = running_map.get(task_name)
        if task_name in latest_map:
            latest_job_id = latest_map[task_name][1]
        elif task_name in disk_latest:
            latest_job_id = disk_latest[task_name][1]
        else:
            latest_job_id = None

        tasks.append({
            "task_name": task_name,
            "pipeline": _task_pipeline(task_dir),
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


@router.get("/tasks/artifact-file")
async def get_artifact_file(path: str, request: Request):
    project_root = _project_root(request).resolve()
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(400, "Invalid artifact path")
    file_path = (project_root / rel).resolve()
    try:
        file_path.relative_to(project_root)
    except ValueError:
        raise HTTPException(403, "Artifact path is outside project root")
    if not file_path.is_file():
        raise HTTPException(404, "Artifact file not found")
    return FileResponse(str(file_path))


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

    manifest = _load_latest_manifest(output_dir, task_name)

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
        "pipeline": _task_pipeline(task_path),
        "manifest": manifest,
        "latest_job_id": latest_job_id,
        "running_job_id": running_job_id,
        "stage_settings": _load_stage_settings(task_path) or _default_stage_settings(task_path),
    }


@router.get("/tasks/{task_name}/artifacts")
async def get_task_artifacts(task_name: str, request: Request):
    task_path = _tasks_dir(request) / task_name
    if not task_path.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")
    return _build_task_artifacts(_project_root(request), task_name)


@router.post("/tasks/{task_name}/review-frame")
async def update_review_frame(task_name: str, body: ReviewStateRequest, request: Request):
    task_path = _tasks_dir(request) / task_name
    if not task_path.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")
    if body.state not in {"accepted", "rejected", "suspect"}:
        raise HTTPException(400, "state must be accepted, rejected, or suspect")
    if "/" in body.frame or "\\" in body.frame or ".." in body.frame:
        raise HTTPException(400, "invalid frame")

    project_root = _project_root(request)
    manifest = _load_latest_manifest(project_root / "output", task_name)
    qa_dir = _stage_dir(manifest, "mask_qa")
    if not qa_dir:
        raise HTTPException(400, "mask_qa output not found")
    report = _safe_json(qa_dir / "qa_report.json")
    if not report:
        raise HTTPException(400, "qa_report.json not found")
    rows = {row["frame"]: row for row in report.get("frames", [])}
    if body.frame not in rows:
        raise HTTPException(404, f"Frame '{body.frame}' not found in QA report")

    review = _load_review_status(qa_dir) or {"task": task_name, "source": "mask_qa", "frames": {}}
    review.setdefault("frames", {})
    review["frames"][body.frame] = {
        "state": body.state,
        "flags": rows[body.frame].get("flags", []),
        "manual": True,
    }
    _save_review_status(qa_dir, review)
    return _rebuild_detection_dataset_from_review(project_root, task_name)


@router.post("/tasks/{task_name}/apply-review")
async def apply_review(task_name: str, request: Request):
    task_path = _tasks_dir(request) / task_name
    if not task_path.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")
    return _rebuild_detection_dataset_from_review(_project_root(request), task_name)


@router.get("/tasks/{task_name}/stage-settings")
async def get_stage_settings(task_name: str, request: Request, preset: str | None = None):
    task_path = _tasks_dir(request) / task_name
    if not task_path.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")
    if preset:
        from pipeline.pipeline import PipelineOrchestrator
        stages = PipelineOrchestrator(project_root=_project_root(request)).resolve_preset(preset)
        return _resolve_stage_settings(preset, stages)
    saved = _load_stage_settings(task_path)
    if saved:
        return saved
    from pipeline.pipeline import PipelineOrchestrator
    task_preset = _task_pipeline(task_path)
    stages = PipelineOrchestrator(project_root=_project_root(request)).resolve_preset(task_preset)
    return _resolve_stage_settings(task_preset, stages)


@router.post("/tasks/{task_name}/stage-settings")
async def save_stage_settings(task_name: str, body: StageSettingsRequest, request: Request):
    task_path = _tasks_dir(request) / task_name
    if not task_path.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")
    from pipeline.pipeline import PipelineOrchestrator
    stages = PipelineOrchestrator(project_root=_project_root(request)).resolve_preset(body.preset)
    settings = _resolve_stage_settings(body.preset, body.enabled, stages=stages)
    _settings_path(task_path).write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _sync_task_pipeline(task_path, settings["preset"])
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
        local_config = _ensure_task_yaml(_project_root(request), task_name)
        if not local_config.exists():
            local_config = _task_config_path(task_path)
        if config_path is None and local_config.exists():
            config_path = str(local_config)
        run_preset = body.preset
        if body.stage_selection is not None:
            stage_selection = _resolve_run_stage_settings(_project_root(request), run_preset, body.stage_selection)
        else:
            saved_selection = _load_stage_settings(task_path)
            if saved_selection and saved_selection.get("preset", run_preset) == run_preset:
                stage_selection = _resolve_run_stage_settings(_project_root(request), run_preset, saved_selection)
            else:
                stage_selection = None
        job_id = await scheduler.submit(
            task_name,
            run_preset,
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
    manifest_path = _latest_manifest_path(output_dir, task_name)
    if manifest_path is None or not manifest_path.exists():
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
