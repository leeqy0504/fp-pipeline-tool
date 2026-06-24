import logging
import json
from pathlib import Path

import yaml
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from web.routes import _project_root
from web.routes.tasks import _resolve_stage_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

ALLOWED_EXTENSIONS = {".yaml", ".yml"}


class SaveConfigBody(BaseModel):
    content: str


class BusinessConfigBody(BaseModel):
    pipeline: str = "pose6d"
    class_id: int = 0
    first_frame: int = 0
    longest_edge: float = 1.0
    points: list[list[int]] = []
    labels: list[int] = []
    enabled_stages: list[str] = []


def _configs_dir(request: Request) -> Path:
    return _project_root(request) / "configs"


def _tasks_dir(request: Request) -> Path:
    return _project_root(request) / "tasks"


def _registry_dir(request: Request) -> Path:
    return _project_root(request) / "registry"


def _task_yaml_path(task_dir: Path) -> Path:
    return task_dir / "task.yaml"


def _task_settings_path(task_dir: Path) -> Path:
    return task_dir / "pipeline_settings.json"


def _read_yaml_file(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read JSON: %s", path, exc_info=True)
        return default


def _write_yaml_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _list_pipelines(project_root: Path) -> list[dict]:
    pipeline_dir = project_root / "configs" / "pipelines"
    items = []
    if not pipeline_dir.exists():
        return [{"id": "foundationpose", "name": "foundationpose"}]
    for path in sorted(pipeline_dir.glob("*.y*ml")):
        data = _read_yaml_file(path)
        pipeline_id = data.get("preset") or path.stem
        items.append({
            "id": pipeline_id,
            "name": pipeline_id,
            "stages": data.get("stages", []),
        })
    return items


def _list_classes(project_root: Path) -> list[dict]:
    raw = _read_json_file(project_root / "registry" / "classes.json", [])
    items = raw.get("classes", []) if isinstance(raw, dict) else raw
    return [
        {
            "class_id": int(item.get("class_id", 0)),
            "name": item.get("name") or item.get("class_name") or "object",
            "description": item.get("description", ""),
        }
        for item in items
    ]


def _class_name(classes: list[dict], class_id: int) -> str:
    for item in classes:
        if int(item["class_id"]) == int(class_id):
            return item["name"]
    return "object"


def _task_business_config(project_root: Path, task_name: str) -> dict:
    task_dir = project_root / "tasks" / task_name
    if not task_dir.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")

    data = _read_yaml_file(_task_yaml_path(task_dir))
    info = _read_json_file(task_dir / "dataset_info.json", {})
    settings = _read_json_file(_task_settings_path(task_dir), None)
    classes = _list_classes(project_root)

    sam2_from_info = info.get("sam2_points", {})
    real_size_from_info = info.get("real_size", {})
    input_data = data.get("input", {})
    sam2_data = data.get("sam2", {})
    real_size_data = data.get("real_size", {})
    pipeline = data.get("pipeline") or data.get("preset") or "pose6d"
    pipeline_stages = _pipeline_stages(project_root, pipeline)
    stage_settings = settings or _resolve_stage_settings(
        pipeline,
        pipeline_stages,
        stages=pipeline_stages,
    )
    class_id = int(data.get("class_id", 0))

    return {
        "task_name": task_name,
        "pipeline": pipeline,
        "class_id": class_id,
        "class_name": _class_name(classes, class_id),
        "first_frame": int(input_data.get("first_frame", 0)),
        "longest_edge": float(
            real_size_data.get(
                "longest_edge",
                real_size_from_info.get("longest_edge", 1.0),
            )
        ),
        "points": sam2_data.get("points", sam2_from_info.get("points", [])),
        "labels": sam2_data.get("labels", sam2_from_info.get("labels", [])),
        "stage_settings": stage_settings,
        "has_task_yaml": _task_yaml_path(task_dir).exists(),
    }


def _pipeline_stages(project_root: Path, pipeline: str) -> list[str]:
    pipeline_path = project_root / "configs" / "pipelines" / f"{pipeline}.yaml"
    if not pipeline_path.exists():
        pipeline_path = project_root / "configs" / "pipelines" / f"{pipeline}.yml"
    if pipeline_path.exists():
        return _read_yaml_file(pipeline_path).get("stages", [])
    from pipeline.pipeline import PipelineOrchestrator

    return PipelineOrchestrator().resolve_preset(pipeline)


def _save_business_config(project_root: Path, task_name: str, body: BusinessConfigBody) -> dict:
    task_dir = project_root / "tasks" / task_name
    if not task_dir.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")
    if body.longest_edge <= 0:
        raise HTTPException(400, "longest_edge must be greater than 0")
    if body.first_frame < 0:
        raise HTTPException(400, "first_frame must be greater than or equal to 0")
    if len(body.points) != len(body.labels):
        raise HTTPException(400, "points and labels arrays must have the same length")

    classes = _list_classes(project_root)
    if classes and not any(int(c["class_id"]) == int(body.class_id) for c in classes):
        raise HTTPException(400, f"Unknown class_id: {body.class_id}")

    stages = _pipeline_stages(project_root, body.pipeline)
    enabled = body.enabled_stages or stages
    stage_settings = _resolve_stage_settings(body.pipeline, enabled, stages=stages)

    existing = _read_yaml_file(_task_yaml_path(task_dir))
    data = {
        **existing,
        "task_id": task_name,
        "pipeline": body.pipeline,
        "runtime": existing.get("runtime", "server"),
        "class_id": int(body.class_id),
        "input": {
            **existing.get("input", {}),
            "rgbd_dir": f"./tasks/{task_name}/",
            "multi_views_dir": f"./tasks/{task_name}/views/",
            "first_frame": int(body.first_frame),
        },
        "sam2": {
            **existing.get("sam2", {}),
            "points": body.points,
            "labels": body.labels,
        },
        "real_size": {
            **existing.get("real_size", {}),
            "longest_edge": float(body.longest_edge),
        },
        "output_dir": existing.get("output_dir", "output/"),
    }
    _write_yaml_file(_task_yaml_path(task_dir), data)
    _task_settings_path(task_dir).write_text(
        json.dumps(stage_settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    info_path = task_dir / "dataset_info.json"
    info = _read_json_file(info_path, {})
    info["sam2_points"] = {
        "points": body.points,
        "labels": body.labels,
        "description": "前景=1 背景=0. 在首帧 RGB 图上采集，供 sam2mask 阶段使用",
    }
    info["real_size"] = {"longest_edge": float(body.longest_edge)}
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")

    logger.info("Business config saved for task '%s'", task_name)
    return _task_business_config(project_root, task_name)


# ── List ──────────────────────────────────────────────────────


@router.get("/configs")
async def list_configs(request: Request):
    configs_dir = _configs_dir(request)
    if not configs_dir.exists():
        return []
    configs = []
    for p in sorted(configs_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
            configs.append({"name": p.name, "size": p.stat().st_size})
    return configs


@router.get("/configs/business")
async def get_business_config_index(request: Request):
    project_root = _project_root(request)
    tasks_dir = _tasks_dir(request)
    tasks = []
    if tasks_dir.exists():
        tasks = [
            {"task_name": p.name, "has_task_yaml": _task_yaml_path(p).exists()}
            for p in sorted(tasks_dir.iterdir())
            if p.is_dir()
        ]
    return {
        "tasks": tasks,
        "pipelines": _list_pipelines(project_root),
        "classes": _list_classes(project_root),
    }


@router.get("/configs/business/tasks/{task_name}")
async def get_business_task_config(task_name: str, request: Request):
    _validate_task_name(task_name)
    return _task_business_config(_project_root(request), task_name)


@router.put("/configs/business/tasks/{task_name}")
async def save_business_task_config(
    task_name: str,
    body: BusinessConfigBody,
    request: Request,
):
    _validate_task_name(task_name)
    return _save_business_config(_project_root(request), task_name, body)


# ── Read ──────────────────────────────────────────────────────


@router.get("/configs/{name}")
async def get_config(name: str, request: Request):
    _validate_config_name(name)
    path = _configs_dir(request) / name
    if not path.exists():
        raise HTTPException(404, f"Config '{name}' not found")
    return {"name": name, "content": path.read_text(encoding="utf-8")}


# ── Save ──────────────────────────────────────────────────────


@router.put("/configs/{name}")
async def save_config(name: str, body: SaveConfigBody, request: Request):
    _validate_config_name(name)
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Config name must end with .yaml or .yml")

    # Validate YAML before saving
    try:
        yaml.safe_load(body.content)
    except yaml.YAMLError as e:
        detail = f"Invalid YAML: {e}"
        if hasattr(e, "problem_mark") and e.problem_mark:
            mark = e.problem_mark
            detail = f"Invalid YAML at line {mark.line + 1}, column {mark.column + 1}: {e.problem}"
        raise HTTPException(400, detail=detail)

    path = _configs_dir(request) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.content, encoding="utf-8")
    logger.info("Config '%s' saved", name)
    return {"detail": "ok"}


# ── Helpers ───────────────────────────────────────────────────


def _validate_config_name(name: str):
    """Reject path traversal attempts."""
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(400, "Invalid config name")


def _validate_task_name(name: str):
    if ".." in name or "/" in name or "\\" in name or "\x00" in name:
        raise HTTPException(400, "Invalid task name")
