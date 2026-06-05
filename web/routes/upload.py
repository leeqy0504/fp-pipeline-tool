import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from starlette.datastructures import UploadFile

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".json", ".txt", ".yaml", ".yml",
    ".obj", ".ply", ".stl", ".glb", ".gltf", ".mtl",
    ".npy", ".npz", ".csv", ".tsv",
    ".pt", ".pth", ".onnx", ".bin",
}
MAX_FILE_COUNT = 500
DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def _project_root(request: Request) -> Path:
    return request.app.state.project_root


# ── Upload (with directory structure preservation) ──────────────


@router.post("/tasks/upload")
async def upload_files(request: Request):
    form = await request.form()
    task_name = form.get("task_name", "")

    if not task_name:
        raise HTTPException(400, "Missing 'task_name' field")

    sanitized_task = _sanitize_name(task_name)
    if not sanitized_task:
        raise HTTPException(400, "Invalid task name")

    max_size = getattr(request.app.state, "max_body_size", DEFAULT_MAX_UPLOAD_BYTES)

    tasks_dir = _project_root(request) / "tasks"
    dest_dir = tasks_dir / sanitized_task

    # Collect files and their relative paths
    file_items = []  # list of (relative_path, UploadFile)
    flat_files = []  # UploadFile items without path info

    for field_name, value in form.multi_items():
        if not isinstance(value, UploadFile):
            continue
        if not value.filename:
            continue
        if field_name == "files":
            flat_files.append(value)
        else:
            # Field name encodes the relative path, e.g. "file:rgb/00000.png"
            file_items.append((value.filename, value))

    # If flat files were sent, check for a paths JSON field
    paths_json = form.get("paths")
    if paths_json and flat_files:
        try:
            paths = json.loads(str(paths_json))
            for p, f in zip(paths, flat_files):
                file_items.append((p, f))
            flat_files = []
        except (json.JSONDecodeError, ValueError):
            pass

    if len(file_items) > MAX_FILE_COUNT:
        raise HTTPException(400, f"Too many files ({len(file_items)}). Maximum is {MAX_FILE_COUNT}.")

    total_bytes = 0
    uploaded = []

    for rel_path, upload_file in file_items:
        filename = upload_file.filename
        if not filename:
            continue

        # Validate the full relative path (each component)
        safe_path = _sanitize_relpath(rel_path)
        if safe_path is None:
            logger.warning("Upload rejected (path traversal): %s", rel_path)
            raise HTTPException(400, f"Invalid path: {rel_path}")

        ext = Path(safe_path).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            logger.warning("Upload rejected (extension): %s", filename)
            raise HTTPException(400, f"File type not allowed: {ext}")

        contents = await upload_file.read()
        total_bytes += len(contents)
        if total_bytes > max_size:
            raise HTTPException(413, f"Upload exceeds {max_size // (1024 * 1024)} MB limit")

        dest_path = dest_dir / safe_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(contents)
        uploaded.append({"path": safe_path, "size": len(contents)})
        logger.info("Uploaded: %s/%s (%d bytes)", sanitized_task, safe_path, len(contents))

    if not uploaded:
        raise HTTPException(400, "No files uploaded")

    # Summarize what was uploaded
    summary = _summarize_upload(dest_dir, uploaded)

    return {
        "task_name": sanitized_task,
        "files": uploaded,
        "total_bytes": total_bytes,
        "summary": summary,
    }


# ── Task directory summary ─────────────────────────────────────


@router.get("/tasks/{task_name}/summary")
async def task_summary(task_name: str, request: Request):
    tasks_dir = _project_root(request) / "tasks" / task_name
    if not tasks_dir.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")
    return _summarize_upload(tasks_dir, [])


# ── Read dataset_info.json ─────────────────────────────────────


@router.get("/tasks/{task_name}/dataset-info")
async def get_dataset_info(task_name: str, request: Request):
    tasks_dir = _project_root(request) / "tasks" / task_name
    if not tasks_dir.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")

    info_path = tasks_dir / "dataset_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text(encoding="utf-8"))

    return _build_dataset_info(tasks_dir)


# ── Serve first RGB frame for pick-points ──────────────────────


@router.get("/tasks/{task_name}/first-frame")
async def get_first_frame(task_name: str, request: Request):
    from fastapi.responses import Response

    tasks_dir = _project_root(request) / "tasks" / task_name
    rgb_dir = tasks_dir / "rgb"
    if not rgb_dir.is_dir():
        raise HTTPException(404, "No rgb/ directory found in task")

    pngs = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(".png")])
    if not pngs:
        raise HTTPException(404, "No PNG frames found in rgb/")

    content = (rgb_dir / pngs[0]).read_bytes()
    return Response(content=content, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=300"})


# ── Save points + real_size ────────────────────────────────────


@router.post("/tasks/{task_name}/points")
async def save_points(task_name: str, request: Request):
    from pydantic import BaseModel

    class PointsBody(BaseModel):
        points: list[list[int]] = []
        labels: list[int] = []
        longest_edge: float | None = None

    data = await request.json()
    body = PointsBody(**data)

    if len(body.points) != len(body.labels):
        raise HTTPException(400, "points and labels arrays must have the same length")

    if body.longest_edge is not None and body.longest_edge <= 0:
        raise HTTPException(400, "longest_edge must be greater than 0")

    tasks_dir = _project_root(request) / "tasks" / task_name
    if not tasks_dir.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")

    # Load existing dataset_info.json if present, or build one
    info_path = tasks_dir / "dataset_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        info = _build_dataset_info(tasks_dir)

    info["sam2_points"] = {
        "points": body.points,
        "labels": body.labels,
        "description": "前景=1 背景=0. 在首帧 RGB 图上采集，供 sam2mask 阶段使用",
    }

    if body.longest_edge is not None:
        info["real_size"] = {"longest_edge": body.longest_edge}

    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info("Points saved for task '%s': %d points, longest_edge=%s",
                task_name, len(body.points), body.longest_edge)

    return {"detail": "ok", "points_count": len(body.points), "longest_edge": body.longest_edge}


# ── Auto-setup config ──────────────────────────────────────────


@router.post("/tasks/{task_name}/setup-config")
async def setup_config(task_name: str, request: Request):
    import yaml

    tasks_dir = _project_root(request) / "tasks" / task_name
    info_path = tasks_dir / "dataset_info.json"

    if not info_path.exists():
        raise HTTPException(400, "No dataset_info.json found. Upload data and save points first.")

    info = json.loads(info_path.read_text(encoding="utf-8"))

    configs_dir = _project_root(request) / "configs"
    config_path = configs_dir / "foundationpose.yaml"

    if not config_path.exists():
        raise HTTPException(404, "Config file foundationpose.yaml not found")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # Update config fields from dataset_info
    config["task"] = task_name
    config["input"]["rgbd_dir"] = f"./tasks/{task_name}/"
    config["input"]["multi_views_dir"] = f"./tasks/{task_name}/views/"

    sp = info.get("sam2_points", {})
    config["sam2"]["points"] = sp.get("points", [])
    config["sam2"]["labels"] = sp.get("labels", [])

    rs = info.get("real_size", {})
    if rs.get("longest_edge"):
        config["real_size"]["longest_edge"] = rs["longest_edge"]

    config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True),
                           encoding="utf-8")
    logger.info("Config updated for task '%s'", task_name)

    return {"detail": "ok", "config": str(config_path)}


# ── Combined save + setup (transactional) ─────────────────────


@router.post("/tasks/{task_name}/save-and-setup")
async def save_and_setup(task_name: str, request: Request):
    """Save points and update config in a single transactional call."""
    from pydantic import BaseModel
    import yaml

    class PointsBody(BaseModel):
        points: list[list[int]] = []
        labels: list[int] = []
        longest_edge: float | None = None

    data = await request.json()
    body = PointsBody(**data)

    if len(body.points) != len(body.labels):
        raise HTTPException(400, "points and labels arrays must have the same length")

    if body.longest_edge is not None and body.longest_edge <= 0:
        raise HTTPException(400, "longest_edge must be greater than 0")

    tasks_dir = _project_root(request) / "tasks" / task_name
    if not tasks_dir.is_dir():
        raise HTTPException(404, f"Task '{task_name}' not found")

    configs_dir = _project_root(request) / "configs"
    config_path = configs_dir / "foundationpose.yaml"
    if not config_path.exists():
        raise HTTPException(404, "Config file foundationpose.yaml not found")

    # Step 1: Save points to dataset_info.json
    info_path = tasks_dir / "dataset_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        info = _build_dataset_info(tasks_dir)

    info["sam2_points"] = {
        "points": body.points,
        "labels": body.labels,
        "description": "前景=1 背景=0. 在首帧 RGB 图上采集，供 sam2mask 阶段使用",
    }

    if body.longest_edge is not None:
        info["real_size"] = {"longest_edge": body.longest_edge}

    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info("Points saved for task '%s': %d points, longest_edge=%s",
                task_name, len(body.points), body.longest_edge)

    # Step 2: Update config (same transaction)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["task"] = task_name
    config["input"]["rgbd_dir"] = f"./tasks/{task_name}/"
    config["input"]["multi_views_dir"] = f"./tasks/{task_name}/views/"

    sp = info.get("sam2_points", {})
    config["sam2"]["points"] = sp.get("points", [])
    config["sam2"]["labels"] = sp.get("labels", [])

    rs = info.get("real_size", {})
    if rs.get("longest_edge"):
        config["real_size"]["longest_edge"] = rs["longest_edge"]

    config_path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True),
                           encoding="utf-8")
    logger.info("Config updated for task '%s'", task_name)

    return {
        "detail": "ok",
        "points_count": len(body.points),
        "longest_edge": body.longest_edge,
        "config": str(config_path),
    }


# ── Helpers ───────────────────────────────────────────────────


def _sanitize_name(name: str) -> str:
    if not name:
        return ""
    name = name.replace("\x00", "")
    name = name.replace("/", "_").replace("\\", "_")
    name = name.strip()
    if ".." in name:
        return ""
    return name


def _sanitize_relpath(path: str) -> str | None:
    """Validate and sanitize a relative path. Returns None if traversal detected."""
    if not path or ".." in path:
        return None
    path = path.replace("\x00", "")
    # Split and sanitize each component
    parts = path.replace("\\", "/").split("/")
    safe_parts = []
    for p in parts:
        p = p.strip()
        if not p or p in (".", ".."):
            continue
        safe_parts.append(p)
    if not safe_parts:
        return None
    return "/".join(safe_parts)


def _summarize_upload(task_dir: Path, uploaded: list) -> dict:
    """Return a summary of what's in the task directory."""
    summary = {"rgb_count": 0, "depth_count": 0, "views": [], "other": []}
    for sub in ["rgb", "depth", "views"]:
        subdir = task_dir / sub
        if not subdir.is_dir():
            continue
        if sub == "rgb":
            summary["rgb_count"] = len([f for f in os.listdir(subdir) if f.endswith(".png")])
        elif sub == "depth":
            summary["depth_count"] = len([f for f in os.listdir(subdir) if f.endswith(".png")])
        elif sub == "views":
            summary["views"] = sorted([
                f for f in os.listdir(subdir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
    # Other files at root level
    for f in sorted(os.listdir(task_dir)):
        fp = task_dir / f
        if fp.is_file():
            summary["other"].append(f)
    return summary


def _build_dataset_info(task_dir: Path) -> dict:
    """Build a basic dataset_info.json from what's on disk."""
    from datetime import datetime
    camera_params = {}
    cp_path = task_dir / "camera_params.json"
    if cp_path.exists():
        try:
            camera_params = json.loads(cp_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    rgb_dir = task_dir / "rgb"
    frame_count = 0
    if rgb_dir.is_dir():
        frame_count = len([f for f in os.listdir(rgb_dir) if f.endswith(".png")])

    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(task_dir),
        "resolution": {"width": camera_params.get("width", 640), "height": camera_params.get("height", 480)},
        "frame_count": frame_count,
        "camera": camera_params,
    }
