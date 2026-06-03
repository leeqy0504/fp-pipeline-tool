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
}
DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB, overridable via app.state.max_body_size


def _project_root(request: Request) -> Path:
    return request.app.state.project_root


# ── Upload ────────────────────────────────────────────────────


@router.post("/tasks/upload")
async def upload_files(request: Request):
    from fastapi import Form

    # FastAPI multipart handling requires explicit form fields.
    # We use request.form() directly for flexibility.
    form = await request.form()
    task_name = form.get("task_name", "")

    if not task_name:
        raise HTTPException(400, "Missing 'task_name' field")

    # Validate task name
    sanitized_task = _sanitize_name(task_name)
    if not sanitized_task:
        raise HTTPException(400, "Invalid task name")

    max_size = getattr(request.app.state, "max_body_size", DEFAULT_MAX_UPLOAD_BYTES)

    tasks_dir = _project_root(request) / "tasks"
    dest_dir = tasks_dir / sanitized_task
    dest_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    total_bytes = 0

    for field_name, value in form.multi_items():
        if not isinstance(value, UploadFile):
            continue

        filename = value.filename
        if not filename:
            continue

        # Security: path traversal rejection
        sanitized = _sanitize_filename(filename)
        if sanitized is None:
            logger.warning("Upload rejected (path traversal): %s", filename)
            raise HTTPException(400, f"Invalid filename: {filename}")

        # Security: extension allowlist
        ext = Path(sanitized).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            logger.warning("Upload rejected (extension): %s", filename)
            raise HTTPException(400, f"File type not allowed: {ext}")

        contents = await value.read()
        total_bytes += len(contents)
        if total_bytes > max_size:
            raise HTTPException(413, f"Upload exceeds {max_size // (1024 * 1024)} MB limit")

        dest_path = dest_dir / sanitized
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(contents)
        uploaded.append({"filename": sanitized, "size": len(contents)})
        logger.info("Uploaded: %s/%s (%d bytes)", sanitized_task, sanitized, len(contents))

    if not uploaded:
        raise HTTPException(400, "No files uploaded")

    return {
        "task_name": sanitized_task,
        "files": uploaded,
        "total_bytes": total_bytes,
    }


# ── Helpers ───────────────────────────────────────────────────


def _sanitize_name(name: str) -> str:
    """Remove path separators and null bytes. Returns empty string if invalid."""
    if not name:
        return ""
    name = name.replace("\x00", "")
    name = name.replace("/", "_").replace("\\", "_")
    name = name.strip()
    # Reject names that look like path traversal after sanitization
    if ".." in name:
        return ""
    return name


def _sanitize_filename(filename: str) -> str | None:
    """Reject path traversal; return basename-safe filename."""
    if not filename or ".." in filename:
        return None
    name = filename.replace("\x00", "")
    name = os.path.basename(name)
    if not name or name in (".", ".."):
        return None
    return name
