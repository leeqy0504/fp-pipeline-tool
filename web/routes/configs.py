import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

ALLOWED_EXTENSIONS = {".yaml", ".yml"}


class SaveConfigBody(BaseModel):
    content: str


def _project_root(request: Request) -> Path:
    return request.app.state.project_root


def _configs_dir(request: Request) -> Path:
    return _project_root(request) / "configs"


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
