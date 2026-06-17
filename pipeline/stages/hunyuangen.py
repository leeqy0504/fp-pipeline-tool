"""Hunyuan 3D generation stage: call a local Hunyuan3D API server."""

import base64
import logging
from pathlib import Path

import requests

from pipeline.config import PipelineConfig
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext


def _log(context: StageContext | None, level: int, message: str, *args):
    if context:
        context.log(level, message, *args)
    else:
        print("[hunyuangen] " + (message % args if args else message))


def _image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _convert_glb_to_obj(glb_path: Path, obj_path: Path) -> None:
    try:
        import trimesh
    except ImportError as exc:
        raise StageError(
            "trimesh is required to convert local Hunyuan3D GLB output to OBJ. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        mesh = trimesh.load(str(glb_path), force="scene")
        mesh.export(str(obj_path))
    except Exception as exc:
        raise StageError(f"Failed to convert GLB to OBJ: {exc}") from exc


def _pick_front_image(config: PipelineConfig, views_dir: Path) -> Path:
    front_name = config.hunyuan.views.get("front")
    if front_name:
        return views_dir / front_name

    for candidate in ("front.png", "front.jpg", "front.jpeg"):
        path = views_dir / candidate
        if path.exists():
            return path

    raise StageError("No front view configured. Set hunyuan.views.front in the pipeline config.")


@register_stage("hunyuangen")
class HunyuanGenStage(BaseStage):
    name = "hunyuangen"

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        views_dir = Path(config.input.multi_views_dir)
        self.check_input_path(str(views_dir), "Multi-view images directory")

        image_path = _pick_front_image(config, views_dir)
        self.check_input_path(str(image_path), "Front view image")

        host = config.hunyuan.api_host
        port = config.hunyuan.api_port
        timeout = config.hunyuan.api_timeout
        url = f"http://{host}:{port}/generate"

        _log(context, logging.INFO, "Calling local Hunyuan3D API: %s", url)
        _log(context, logging.INFO, "Using front image: %s", image_path)

        payload = {"image": _image_to_base64(image_path)}
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            raise StageError(f"Local Hunyuan3D API request failed: {exc}") from exc

        if response.status_code != 200:
            raise StageError(
                "Local Hunyuan3D API failed "
                f"(status={response.status_code}): {response.text[:500]}"
            )
        if not response.content:
            raise StageError("Local Hunyuan3D API returned empty content")

        glb_path = output_dir / "result.glb"
        glb_path.write_bytes(response.content)
        _log(context, logging.INFO, "Saved GLB: %s (%.2f MB)",
             glb_path, glb_path.stat().st_size / 1024 / 1024)

        raw_obj_path = output_dir / "raw.obj"
        _convert_glb_to_obj(glb_path, raw_obj_path)
        if not raw_obj_path.exists():
            raise StageError(f"OBJ conversion completed but file is missing: {raw_obj_path}")

        _log(context, logging.INFO, "Saved OBJ: %s", raw_obj_path)
        _log(context, logging.INFO, "Done: %s", output_dir)
        return output_dir
