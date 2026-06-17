"""Hunyuan 3D generation stage: run the local multi-view Hunyuan3D model."""

import logging
import time
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext


def _load_hunyuan3d():
    try:
        import torch
        from PIL import Image
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        return torch, Image, BackgroundRemover, Hunyuan3DDiTFlowMatchingPipeline
    except ImportError as exc:
        raise StageError(
            "Local Hunyuan3D dependencies are not available. "
            "Run this stage in the Hunyuan3D conda environment, for example: "
            "conda activate hunyuan"
        ) from exc


def _log(context: StageContext | None, level: int, message: str, *args):
    if context:
        context.log(level, message, *args)
    else:
        print("[hunyuangen] " + (message % args if args else message))


def _load_view_images(config: PipelineConfig, views_dir: Path, Image, BackgroundRemover) -> dict:
    images = {}
    rembg = None

    for view_name, filename in config.hunyuan.views.items():
        image_path = views_dir / filename
        if not image_path.exists():
            raise StageError(f"View image ({view_name}) not found: {image_path}")

        image = Image.open(image_path)
        if config.hunyuan.remove_background and image.mode == "RGB":
            if rembg is None:
                rembg = BackgroundRemover()
            image = rembg(image)
        else:
            image = image.convert("RGBA")

        images[view_name] = image

    if "front" not in images:
        raise StageError("No 'front' view found in config — required as main image")
    return images


@register_stage("hunyuangen")
class HunyuanGenStage(BaseStage):
    name = "hunyuangen"

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        torch, Image, BackgroundRemover, HunyuanPipeline = _load_hunyuan3d()

        views_dir = Path(config.input.multi_views_dir)
        self.check_input_path(str(views_dir), "Multi-view images directory")

        images = _load_view_images(config, views_dir, Image, BackgroundRemover)
        _log(context, logging.INFO, "Loaded Hunyuan3D views: %s", ", ".join(images.keys()))
        _log(context, logging.INFO, "Loading local Hunyuan3D model: %s", config.hunyuan.model)

        try:
            pipeline = HunyuanPipeline.from_pretrained(
                config.hunyuan.model,
                subfolder=config.hunyuan.subfolder,
                variant=config.hunyuan.variant,
            )
        except Exception as exc:
            raise StageError(f"Failed to load local Hunyuan3D model: {exc}") from exc

        start_time = time.time()
        try:
            mesh = pipeline(
                image=images,
                num_inference_steps=config.hunyuan.num_inference_steps,
                octree_resolution=config.hunyuan.octree_resolution,
                num_chunks=config.hunyuan.num_chunks,
                generator=torch.manual_seed(config.hunyuan.seed),
                output_type=config.hunyuan.output_type,
            )[0]
        except Exception as exc:
            raise StageError(f"Local Hunyuan3D generation failed: {exc}") from exc

        elapsed = time.time() - start_time
        _log(context, logging.INFO, "Local Hunyuan3D generation completed in %.1fs", elapsed)

        glb_path = output_dir / "result.glb"
        raw_obj_path = output_dir / "raw.obj"
        try:
            mesh.export(str(glb_path))
            mesh.export(str(raw_obj_path))
        except Exception as exc:
            raise StageError(f"Failed to export Hunyuan3D mesh: {exc}") from exc

        if not raw_obj_path.exists():
            raise StageError(f"OBJ export completed but file is missing: {raw_obj_path}")

        _log(context, logging.INFO, "Saved GLB: %s", glb_path)
        _log(context, logging.INFO, "Saved OBJ: %s", raw_obj_path)
        _log(context, logging.INFO, "Done: %s", output_dir)
        return output_dir
