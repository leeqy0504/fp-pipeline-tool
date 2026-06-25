"""Package stage: assemble FP dataset from RGB-D, mask, and OBJ."""

import json
import shutil
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.manifest import load_manifest_for_config
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext


def _generate_cam_k_txt(fx: float, fy: float, ppx: float, ppy: float) -> str:
    """Generate 3x3 camera intrinsics matrix in cam_K.txt format."""
    return (
        f"{fx:.6f} 0.000000 {ppx:.6f}\n"
        f"0.000000 {fy:.6f} {ppy:.6f}\n"
        f"0.000000 0.000000 1.000000\n"
    )


@register_stage("package")
class PackageStage(BaseStage):
    name = "package"

    _mask_path_override: str | None = None
    _obj_path_override: str | None = None

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        rgbd_dir = Path(config.input.rgbd_dir)
        self.check_input_path(str(rgbd_dir), "RGB-D directory")

        rgb_src = rgbd_dir / "rgb"
        depth_src = rgbd_dir / "depth"
        cam_src = rgbd_dir / "camera_params.json"
        self.check_input_path(str(rgb_src), "RGB source directory")
        self.check_input_path(str(depth_src), "Depth source directory")
        self.check_input_path(str(cam_src), "Camera params file")

        with open(cam_src) as f:
            cam_data = json.load(f)

        if self._mask_path_override:
            mask_src = Path(self._mask_path_override)
        else:
            mask_dir = None
            if context and context.data and context.data.get_input("masks"):
                mask_dir = str(context.input("masks"))
            else:
                manifest = load_manifest_for_config(config)
                mask_dir = manifest.get_output_dir("masks")
            if not mask_dir:
                raise StageError("No masks output found in manifest")
            mask_files = sorted(Path(mask_dir).glob("*.png"))
            if not mask_files:
                raise StageError(f"No mask PNG found in {mask_dir}")
            mask_src = mask_files[0]
        self.check_input_path(str(mask_src), "Mask file")

        if self._obj_path_override:
            obj_src = Path(self._obj_path_override)
        else:
            scale_dir = None
            if context and context.data and context.data.get_input("scale"):
                scale_dir = str(context.input("scale"))
            else:
                manifest = load_manifest_for_config(config)
                scale_dir = manifest.get_output_dir("scale")
            if not scale_dir:
                raise StageError("No scale output found in manifest")
            obj_src = Path(scale_dir) / "scaled.obj"
        self.check_input_path(str(obj_src), "Scaled OBJ file")

        # Copy and rename RGB frames
        out_rgb = output_dir / "rgb"
        out_rgb.mkdir(exist_ok=True)

        rgb_files = sorted(rgb_src.glob("*.png"))
        if not rgb_files:
            raise StageError(f"No PNG images found in {rgb_src}")

        for idx, src_file in enumerate(rgb_files):
            dst = out_rgb / f"{idx:05d}.png"
            shutil.copy2(src_file, dst)

        # Copy and rename Depth frames
        out_depth = output_dir / "depth"
        out_depth.mkdir(exist_ok=True)

        depth_files = sorted(depth_src.glob("*.png"))
        for idx, src_file in enumerate(depth_files):
            dst = out_depth / f"{idx:05d}.png"
            shutil.copy2(src_file, dst)

        # Copy mask as first-frame mask
        out_mask = output_dir / "masks"
        out_mask.mkdir(exist_ok=True)
        mask_dst = out_mask / f"{config.input.first_frame:05d}.png"
        shutil.copy2(mask_src, mask_dst)

        # Copy scaled OBJ as textured_simple.obj
        out_mesh = output_dir / "mesh"
        out_mesh.mkdir(exist_ok=True)
        shutil.copy2(obj_src, out_mesh / "textured_simple.obj")

        # Copy cam_K.txt directly from source, no reformatting
        cam_k_src = rgbd_dir / "cam_K.txt"
        if cam_k_src.exists():
            shutil.copy2(cam_k_src, output_dir / "cam_K.txt")
        else:
            cam_k = _generate_cam_k_txt(
                cam_data["fx"], cam_data["fy"],
                cam_data["ppx"], cam_data["ppy"],
            )
            (output_dir / "cam_K.txt").write_text(cam_k)

        out_cam_params = {
            "width": cam_data["width"],
            "height": cam_data["height"],
            "fx": cam_data["fx"],
            "fy": cam_data["fy"],
            "ppx": cam_data["ppx"],
            "ppy": cam_data["ppy"],
            "depth_scale": cam_data["depth_scale"],
        }
        with open(output_dir / "camera_params.json", "w") as f:
            json.dump(out_cam_params, f, indent=4)

        return output_dir
