"""SAM2 video propagation stage."""

import json
import logging
import subprocess
from pathlib import Path
from shlex import quote

from pipeline.config import PipelineConfig
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext


def _resolve_points(config: PipelineConfig) -> tuple[list[list[int]], list[int]]:
    rgbd_dir = Path(config.input.rgbd_dir)
    info_path = rgbd_dir / "dataset_info.json"
    points = config.sam2.points
    labels = config.sam2.labels
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        sam2_points = info.get("sam2_points", {})
        pts = sam2_points.get("points", [])
        lbls = sam2_points.get("labels", [])
        if pts and lbls and len(pts) == len(lbls):
            points = pts
            labels = lbls
    if not points or not labels or len(points) != len(labels):
        raise StageError("SAM2 points/labels are required for video propagation")
    return points, labels


def _points_arg(points: list[list[int]]) -> str:
    return " ".join(f"{int(x)},{int(y)}" for x, y in points)


def _labels_arg(labels: list[int]) -> str:
    return " ".join(str(int(label)) for label in labels)


def _container_path(host_path: Path, project_root: Path, project_mount: str) -> str:
    """Map a host path under the project root to the SAM2 container mount path."""
    host_abs = host_path.resolve()
    root_abs = project_root.resolve()
    try:
        relative = host_abs.relative_to(root_abs)
    except ValueError:
        return str(host_abs)
    mount_root = project_mount.rstrip("/")
    return f"{mount_root}/{relative.as_posix()}"


@register_stage("sam2_video_propagation")
class Sam2VideoPropagationStage(BaseStage):
    name = "sam2_video_propagation"

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        rgbd_dir = Path(config.input.rgbd_dir)
        rgb_dir = rgbd_dir / "rgb"
        self.check_input_path(str(rgb_dir), "RGB sequence directory")
        frames = sorted(rgb_dir.glob("*.png"))
        if not frames:
            raise StageError(f"No PNG frames found in {rgb_dir}")
        if config.input.first_frame >= len(frames):
            raise StageError(
                f"first_frame {config.input.first_frame} is outside frame count {len(frames)}")

        points, labels = _resolve_points(config)
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        prompt_mask = None
        if context and context.data and context.data.get_input("prompt_mask"):
            prompt_dir = context.input("prompt_mask")
            candidate = prompt_dir / frames[config.input.first_frame].name
            if candidate.exists():
                prompt_mask = candidate

        project_root = Path(__file__).resolve().parents[2]
        script_host = project_root / config.sam2.video_cli
        self.check_input_path(str(script_host), "SAM2 video CLI")
        script_container = _container_path(script_host, project_root, config.sam2.project_mount)
        rgb_dir_container = _container_path(rgb_dir, project_root, config.sam2.project_mount)
        masks_dir_container = _container_path(masks_dir, project_root, config.sam2.project_mount)

        cmd = (
            f"PYTHONPATH=/opt/sam2/server python {quote(script_container)} "
            f"--video-dir {quote(rgb_dir_container)} "
            f"--points {quote(_points_arg(points))} "
            f"--labels {quote(_labels_arg(labels))} "
            f"--output-dir {quote(masks_dir_container)} "
            f"--first-frame {int(config.input.first_frame)} "
            f"--checkpoint {quote(config.sam2.checkpoint)} "
            f"--config {quote(config.sam2.config_file)}"
        )
        if prompt_mask:
            prompt_mask_container = _container_path(prompt_mask, project_root, config.sam2.project_mount)
            cmd += f" --prompt-mask {quote(prompt_mask_container)}"

        if context:
            context.log(logging.INFO, "Running SAM2 video propagation for %d frame(s)", len(frames))

        result = subprocess.run(
            ["docker", "exec", config.sam2.container, "bash", "-c", cmd],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise StageError(
                f"SAM2 video propagation failed (exit {result.returncode}):\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        payload = {}
        if result.stdout.strip():
            payload = json.loads(result.stdout.strip().split("\n")[-1])

        mask_files = sorted(masks_dir.glob("*.png"))
        if not mask_files:
            raise StageError(f"SAM2 video propagation completed but no masks found in {masks_dir}")

        metadata = {
            "frame_count": int(payload.get("frame_count", len(frames))),
            "mask_count": int(payload.get("mask_count", len(mask_files))),
            "first_frame": config.input.first_frame,
            "points": points,
            "labels": labels,
            "rgb_dir": str(rgb_dir),
            "masks_dir": str(masks_dir),
            "prompt_mask": str(prompt_mask) if prompt_mask else None,
            "prompt_mode": payload.get("prompt_mode", "points"),
            "checkpoint": config.sam2.checkpoint,
            "config_file": config.sam2.config_file,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_dir
