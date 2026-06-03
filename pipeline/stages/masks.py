"""SAM2 mask generation stage: runs SAM2 via a running Docker container."""

import json
import logging
import subprocess
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext


@register_stage("masks")
class Sam2MaskStage(BaseStage):
    name = "masks"

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        rgbd_dir = Path(config.input.rgbd_dir)
        self.check_input_path(str(rgbd_dir), "RGB-D directory")

        rgb_src = rgbd_dir / "rgb"
        self.check_input_path(str(rgb_src), "RGB source directory")

        rgb_files = sorted(rgb_src.glob("*.png"))
        if not rgb_files:
            raise StageError(f"No RGB images found in {rgb_src}")

        first_frame = rgb_files[config.input.first_frame]
        container = config.sam2.container
        container_image = f"/tmp/input_{first_frame.name}"
        container_output = "/tmp/output_mask.png"

        # 1. Copy first frame into container
        subprocess.run(
            ["docker", "cp", str(first_frame), f"{container}:{container_image}"],
            check=True,
        )

        # 2. Resolve points/labels — prefer dataset_info.json over config
        points = config.sam2.points
        labels = config.sam2.labels

        ds_candidates = [rgbd_dir / "dataset_info.json"]
        dataset_info_path = None
        for p in ds_candidates:
            if p.exists():
                dataset_info_path = p
                break

        if dataset_info_path:
            with open(dataset_info_path) as f:
                ds = json.load(f)
            sam2_data = ds.get("sam2_points", {})
            pts = sam2_data.get("points", [])
            lbls = sam2_data.get("labels", [])
            if pts and lbls and len(pts) == len(lbls):
                points = pts
                labels = lbls
                if context:
                    context.log(logging.INFO, "Using %d point(s) from %s", len(points), dataset_info_path)
                else:
                    print(f"[sam2mask] Using {len(points)} point(s) from {dataset_info_path}")
            else:
                if context:
                    context.log(logging.WARNING, "dataset_info.json found but sam2_points invalid, using config values")
                else:
                    print("[sam2mask] WARNING: dataset_info.json found but sam2_points invalid, using config values")
        else:
            if context:
                context.log(logging.INFO, "No dataset_info.json found, using config pts/labels")
            else:
                print("[sam2mask] No dataset_info.json found, using config pts/labels")

        points_str = " ".join(f"{x},{y}" for x, y in points)
        labels_str = " ".join(str(l) for l in labels)

        # 3. Run inference inside container
        result = subprocess.run(
            [
                "docker", "exec", container,
                "bash", "-c",
                f"PYTHONPATH=/opt/sam2/server python /opt/sam2_cli.py "
                f"--image {container_image} "
                f"--points '{points_str}' "
                f"--labels '{labels_str}' "
                f"--output {container_output} "
                f"--checkpoint {config.sam2.checkpoint} "
                f"--config {config.sam2.config_file}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise StageError(
                f"SAM2 CLI failed (exit {result.returncode}):\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        # 4. Parse JSON output from CLI
        output = json.loads(result.stdout.strip().split("\n")[-1])

        # 5. Copy mask back to host
        mask_output = output_dir / first_frame.name
        subprocess.run(
            ["docker", "cp", f"{container}:{container_output}", str(mask_output)],
            check=True,
        )

        if not mask_output.exists():
            raise StageError(f"SAM2 completed but mask not found at {mask_output}")

        return output_dir
