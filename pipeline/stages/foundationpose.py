"""FoundationPose stage: run pose estimation in Docker container."""

import logging
import subprocess
import time
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.manifest import Manifest
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext


@register_stage("foundationpose")
class FoundationPoseStage(BaseStage):
    name = "foundationpose"

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = Path(config.output_dir) / config.task / "manifest.json"
        if manifest_path.exists():
            manifest = Manifest.load(str(manifest_path))
        else:
            manifest = Manifest(config.task, config.output_dir)

        package_dir = manifest.get_output_dir("package")
        if not package_dir:
            raise StageError(
                "No package output found in manifest — run 'package' stage first"
            )

        package_path = Path(package_dir)
        mesh_file = package_path / "mesh" / "textured_simple.obj"
        if not mesh_file.exists():
            raise StageError(f"Mesh file not found: {mesh_file}")

        masks_dir = package_path / "masks"
        if not masks_dir.exists() or not any(masks_dir.glob("*.png")):
            raise StageError(f"No mask files found in {masks_dir}")

        fp_config = config.foundationpose
        container = fp_config.container
        workdir = fp_config.workdir

        # Ensure container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={container}",
             "--filter", "status=running", "-q"],
            capture_output=True, text=True,
        )
        if not result.stdout.strip():
            if context:
                context.log(logging.INFO, "Container '%s' not running, starting...", container)
            else:
                print(f"[foundationpose] Container '{container}' not running, starting...")
            run_script = str(Path(workdir) / "docker" / "run_container.sh")
            subprocess.run(
                ["bash", run_script],
                cwd=workdir, check=True,
            )
            if context:
                context.log(logging.INFO, "Waiting 10s for container to be ready...")
            else:
                print("[foundationpose] Waiting 10s for container to be ready...")
            time.sleep(10)

            verify = subprocess.run(
                ["docker", "ps", "--filter", f"name={container}",
                 "--filter", "status=running", "-q"],
                capture_output=True, text=True,
            )
            if not verify.stdout.strip():
                raise StageError(
                    f"Container '{container}' failed to start"
                )

        # Resolve to absolute paths — container mounts /home:/home
        mesh_file_abs = mesh_file.resolve()
        package_path_abs = package_path.resolve()
        output_dir_abs = output_dir.resolve()

        # Run inference
        cmd = (
            f"cd {workdir} && "
            f"python run_demo.py "
            f"--mesh_file {mesh_file_abs} "
            f"--test_scene_dir {package_path_abs} "
            f"--debug 2 "#{fp_config.debug}
            f"--debug_dir {output_dir_abs}"
        )
        if context:
            context.log(logging.INFO, "Running inference...")
        else:
            print("[foundationpose] Running inference...")
        result = subprocess.run(
            ["docker", "exec", container, "bash", "-c", cmd],
        )
        if result.returncode != 0:
            raise StageError(
                f"FoundationPose inference failed (exit {result.returncode})"
            )

        # Validate: ob_in_cam/ txt count == rgb frame count
        ob_in_cam = output_dir / "ob_in_cam"
        if not ob_in_cam.exists():
            raise StageError(
                f"Output not found after inference: {ob_in_cam}"
            )

        txt_files = sorted(ob_in_cam.glob("*.txt"))
        rgb_files = sorted((package_path / "rgb").glob("*.png"))

        if len(txt_files) != len(rgb_files):
            raise StageError(
                f"Pose/frame count mismatch: "
                f"{len(txt_files)} poses vs {len(rgb_files)} RGB frames"
            )

        if context:
            context.log(logging.INFO, "Done: %d poses -> %s", len(txt_files), ob_in_cam)
        else:
            print(f"[foundationpose] Done: {len(txt_files)} poses -> {ob_in_cam}")
        return output_dir
