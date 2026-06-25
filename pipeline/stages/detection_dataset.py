"""Detection dataset stage: convert 3D object poses to 2D training boxes."""

import json
import shutil
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.manifest import load_manifest_for_config
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext


def _read_vertices(obj_path: Path) -> list[tuple[float, float, float]]:
    vertices: list[tuple[float, float, float]] = []
    with open(obj_path) as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        raise StageError(f"No OBJ vertices found: {obj_path}")
    return vertices


def _bbox_corners(vertices: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    z0, z1 = min(zs), max(zs)
    return [
        (x, y, z)
        for x in (x0, x1)
        for y in (y0, y1)
        for z in (z0, z1)
    ]


def _read_matrix(path: Path) -> list[list[float]]:
    values: list[list[float]] = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                values.append([float(part) for part in stripped.split()])
    if len(values) != 4 or any(len(row) != 4 for row in values):
        raise StageError(f"Pose matrix must be 4x4: {path}")
    return values


def _transform_point(matrix: list[list[float]], point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def _project_bbox(
    corners: list[tuple[float, float, float]],
    pose: list[list[float]],
    camera: dict,
) -> tuple[int, int, int, int] | None:
    points: list[tuple[float, float]] = []
    for corner in corners:
        x, y, z = _transform_point(pose, corner)
        if z <= 1e-9:
            continue
        u = camera["fx"] * x / z + camera["ppx"]
        v = camera["fy"] * y / z + camera["ppy"]
        points.append((u, v))

    if not points:
        return None

    width = int(camera["width"])
    height = int(camera["height"])
    x1 = max(0, min(width - 1, int(min(p[0] for p in points))))
    y1 = max(0, min(height - 1, int(min(p[1] for p in points))))
    x2 = max(0, min(width - 1, int(max(p[0] for p in points))))
    y2 = max(0, min(height - 1, int(max(p[1] for p in points))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _yolo_line(class_id: int, box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


@register_stage("detection_dataset")
class DetectionDatasetStage(BaseStage):
    name = "detection_dataset"

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = None
        if context and context.data and context.data.get_input("package"):
            package_dir = str(context.input("package"))
        else:
            manifest = load_manifest_for_config(config)
            package_dir = manifest.get_output_dir("package")
        if not package_dir:
            raise StageError("No package output found in manifest - run 'package' stage first")
        if context and context.data and context.data.get_input("foundationpose"):
            fp_dir = str(context.input("foundationpose"))
        else:
            if manifest is None:
                manifest = load_manifest_for_config(config)
            fp_dir = manifest.get_output_dir("foundationpose")
        if not fp_dir:
            raise StageError("No foundationpose output found in manifest - run 'foundationpose' stage first")

        package_path = Path(package_dir)
        pose_dir = Path(fp_dir) / "ob_in_cam"
        rgb_dir = package_path / "rgb"
        mesh_file = package_path / "mesh" / "textured_simple.obj"
        camera_file = package_path / "camera_params.json"

        self.check_input_path(str(rgb_dir), "RGB directory")
        self.check_input_path(str(pose_dir), "FoundationPose ob_in_cam directory")
        self.check_input_path(str(mesh_file), "Mesh file")
        self.check_input_path(str(camera_file), "Camera params file")

        with open(camera_file) as f:
            camera = json.load(f)

        corners = _bbox_corners(_read_vertices(mesh_file))
        det_config = config.detection_dataset

        images_out = output_dir / "images"
        labels_out = output_dir / "labels"
        images_out.mkdir(exist_ok=True)
        labels_out.mkdir(exist_ok=True)

        annotations: list[dict] = []
        skipped: list[dict] = []
        rgb_files = sorted(rgb_dir.glob("*.png"))
        if not rgb_files:
            raise StageError(f"No RGB PNG images found in {rgb_dir}")

        for rgb_file in rgb_files:
            pose_file = pose_dir / f"{rgb_file.stem}.txt"
            if not pose_file.exists():
                skipped.append({"image": rgb_file.name, "reason": "missing_pose"})
                continue

            box = _project_bbox(corners, _read_matrix(pose_file), camera)
            if box is None:
                skipped.append({"image": rgb_file.name, "reason": "box_not_visible"})
                continue

            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area < det_config.min_box_area:
                skipped.append({"image": rgb_file.name, "reason": "box_too_small", "area": area})
                continue

            dst_image = images_out / rgb_file.name
            if det_config.copy_images:
                shutil.copy2(rgb_file, dst_image)
            else:
                dst_image = rgb_file

            label_file = labels_out / f"{rgb_file.stem}.txt"
            label_file.write_text(_yolo_line(det_config.class_id, box, int(camera["width"]), int(camera["height"])))

            annotations.append({
                "image": str(dst_image),
                "label": str(label_file),
                "pose": str(pose_file),
                "class_id": det_config.class_id,
                "class_name": det_config.class_name,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_area": area,
            })

        if not annotations:
            raise StageError("No valid detection boxes were generated")

        dataset_yaml = (
            f"path: {output_dir.resolve()}\n"
            "train: images\n"
            "val: images\n"
            "names:\n"
            f"  {det_config.class_id}: {det_config.class_name}\n"
        )
        (output_dir / "dataset.yaml").write_text(dataset_yaml)
        with open(output_dir / "annotations.json", "w") as f:
            json.dump({
                "task": config.task,
                "format": "yolo",
                "class_id": det_config.class_id,
                "class_name": det_config.class_name,
                "count": len(annotations),
                "skipped": skipped,
                "annotations": annotations,
            }, f, indent=2)

        return output_dir
