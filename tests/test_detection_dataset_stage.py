import json
from pathlib import Path

from pipeline.config import (
    DetectionDatasetConfig,
    HunyuanConfig,
    InputConfig,
    PipelineConfig,
    RealSizeConfig,
    Sam2Config,
)
from pipeline.manifest import Manifest
from pipeline.stages.detection_dataset import DetectionDatasetStage


def make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        task="test",
        preset="foundationpose",
        input=InputConfig(rgbd_dir=str(tmp_path / "rgbd"), multi_views_dir=str(tmp_path / "views")),
        sam2=Sam2Config(container="sam2-test", points=[[0, 0]], labels=[1]),
        hunyuan=HunyuanConfig(secret_id="id", secret_key="key"),
        real_size=RealSizeConfig(longest_edge=1.0),
        detection_dataset=DetectionDatasetConfig(class_name="mouse", class_id=0),
        output_dir=str(tmp_path / "output"),
    )


def write_fixture(tmp_path: Path, config: PipelineConfig) -> tuple[Path, Path]:
    package_dir = tmp_path / "package"
    fp_dir = tmp_path / "foundationpose"
    (package_dir / "rgb").mkdir(parents=True)
    (package_dir / "mesh").mkdir()
    (fp_dir / "ob_in_cam").mkdir(parents=True)

    for idx in range(2):
        (package_dir / "rgb" / f"{idx:05d}.png").write_text(f"rgb{idx}")
        (fp_dir / "ob_in_cam" / f"{idx:05d}.txt").write_text(
            "1 0 0 0\n"
            "0 1 0 0\n"
            "0 0 1 2\n"
            "0 0 0 1\n"
        )

    (package_dir / "mesh" / "textured_simple.obj").write_text(
        "v -0.5 -0.5 -0.5\n"
        "v 0.5 -0.5 -0.5\n"
        "v -0.5 0.5 -0.5\n"
        "v 0.5 0.5 -0.5\n"
        "v -0.5 -0.5 0.5\n"
        "v 0.5 -0.5 0.5\n"
        "v -0.5 0.5 0.5\n"
        "v 0.5 0.5 0.5\n"
    )
    (package_dir / "camera_params.json").write_text(json.dumps({
        "width": 640,
        "height": 480,
        "fx": 600.0,
        "fy": 600.0,
        "ppx": 320.0,
        "ppy": 240.0,
        "depth_scale": 0.001,
    }))

    manifest = Manifest(config.task, config.output_dir)
    manifest.mark_stage_done("package", str(package_dir), 1.0)
    manifest.mark_stage_done("foundationpose", str(fp_dir), 1.0)
    manifest.save(str(Path(config.output_dir) / config.task / "manifest.json"))
    return package_dir, fp_dir


def test_detection_dataset_stage_generates_yolo_dataset(tmp_path):
    config = make_config(tmp_path)
    write_fixture(tmp_path, config)

    output_dir = tmp_path / "det"
    result = DetectionDatasetStage().run(config, output_dir)

    assert (result / "images" / "00000.png").read_text() == "rgb0"
    assert (result / "labels" / "00000.txt").exists()
    assert (result / "dataset.yaml").read_text().splitlines()[-1] == "  0: mouse"

    label = (result / "labels" / "00000.txt").read_text().strip().split()
    assert label[0] == "0"
    assert len(label) == 5

    annotations = json.loads((result / "annotations.json").read_text())
    assert annotations["count"] == 2
    assert annotations["annotations"][0]["class_name"] == "mouse"
    assert annotations["annotations"][0]["bbox_xyxy"] == [120, 40, 520, 440]
