import json
import shutil
from pathlib import Path
import pytest
from pipeline.config import PipelineConfig, InputConfig, Sam2Config, HunyuanConfig, RealSizeConfig
from pipeline.stages.package import PackageStage


def make_config(rgbd_dir="/tmp/r", first_frame=0):
    return PipelineConfig(
        task="test",
        preset="foundationpose",
        input=InputConfig(rgbd_dir=rgbd_dir, multi_views_dir="/tmp/v", first_frame=first_frame),
        sam2=Sam2Config(container="sam2-test", points=[[0, 0], [10, 10]], labels=[1, 1]),
        hunyuan=HunyuanConfig(
            secret_id="id", secret_key="key", region="ap", model="3.1",
            face_count=500000, enable_pbr=False, views={"front": "f.jpg"}
        ),
        real_size=RealSizeConfig(longest_edge=5.0),
        output_dir="output/",
    )


def build_mock_rgbd_dir(base: Path, num_frames=3):
    rgb_dir = base / "rgb"
    depth_dir = base / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    for i in range(num_frames):
        (rgb_dir / f"frame_{i:04d}.png").write_text(f"rgb{i}")
        (depth_dir / f"frame_{i:04d}.png").write_text(f"depth{i}")

    cam_params = {
        "width": 640, "height": 480,
        "fx": 603.19, "fy": 602.36,
        "ppx": 322.92, "ppy": 250.40,
        "depth_scale": 0.001,
    }
    (base / "camera_params.json").write_text(json.dumps(cam_params))


def test_package_creates_fp_dataset(tmp_path):
    rgbd_dir = tmp_path / "rgbd"
    build_mock_rgbd_dir(rgbd_dir, num_frames=3)

    mask_dir = tmp_path / "mask_dir"
    mask_dir.mkdir()
    mask_file = mask_dir / "mask.png"
    mask_file.write_text("maskdata")

    obj_dir = tmp_path / "obj_dir"
    obj_dir.mkdir()
    obj_file = obj_dir / "obj.obj"
    obj_file.write_text("v 0 0 0\nv 1 1 1\nf 1 2\n")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = make_config(rgbd_dir=str(rgbd_dir), first_frame=0)
    stage = PackageStage()
    stage._mask_path_override = str(mask_file)
    stage._obj_path_override = str(obj_file)

    result = stage.run(config, output_dir)

    assert (result / "rgb").is_dir()
    assert (result / "depth").is_dir()
    assert (result / "masks").is_dir()
    assert (result / "mesh").is_dir()
    assert (result / "cam_K.txt").exists()
    assert (result / "camera_params.json").exists()

    rgb_files = sorted((result / "rgb").iterdir())
    assert rgb_files[0].name == "00000.png"
    assert rgb_files[1].name == "00001.png"
    assert rgb_files[2].name == "00002.png"

    assert (result / "masks" / "00000.png").exists()
    assert (result / "masks" / "00000.png").read_text() == "maskdata"

    assert (result / "mesh" / "textured_simple.obj").exists()
    assert (result / "mesh" / "textured_simple.obj").read_text() == "v 0 0 0\nv 1 1 1\nf 1 2\n"


def test_package_generates_cam_k(tmp_path):
    rgbd_dir = tmp_path / "rgbd"
    build_mock_rgbd_dir(rgbd_dir, num_frames=1)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    config = make_config(rgbd_dir=str(rgbd_dir))
    stage = PackageStage()
    stage._mask_path_override = str(tmp_path / "m.png")
    stage._obj_path_override = str(tmp_path / "o.obj")
    (tmp_path / "m.png").write_text("x")
    (tmp_path / "o.obj").write_text("v 0 0 0\nf 1\n")

    result = stage.run(config, output_dir)

    cam_k = (result / "cam_K.txt").read_text().strip().split("\n")
    assert len(cam_k) == 3
    assert "603.19" in cam_k[0]


def test_package_camera_params_json(tmp_path):
    rgbd_dir = tmp_path / "rgbd"
    build_mock_rgbd_dir(rgbd_dir)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    config = make_config(rgbd_dir=str(rgbd_dir))
    stage = PackageStage()
    stage._mask_path_override = str(tmp_path / "m.png")
    stage._obj_path_override = str(tmp_path / "o.obj")
    (tmp_path / "m.png").write_text("x")
    (tmp_path / "o.obj").write_text("v 0 0 0\nf 1\n")

    result = stage.run(config, output_dir)

    params = json.loads((result / "camera_params.json").read_text())
    assert params["width"] == 640
    assert params["height"] == 480
    assert params["fx"] == 603.19
    assert params["depth_scale"] == 0.001


def test_package_first_frame_selection(tmp_path):
    rgbd_dir = tmp_path / "rgbd"
    build_mock_rgbd_dir(rgbd_dir, num_frames=5)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    config = make_config(rgbd_dir=str(rgbd_dir), first_frame=0)
    stage = PackageStage()
    stage._mask_path_override = str(tmp_path / "m.png")
    stage._obj_path_override = str(tmp_path / "o.obj")
    (tmp_path / "m.png").write_text("x")
    (tmp_path / "o.obj").write_text("v 0 0 0\nf 1\n")

    result = stage.run(config, output_dir)
    rgb_files = sorted((result / "rgb").iterdir())
    assert rgb_files[0].name == "00000.png"
    assert len(rgb_files) == 5
