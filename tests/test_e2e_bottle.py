"""E2E test: scale + package using bottle reference dataset.

Tests the minimal closed loop:
  1. Scale: verify scale factor math with bottle's mesh
  2. Package: produce a dataset matching bottle structure
"""

import json
import shutil
from pathlib import Path
import pytest

from pipeline.stages.scale import ScaleStage, parse_obj_bounds
from pipeline.stages.package import PackageStage
from pipeline.config import (
    PipelineConfig, InputConfig, Sam2Config, HunyuanConfig, RealSizeConfig,
)

BOTTLE_DIR = Path("/home/ubt2204/Downloads/data/bottle")


@pytest.fixture(scope="module")
def bottle_copy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bottle_test")
    dst = tmp / "bottle"
    shutil.copytree(BOTTLE_DIR, dst)
    return dst


def test_bottle_rgb_count(bottle_copy):
    rgb_files = sorted((bottle_copy / "rgb").glob("*.png"))
    assert len(rgb_files) == 56


def test_bottle_has_mesh(bottle_copy):
    mesh = bottle_copy / "mesh" / "textured_simple.obj"
    assert mesh.exists()


def test_bottle_has_mask(bottle_copy):
    mask = bottle_copy / "masks" / "00000.png"
    assert mask.exists()


def test_scale_then_package_produces_valid_fp_dataset(tmp_path, bottle_copy):
    """Full scale+package pipeline against bottle data."""
    # Prepare input: rgbd dir (bottle has rgb + depth + camera_params.json)
    rgbd_dir = tmp_path / "rgbd_input"
    rgbd_dir.mkdir()
    shutil.copytree(bottle_copy / "rgb", rgbd_dir / "rgb")
    shutil.copytree(bottle_copy / "depth", rgbd_dir / "depth")
    shutil.copy2(bottle_copy / "camera_params.json", rgbd_dir / "camera_params.json")

    output_dir = tmp_path / "output"

    config = PipelineConfig(
        task="bottle_test",
        preset="foundationpose",
        input=InputConfig(
            rgbd_dir=str(rgbd_dir),
            multi_views_dir="/nonexistent",
            first_frame=0,
        ),
        sam2=Sam2Config(container="sam2-test", points=[[0, 0], [10, 10]], labels=[1, 1]),
        hunyuan=HunyuanConfig(
            secret_id="id", secret_key="key", region="ap", model="3.1",
            face_count=500000, enable_pbr=False,
            views={"front": "f.jpg", "left": "l.jpg", "right": "r.jpg", "back": "b.jpg"},
        ),
        real_size=RealSizeConfig(longest_edge=1.0),
        output_dir=str(output_dir),
    )

    # --- Stage: Scale ---
    scale_out = output_dir / "scale"
    scale_out.mkdir(parents=True)

    original_obj = (bottle_copy / "mesh" / "textured_simple.obj").read_text()
    input_obj_dir = tmp_path / "input_obj"
    input_obj_dir.mkdir(parents=True, exist_ok=True)
    (input_obj_dir / "obj.obj").write_text(original_obj)

    scale_stage = ScaleStage()
    scale_stage._input_obj_path = str(input_obj_dir / "obj.obj")

    # Get original longest edge, then scale to same size (factor=1.0)
    min_c, max_c = parse_obj_bounds(original_obj)
    original_longest = max(max_c[i] - min_c[i] for i in range(3))

    config.real_size.longest_edge = original_longest
    scale_stage.run(config, scale_out)

    scaled_obj = (scale_out / "obj.obj").read_text()
    min_c2, max_c2 = parse_obj_bounds(scaled_obj)
    scaled_longest = max(max_c2[i] - min_c2[i] for i in range(3))
    assert scaled_longest == pytest.approx(original_longest, rel=1e-3)

    # --- Stage: Package ---
    package_out = output_dir / "package"
    package_out.mkdir(parents=True)

    package_stage = PackageStage()
    package_stage._obj_path_override = str(scale_out / "obj.obj")
    package_stage._mask_path_override = str(bottle_copy / "masks" / "00000.png")

    result = package_stage.run(config, package_out)

    # Verify output structure matches bottle exactly
    assert (result / "rgb").is_dir()
    assert (result / "depth").is_dir()
    assert (result / "masks").is_dir()
    assert (result / "mesh").is_dir()
    assert (result / "cam_K.txt").exists()
    assert (result / "camera_params.json").exists()

    # Same number of frames
    expected_count = len(sorted(rgbd_dir.glob("rgb/*.png")))
    actual_count = len(sorted(result.glob("rgb/*.png")))
    assert actual_count == expected_count

    # Camera params match
    result_params = json.loads((result / "camera_params.json").read_text())
    original_params = json.loads((bottle_copy / "camera_params.json").read_text())
    assert result_params["fx"] == pytest.approx(original_params["fx"])

    # Mask copied correctly
    assert (result / "masks" / "00000.png").exists()

    # Mesh placed correctly
    assert (result / "mesh" / "textured_simple.obj").exists()
