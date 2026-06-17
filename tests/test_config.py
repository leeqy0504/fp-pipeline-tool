import pytest
from pipeline.config import load_config, ConfigError


def test_load_config_basic(tmp_path):
    """Load a minimal valid config with env var substitution."""
    yaml_content = """
task: test_task
preset: foundationpose
input:
  rgbd_dir: /tmp/rgbd
  multi_views_dir: /tmp/views
  first_frame: 0
sam2:
  container: sam2-backend-1
  points: [[10, 20], [30, 40]]
  labels: [1, 1]
hunyuan:
  model: tencent/Hunyuan3D-2mv
  subfolder: hunyuan3d-dit-v2-mv
  variant: fp16
  num_inference_steps: 50
  octree_resolution: 380
  num_chunks: 20000
  seed: 12345
  output_type: trimesh
  remove_background: true
  views:
    front: front.jpg
    left: left.jpg
    right: right.jpg
    back: back.jpg
real_size:
  longest_edge: 5.2
output_dir: output/
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)

    config = load_config(str(config_path))

    assert config.task == "test_task"
    assert config.preset == "foundationpose"
    assert config.input.rgbd_dir == "/tmp/rgbd"
    assert config.input.multi_views_dir == "/tmp/views"
    assert config.input.first_frame == 0
    assert config.sam2.container == "sam2-backend-1"
    assert config.sam2.points == [[10, 20], [30, 40]]
    assert config.sam2.labels == [1, 1]
    assert config.hunyuan.model == "tencent/Hunyuan3D-2mv"
    assert config.hunyuan.subfolder == "hunyuan3d-dit-v2-mv"
    assert config.hunyuan.variant == "fp16"
    assert config.hunyuan.num_inference_steps == 50
    assert config.hunyuan.octree_resolution == 380
    assert config.hunyuan.num_chunks == 20000
    assert config.hunyuan.seed == 12345
    assert config.hunyuan.output_type == "trimesh"
    assert config.hunyuan.remove_background is True
    assert config.hunyuan.views == {"front": "front.jpg", "left": "left.jpg", "right": "right.jpg", "back": "back.jpg"}
    assert config.real_size.longest_edge == 5.2
    assert config.output_dir == "output/"


def test_load_config_missing_required_field(tmp_path):
    """Missing required field raises ConfigError."""
    yaml_content = """
preset: foundationpose
input:
  rgbd_dir: /tmp/rgbd
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)

    with pytest.raises(ConfigError, match="Missing required field"):
        load_config(str(config_path))


def test_load_config_env_var_not_set(tmp_path):
    """Legacy secret fields are still loaded for backward compatibility."""
    yaml_content = """
task: test
preset: foundationpose
input:
  rgbd_dir: /tmp/rgbd
  multi_views_dir: /tmp/views
  first_frame: 0
sam2:
  container: sam2-backend-1
  points: [[10, 20], [30, 40]]
  labels: [1, 1]
hunyuan:
  secret_id: ${MISSING_VAR}
  secret_key: sk
  region: ap-guangzhou
  model: "3.1"
  face_count: 500000
  enable_pbr: false
  views:
    front: f.jpg
    left: l.jpg
    right: r.jpg
    back: b.jpg
real_size:
  longest_edge: 5.2
output_dir: output/
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)

    config = load_config(str(config_path))
    assert config.hunyuan.secret_id == "${MISSING_VAR}"


def test_load_config_hunyuan_local_defaults(tmp_path):
    yaml_content = """
task: test
preset: foundationpose
input:
  rgbd_dir: /tmp/rgbd
  multi_views_dir: /tmp/views
sam2:
  container: sam2-backend-1
  points: [[10, 20]]
  labels: [1]
hunyuan:
  views:
    front: front.jpg
output_dir: output/
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)

    config = load_config(str(config_path))

    assert config.hunyuan.model == "tencent/Hunyuan3D-2mv"
    assert config.hunyuan.subfolder == "hunyuan3d-dit-v2-mv"
    assert config.hunyuan.variant == "fp16"
    assert config.hunyuan.num_inference_steps == 50
    assert config.hunyuan.octree_resolution == 380
    assert config.hunyuan.num_chunks == 20000
    assert config.hunyuan.seed == 12345
    assert config.hunyuan.output_type == "trimesh"
    assert config.hunyuan.remove_background is True


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_load_config_without_hunyuan_and_real_size(tmp_path):
    """Config loads without hunyuan/real_size sections — only sam2 needs them."""
    yaml_content = """
task: test
preset: foundationpose
input:
  rgbd_dir: /tmp/rgbd
  multi_views_dir: /tmp/views
sam2:
  container: sam2-backend-1
  points: [[10, 20]]
  labels: [1]
output_dir: output/
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)

    config = load_config(str(config_path))
    assert config.task == "test"
    assert config.sam2.container == "sam2-backend-1"
    assert config.hunyuan.secret_id == ""
    assert config.real_size.longest_edge == 1.0
