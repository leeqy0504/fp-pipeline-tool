import os
import pytest
from pipeline.config import PipelineConfig, load_config, ConfigError


def test_load_config_basic(tmp_path, monkeypatch):
    """Load a minimal valid config with env var substitution."""
    monkeypatch.setenv("TENCENT_SECRET_ID", "test-id")
    monkeypatch.setenv("TENCENT_SECRET_KEY", "test-key")

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
  secret_id: ${TENCENT_SECRET_ID}
  secret_key: ${TENCENT_SECRET_KEY}
  region: ap-guangzhou
  model: "3.1"
  face_count: 500000
  enable_pbr: false
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
    assert config.hunyuan.secret_id == "test-id"
    assert config.hunyuan.secret_key == "test-key"
    assert config.hunyuan.region == "ap-guangzhou"
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
    """Unresolved env var is kept as-is — stage will fail at runtime."""
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
