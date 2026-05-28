import pytest
from pathlib import Path
from pipeline.pipeline import PipelineOrchestrator
from pipeline.config import PipelineConfig, InputConfig, Sam2Config, HunyuanConfig, RealSizeConfig


def make_config(rgbd_dir="/tmp/r"):
    return PipelineConfig(
        task="test",
        preset="foundationpose",
        input=InputConfig(rgbd_dir=rgbd_dir, multi_views_dir="/tmp/v", first_frame=0),
        sam2=Sam2Config(container="sam2-test", points=[[0, 0], [10, 10]], labels=[1, 1]),
        hunyuan=HunyuanConfig(
            secret_id="id", secret_key="key", region="ap", model="3.1",
            face_count=500000, enable_pbr=False, views={"front": "f.jpg"}
        ),
        real_size=RealSizeConfig(longest_edge=5.0),
        output_dir="output/",
    )


def test_pipeline_resolve_preset(tmp_path):
    presets_yaml = tmp_path / "presets.yaml"
    presets_yaml.write_text("""
test_preset:
  stages:
    - stage_a
    - stage_b
""")
    orch = PipelineOrchestrator(presets_path=str(presets_yaml))
    stages = orch.resolve_preset("test_preset")
    assert stages == ["stage_a", "stage_b"]


def test_pipeline_unknown_preset(tmp_path):
    presets_yaml = tmp_path / "presets.yaml"
    presets_yaml.write_text("known: {stages: [a]}")
    orch = PipelineOrchestrator(presets_path=str(presets_yaml))

    with pytest.raises(KeyError, match="Unknown preset"):
        orch.resolve_preset("nonexistent")


def test_pipeline_manifest_path(tmp_path):
    config = make_config(rgbd_dir=str(tmp_path))
    config.task = "my_task"
    config.output_dir = str(tmp_path / "output")

    orch = PipelineOrchestrator()
    path = orch._manifest_path(config)
    assert "my_task/manifest.json" in path
