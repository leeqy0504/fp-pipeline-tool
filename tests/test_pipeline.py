import pytest
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


def test_pipeline_manifest_path_uses_run_dir_when_present(tmp_path):
    config = make_config(rgbd_dir=str(tmp_path))
    config.task = "my_task"
    config.output_dir = str(tmp_path / "output")
    config.run_id = "job123"

    orch = PipelineOrchestrator()

    assert orch._run_dir(config) == str(tmp_path / "output" / "my_task" / "runs" / "job123")
    assert orch._manifest_path(config) == str(tmp_path / "output" / "my_task" / "runs" / "job123" / "manifest.json")
    assert orch._stage_output_dir(config, "masks") == str(tmp_path / "output" / "my_task" / "runs" / "job123" / "stages" / "masks")


def test_scheduler_enabled_stages_filters_in_preset_order():
    from web.scheduler import Scheduler

    stages = ["masks", "hunyuangen", "scale", "package"]
    selection = {"enabled": ["package", "masks"]}

    assert Scheduler._enabled_stages(stages, selection) == ["masks", "package"]


def test_scheduler_enabled_stages_accepts_skipped_only():
    from web.scheduler import Scheduler

    stages = ["masks", "hunyuangen", "scale", "package"]
    selection = {"skipped": ["scale"]}

    assert Scheduler._enabled_stages(stages, selection) == ["masks", "hunyuangen", "package"]


def test_load_stage_settings_reads_saved_json(tmp_path):
    from web.routes.tasks import _load_stage_settings

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "pipeline_settings.json").write_text(
        '{"preset":"pose6d","enabled":["masks"],"skipped":["scale"]}',
        encoding="utf-8",
    )

    settings = _load_stage_settings(task_dir)

    assert settings["preset"] == "pose6d"
    assert settings["enabled"] == ["masks"]


def test_ensure_task_yaml_uses_dataset_info(tmp_path):
    from pipeline.config import load_config
    from web.routes.tasks import _ensure_task_yaml
    from tests.test_business_config import _write_platform_files

    _write_platform_files(tmp_path)
    task_dir = tmp_path / "tasks" / "mouse02"
    task_dir.mkdir(parents=True)
    (task_dir / "dataset_info.json").write_text(
        '{"sam2_points":{"points":[[9,8]],"labels":[1]},"real_size":{"longest_edge":0.42}}',
        encoding="utf-8",
    )

    task_yaml = _ensure_task_yaml(tmp_path, "mouse02")
    config = load_config(str(task_yaml), project_root=tmp_path)

    assert task_yaml.name == "task.yaml"
    assert config.task == "mouse02"
    assert config.preset == "pose6d"
    assert config.sam2.points == [[9, 8]]
    assert config.real_size.longest_edge == 0.42
    assert config.hunyuan.project_dir == "/home/try/code/Hunyuan3D-2"


def test_scheduler_requires_task_config_when_no_config_path():
    from web.scheduler import Scheduler

    try:
        Scheduler._resolve_config_path(None)
    except ValueError as exc:
        assert "task.yaml" in str(exc)
    else:
        raise AssertionError("Expected missing config path to fail")
