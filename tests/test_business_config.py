import json

import yaml

from web.routes.configs import BusinessConfigBody, _save_business_config, _task_business_config


def _write_platform_files(root):
    (root / "configs" / "pipelines").mkdir(parents=True)
    (root / "configs" / "algorithms").mkdir(parents=True)
    (root / "configs" / "runtime").mkdir(parents=True)
    (root / "registry").mkdir()
    (root / "configs" / "pipelines" / "pose6d.yaml").write_text("""
preset: pose6d
stages:
  - masks
  - hunyuangen
  - scale
  - package
  - foundationpose
""")
    (root / "configs" / "algorithms" / "sam2.yaml").write_text("""
sam2:
  container: sam2-backend-1
  points: []
  labels: []
""")
    (root / "configs" / "algorithms" / "hunyuan3d.yaml").write_text("""
hunyuan:
  model: /home/try/.cache/huggingface/hub/models--tencent--Hunyuan3D-2mv/snapshots/local
  project_dir: /home/try/code/Hunyuan3D-2
  python: /home/try/.conda/envs/hunyuan/bin/python
  views:
    front: front.jpg
""")
    (root / "configs" / "algorithms" / "foundationpose.yaml").write_text("""
foundationpose:
  container: foundationpose
  workdir: /home/try/code/FoundationPose
""")
    (root / "configs" / "runtime" / "server.yaml").write_text("""
runtime:
  name: server
""")
    (root / "registry" / "classes.json").write_text("""
[
  {"class_id": 0, "name": "mouse"},
  {"class_id": 1, "name": "cup"}
]
""")


def test_save_business_config_writes_task_yaml_and_stage_settings(tmp_path):
    _write_platform_files(tmp_path)
    task_dir = tmp_path / "tasks" / "mouse02"
    task_dir.mkdir(parents=True)

    saved = _save_business_config(
        tmp_path,
        "mouse02",
        BusinessConfigBody(
            pipeline="pose6d",
            class_id=1,
            first_frame=2,
            longest_edge=0.25,
            points=[[10, 20], [30, 40]],
            labels=[1, 0],
            enabled_stages=["masks", "package", "foundationpose"],
        ),
    )

    task_yaml = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    settings = json.loads((task_dir / "pipeline_settings.json").read_text(encoding="utf-8"))
    dataset_info = json.loads((task_dir / "dataset_info.json").read_text(encoding="utf-8"))

    assert saved["class_id"] == 1
    assert saved["class_name"] == "cup"
    assert task_yaml["task_id"] == "mouse02"
    assert task_yaml["pipeline"] == "pose6d"
    assert task_yaml["runtime"] == "server"
    assert task_yaml["input"]["rgbd_dir"] == "./tasks/mouse02/"
    assert task_yaml["input"]["first_frame"] == 2
    assert task_yaml["real_size"]["longest_edge"] == 0.25
    assert task_yaml["sam2"]["points"] == [[10, 20], [30, 40]]
    assert settings["enabled"] == ["masks", "package", "foundationpose"]
    assert settings["skipped"] == ["hunyuangen", "scale"]
    assert dataset_info["sam2_points"]["labels"] == [1, 0]


def test_task_business_config_falls_back_to_dataset_info(tmp_path):
    _write_platform_files(tmp_path)
    task_dir = tmp_path / "tasks" / "mouse02"
    task_dir.mkdir(parents=True)
    (task_dir / "dataset_info.json").write_text(json.dumps({
        "sam2_points": {"points": [[5, 6]], "labels": [1]},
        "real_size": {"longest_edge": 0.126},
    }))

    config = _task_business_config(tmp_path, "mouse02")

    assert config["pipeline"] == "pose6d"
    assert config["points"] == [[5, 6]]
    assert config["labels"] == [1]
    assert config["longest_edge"] == 0.126
    assert config["stage_settings"]["enabled"] == [
        "masks",
        "hunyuangen",
        "scale",
        "package",
        "foundationpose",
    ]


def test_root_foundationpose_yaml_is_retired():
    from pathlib import Path

    assert not Path("configs/foundationpose.yaml").exists()
