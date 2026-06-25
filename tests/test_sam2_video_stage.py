import json
from pathlib import Path

from pipeline.config import (
    HunyuanConfig,
    InputConfig,
    PipelineConfig,
    RealSizeConfig,
    Sam2Config,
)
from pipeline.stages.context import DataContext, RunContext, StageContext
from pipeline.stages.sam2_video import Sam2VideoPropagationStage, _container_path


def _make_config(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse02"
    rgb_dir = task_dir / "rgb"
    rgb_dir.mkdir(parents=True)
    for idx in range(3):
        (rgb_dir / f"{idx:05d}.png").write_bytes(b"png")
    (task_dir / "dataset_info.json").write_text(json.dumps({
        "sam2_points": {"points": [[10, 20]], "labels": [1]},
    }))
    return PipelineConfig(
        task="mouse02",
        preset="annotation_dataset",
        input=InputConfig(
            rgbd_dir=str(task_dir),
            multi_views_dir=str(task_dir / "views"),
            first_frame=0,
        ),
        sam2=Sam2Config(
            container="sam2-test",
            checkpoint="/opt/sam2/checkpoints/model.pt",
            config_file="configs/sam2.yaml",
            project_mount="/mnt/fp-pipeline-tool",
            video_cli="tools/sam2/sam2_video_cli.py",
            points=[],
            labels=[],
        ),
        hunyuan=HunyuanConfig(views={"front": "front.jpg"}),
        real_size=RealSizeConfig(longest_edge=0.1),
        output_dir=str(tmp_path / "output"),
        run_id="job1",
    )


def test_sam2_video_propagation_builds_docker_command(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    output_dir = tmp_path / "output" / "mouse02" / "runs" / "job1" / "stages" / "sam2_video_propagation"
    prompt_dir = tmp_path / "output" / "mouse02" / "runs" / "job1" / "stages" / "prompt_mask"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "00000.png").write_bytes(b"mask")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        assert cmd[:2] != ["docker", "cp"]
        if cmd[:2] == ["docker", "exec"]:
            masks_dir = output_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(3):
                (masks_dir / f"{idx:05d}.png").write_bytes(b"mask")
            return type("Result", (), {
                "returncode": 0,
                "stdout": json.dumps({"frame_count": 3, "mask_count": 3, "prompt_mode": "mask"}) + "\n",
                "stderr": "",
            })()
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("pipeline.stages.sam2_video.subprocess.run", fake_run)
    context = StageContext(
        run=RunContext(run_id="job1", task_name="mouse02"),
        data=DataContext(
            task_dir=Path(config.input.rgbd_dir),
            run_dir=tmp_path / "output" / "mouse02" / "runs" / "job1",
            output_dir=output_dir,
            inputs={"prompt_mask": prompt_dir},
        ),
        stage_name="sam2_video_propagation",
    )

    result = Sam2VideoPropagationStage().run(config, output_dir, context=context)

    assert result == output_dir
    exec_cmd = next(cmd for cmd, _ in calls if cmd[:2] == ["docker", "exec"])
    shell = exec_cmd[-1]
    assert "/mnt/fp-pipeline-tool/tools/sam2/sam2_video_cli.py" in shell
    assert "--video-dir" in shell
    assert str(Path(config.input.rgbd_dir) / "rgb") in shell
    assert "--prompt-mask" in shell
    assert str(prompt_dir / "00000.png") in shell
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["frame_count"] == 3
    assert metadata["mask_count"] == 3
    assert metadata["points"] == [[10, 20]]
    assert metadata["prompt_mode"] == "mask"


def test_container_path_maps_project_files_to_configured_mount(tmp_path):
    project_root = tmp_path / "fp-pipeline-tool"
    host_path = project_root / "tasks" / "mouse02" / "rgb"
    host_path.mkdir(parents=True)

    mapped = _container_path(
        host_path=host_path,
        project_root=project_root,
        project_mount="/workspace/fp-pipeline-tool",
    )

    assert mapped == "/workspace/fp-pipeline-tool/tasks/mouse02/rgb"


def test_sam2_video_command_maps_project_paths_to_container_mount(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    task_dir = project_root / "tasks" / "__sam2_video_test__"
    rgb_dir = task_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(2):
        (rgb_dir / f"{idx:05d}.png").write_bytes(b"png")
    (task_dir / "dataset_info.json").write_text(json.dumps({
        "sam2_points": {"points": [[10, 20]], "labels": [1]},
    }))
    output_dir = project_root / "output" / "__sam2_video_test__" / "runs" / "job1" / "stages" / "sam2_video_propagation"
    config = PipelineConfig(
        task="__sam2_video_test__",
        preset="annotation_dataset",
        input=InputConfig(
            rgbd_dir=str(task_dir),
            multi_views_dir=str(task_dir / "views"),
            first_frame=0,
        ),
        sam2=Sam2Config(
            container="sam2-test",
            checkpoint="/opt/sam2/checkpoints/model.pt",
            config_file="configs/sam2.yaml",
            project_mount="/workspace/fp-pipeline-tool",
            video_cli="tools/sam2/sam2_video_cli.py",
            points=[],
            labels=[],
        ),
        hunyuan=HunyuanConfig(views={"front": "front.jpg"}),
        real_size=RealSizeConfig(longest_edge=0.1),
        output_dir=str(project_root / "output"),
        run_id="job1",
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            (masks_dir / f"{idx:05d}.png").write_bytes(b"mask")
        return type("Result", (), {
            "returncode": 0,
            "stdout": json.dumps({"frame_count": 2, "mask_count": 2}) + "\n",
            "stderr": "",
        })()

    monkeypatch.setattr("pipeline.stages.sam2_video.subprocess.run", fake_run)
    try:
        Sam2VideoPropagationStage().run(config, output_dir)
        exec_cmd = next(cmd for cmd, _ in calls if cmd[:2] == ["docker", "exec"])
        shell = exec_cmd[-1]
        assert "--video-dir /workspace/fp-pipeline-tool/tasks/__sam2_video_test__/rgb" in shell
        assert "--output-dir /workspace/fp-pipeline-tool/output/__sam2_video_test__/runs/job1/stages/sam2_video_propagation/masks" in shell
    finally:
        for path in [
            output_dir / "metadata.json",
            output_dir / "masks" / "00000.png",
            output_dir / "masks" / "00001.png",
            output_dir / "masks",
            output_dir,
            output_dir.parent,
            output_dir.parent.parent,
            output_dir.parent.parent.parent,
            output_dir.parent.parent.parent.parent,
            task_dir / "dataset_info.json",
            rgb_dir / "00000.png",
            rgb_dir / "00001.png",
            rgb_dir,
            task_dir,
        ]:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
