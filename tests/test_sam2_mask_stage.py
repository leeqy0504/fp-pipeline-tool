import json
from pathlib import Path

from pipeline.config import (
    HunyuanConfig,
    InputConfig,
    PipelineConfig,
    RealSizeConfig,
    Sam2Config,
)
from pipeline.stages.masks import Sam2MaskStage


def test_sam2_mask_stage_uses_mounted_pic_cli_without_docker_cp(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    task_dir = project_root / "tasks" / "__sam2_pic_test__"
    rgb_dir = task_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    (rgb_dir / "00000.png").write_bytes(b"png")
    (task_dir / "dataset_info.json").write_text(json.dumps({
        "sam2_points": {"points": [[10, 20]], "labels": [1]},
    }))
    output_dir = project_root / "output" / "__sam2_pic_test__" / "runs" / "job1" / "stages" / "prompt_mask"
    config = PipelineConfig(
        task="__sam2_pic_test__",
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
            pic_cli="tools/sam2/sam2_pic_cli.py",
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
        assert cmd[:2] != ["docker", "cp"]
        if cmd[:2] == ["docker", "exec"]:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "00000.png").write_bytes(b"mask")
            return type("Result", (), {
                "returncode": 0,
                "stdout": json.dumps({"score": 0.9, "foreground_pixels": 12, "mask_shape": [10, 10]}) + "\n",
                "stderr": "",
            })()
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("pipeline.stages.masks.subprocess.run", fake_run)
    try:
        Sam2MaskStage().run(config, output_dir)
        exec_cmd = next(cmd for cmd, _ in calls if cmd[:2] == ["docker", "exec"])
        shell = exec_cmd[-1]
        assert "/workspace/fp-pipeline-tool/tools/sam2/sam2_pic_cli.py" in shell
        assert "--image /workspace/fp-pipeline-tool/tasks/__sam2_pic_test__/rgb/00000.png" in shell
        assert "--output /workspace/fp-pipeline-tool/output/__sam2_pic_test__/runs/job1/stages/prompt_mask/00000.png" in shell
        metadata = json.loads((output_dir / "metadata.json").read_text())
        assert metadata["foreground_pixels"] == 12
    finally:
        for path in [
            output_dir / "metadata.json",
            output_dir / "00000.png",
            output_dir,
            output_dir.parent,
            output_dir.parent.parent,
            output_dir.parent.parent.parent,
            output_dir.parent.parent.parent.parent,
            task_dir / "dataset_info.json",
            rgb_dir / "00000.png",
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
