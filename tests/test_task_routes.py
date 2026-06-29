from pathlib import Path

from web.routes.tasks import _build_task_artifacts, _sync_task_pipeline, _task_pipeline


def test_task_pipeline_reads_annotation_dataset_from_task_yaml(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse01"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "task_id: mouse01\npipeline: annotation_dataset\n",
        encoding="utf-8",
    )

    assert _task_pipeline(task_dir) == "annotation_dataset"


def test_build_task_artifacts_includes_dataset_images_and_pipeline(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse01" / "rgb"
    task_dir.mkdir(parents=True)
    (tmp_path / "tasks" / "mouse01" / "task.yaml").write_text(
        "task_id: mouse01\npipeline: annotation_dataset\n",
        encoding="utf-8",
    )
    (task_dir / "00000.png").write_bytes(b"png")

    artifacts = _build_task_artifacts(Path(tmp_path), "mouse01")

    assert artifacts["pipeline"] == "annotation_dataset"
    assert artifacts["dataset"]["rgb"][0]["path"] == "tasks/mouse01/rgb/00000.png"
    assert artifacts["annotation"] is not None


def test_sync_task_pipeline_updates_task_yaml(tmp_path):
    task_dir = tmp_path / "tasks" / "mouse01"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "task_id: mouse01\npipeline: pose6d\n",
        encoding="utf-8",
    )

    _sync_task_pipeline(task_dir, "annotation_dataset")

    assert _task_pipeline(task_dir) == "annotation_dataset"
