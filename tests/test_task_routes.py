import json
from pathlib import Path

from pipeline.manifest import Manifest
from pipeline.stages.annotation_dataset import DetectionDatasetExportStage, MaskQaStage
from pipeline.stages.context import DataContext, RunContext, StageContext
from tests.test_annotation_dataset_stage import _config, _png_bytes, _write_box_png
from web.routes.tasks import (
    RunRequest,
    _build_task_artifacts,
    _rebuild_detection_dataset_from_review,
    _sync_task_pipeline,
    _task_pipeline,
    run_task,
)


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


def test_run_task_uses_requested_pipeline_when_saved_settings_are_stale(tmp_path):
    import asyncio
    from types import SimpleNamespace

    (tmp_path / "tasks" / "mouse01").mkdir(parents=True)
    (tmp_path / "configs" / "pipelines").mkdir(parents=True)
    (tmp_path / "configs" / "runtime").mkdir(parents=True)
    (tmp_path / "configs" / "algorithms").mkdir(parents=True)
    (tmp_path / "configs" / "pipelines" / "annotation_dataset.yaml").write_text(
        "preset: annotation_dataset\n"
        "stages:\n"
        "  - prompt_mask\n"
        "  - sam2_video_propagation\n"
        "  - mask_qa\n",
        encoding="utf-8",
    )
    (tmp_path / "tasks" / "mouse01" / "pipeline_settings.json").write_text(
        json.dumps({
            "preset": "pose6d",
            "stages": ["masks", "hunyuangen"],
            "enabled": [],
            "skipped": ["masks", "hunyuangen"],
        }),
        encoding="utf-8",
    )

    class FakeScheduler:
        def __init__(self):
            self.submitted = None

        async def submit(self, task_name, preset, config_path, stage_selection=None):
            self.submitted = {
                "task_name": task_name,
                "preset": preset,
                "config_path": config_path,
                "stage_selection": stage_selection,
            }
            return "job1"

    scheduler = FakeScheduler()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        project_root=tmp_path,
        scheduler=scheduler,
    )))

    response = asyncio.run(run_task("mouse01", RunRequest(preset="annotation_dataset"), request))

    assert response == {"job_id": "job1"}
    assert scheduler.submitted["preset"] == "annotation_dataset"
    assert scheduler.submitted["stage_selection"] is None


def test_run_task_normalizes_explicit_annotation_stage_selection(tmp_path):
    import asyncio
    from types import SimpleNamespace

    (tmp_path / "tasks" / "mouse01").mkdir(parents=True)
    (tmp_path / "configs" / "pipelines").mkdir(parents=True)
    (tmp_path / "configs" / "runtime").mkdir(parents=True)
    (tmp_path / "configs" / "algorithms").mkdir(parents=True)
    (tmp_path / "configs" / "pipelines" / "annotation_dataset.yaml").write_text(
        "preset: annotation_dataset\n"
        "stages:\n"
        "  - prompt_mask\n"
        "  - sam2_video_propagation\n"
        "  - mask_qa\n",
        encoding="utf-8",
    )

    class FakeScheduler:
        def __init__(self):
            self.submitted = None

        async def submit(self, task_name, preset, config_path, stage_selection=None):
            self.submitted = {
                "task_name": task_name,
                "preset": preset,
                "config_path": config_path,
                "stage_selection": stage_selection,
            }
            return "job2"

    scheduler = FakeScheduler()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        project_root=tmp_path,
        scheduler=scheduler,
    )))

    response = asyncio.run(run_task(
        "mouse01",
        RunRequest(
            preset="annotation_dataset",
            stage_selection={
                "preset": "annotation_dataset",
                "enabled": ["mask_qa", "prompt_mask"],
            },
        ),
        request,
    ))

    assert response == {"job_id": "job2"}
    assert scheduler.submitted["preset"] == "annotation_dataset"
    assert scheduler.submitted["stage_selection"]["preset"] == "annotation_dataset"
    assert scheduler.submitted["stage_selection"]["enabled"] == ["prompt_mask", "mask_qa"]
    assert scheduler.submitted["stage_selection"]["skipped"] == ["sam2_video_propagation"]


def test_artifacts_and_review_rebuild_detection_dataset(tmp_path):
    (tmp_path / "configs" / "pipelines").mkdir(parents=True)
    (tmp_path / "configs" / "algorithms").mkdir(parents=True)
    (tmp_path / "configs" / "runtime").mkdir(parents=True)
    (tmp_path / "configs" / "pipelines" / "annotation_dataset.yaml").write_text(
        "preset: annotation_dataset\n"
        "stages:\n"
        "  - prompt_mask\n"
        "  - sam2_video_propagation\n"
        "  - mask_qa\n"
        "  - review_pack\n"
        "  - detection_dataset_export\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "algorithms" / "sam2.yaml").write_text(
        "sam2:\n  container: sam2-test\n  points: []\n  labels: []\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "runtime" / "server.yaml").write_text("runtime: {name: server}\n", encoding="utf-8")
    config = _config(tmp_path)
    task_dir = Path(config.input.rgbd_dir)
    rgb_dir = task_dir / "rgb"
    rgb_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "task_id: mouse01\n"
        "pipeline: annotation_dataset\n"
        "runtime: server\n"
        "input:\n"
        "  rgbd_dir: ./tasks/mouse01/\n"
        "  multi_views_dir: ./tasks/mouse01/views/\n"
        "sam2:\n"
        "  points: [[1, 2]]\n"
        "  labels: [1]\n"
        "real_size:\n"
        "  longest_edge: 0.1\n",
        encoding="utf-8",
    )
    for idx in range(2):
        (rgb_dir / f"{idx:05d}.png").write_bytes(_png_bytes(10, 10, [[idx] * 10 for _ in range(10)]))
    video_dir = tmp_path / "output" / "mouse01" / "runs" / "job1" / "stages" / "sam2_video_propagation"
    masks_dir = video_dir / "masks"
    masks_dir.mkdir(parents=True)
    _write_box_png(masks_dir / "00000.png", 10, 10, (2, 2, 6, 6))
    _write_box_png(masks_dir / "00001.png", 10, 10, (3, 2, 7, 6))

    run_dir = tmp_path / "output" / "mouse01" / "runs" / "job1"
    qa_dir = run_dir / "stages" / "mask_qa"
    export_dir = run_dir / "stages" / "detection_dataset_export"
    qa_ctx = StageContext(
        run=RunContext(run_id="job1", task_name="mouse01"),
        data=DataContext(task_dir=task_dir, run_dir=run_dir, output_dir=qa_dir,
                         inputs={"sam2_video_propagation": video_dir}),
        stage_name="mask_qa",
    )
    MaskQaStage().run(config, qa_dir, context=qa_ctx)
    export_ctx = StageContext(
        run=RunContext(run_id="job1", task_name="mouse01"),
        data=DataContext(task_dir=task_dir, run_dir=run_dir, output_dir=export_dir,
                         inputs={"mask_qa": qa_dir}),
        stage_name="detection_dataset_export",
    )
    DetectionDatasetExportStage().run(config, export_dir, context=export_ctx)

    manifest = Manifest("mouse01", str(task_dir / "task.yaml"), run_id="job1")
    manifest.metadata["run_dir"] = str(run_dir)
    manifest.mark_stage_done("mask_qa", str(qa_dir), 1.0)
    manifest.mark_stage_done("detection_dataset_export", str(export_dir), 1.0)
    manifest.save(str(run_dir / "manifest.json"))

    artifacts = _build_task_artifacts(tmp_path, "mouse01")
    assert artifacts["dataset"]["rgb"][0]["path"] == "tasks/mouse01/rgb/00000.png"
    assert artifacts["annotation"]["contact_sheet"]["path"].endswith("contact_sheet.svg")
    assert len(artifacts["annotation"]["frames"]) == 2
    assert artifacts["annotation"]["frames"][0]["preview"]["path"].endswith("preview/00000.svg")

    review = (qa_dir / "review_status.json")
    payload = json.loads(review.read_text())
    payload["frames"]["00001.png"]["state"] = "rejected"
    payload["frames"]["00001.png"]["manual"] = True
    review.write_text(json.dumps(payload))

    artifacts = _rebuild_detection_dataset_from_review(tmp_path, "mouse01")

    assert artifacts["annotation"]["annotations"]["count"] == 1
    rejected = [f for f in artifacts["annotation"]["frames"] if f["frame"] == "00001.png"][0]
    assert rejected["state"] == "rejected"
    assert rejected["exported"] is False
