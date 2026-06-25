import logging

from pipeline.stages.context import DataContext, RunContext, StageContext
from pipeline.config import PipelineConfig, InputConfig, Sam2Config, HunyuanConfig, RealSizeConfig
from pipeline.manifest import Manifest
from pipeline.pipeline import PipelineOrchestrator


def test_data_context_exposes_named_inputs_and_output(tmp_path):
    data = DataContext(
        task_dir=tmp_path / "tasks" / "mouse02",
        run_dir=tmp_path / "output" / "mouse02" / "runs" / "job1",
        output_dir=tmp_path / "output" / "mouse02" / "runs" / "job1" / "stages" / "package",
        inputs={
            "masks": tmp_path / "output" / "mouse02" / "runs" / "job1" / "stages" / "masks",
            "scale": tmp_path / "output" / "mouse02" / "runs" / "job1" / "stages" / "scale",
        },
    )

    assert data.input("masks").name == "masks"
    assert data.output.name == "package"


def test_stage_context_creates_stage_scoped_logger(tmp_path):
    logger = logging.getLogger("test.stage-context")
    run = RunContext(
        run_id="job1",
        task_name="mouse02",
        logger=logger,
        resolved_config_path=tmp_path / "resolved_config.yaml",
    )
    data = DataContext(
        task_dir=tmp_path / "tasks" / "mouse02",
        run_dir=tmp_path / "output" / "mouse02" / "runs" / "job1",
        output_dir=tmp_path / "output" / "mouse02" / "runs" / "job1" / "stages" / "masks",
    )

    ctx = StageContext(run=run, data=data, stage_name="masks")

    assert ctx.stage_name == "masks"
    assert ctx.output_dir == data.output
    assert ctx.logger.name.endswith(".masks")


def test_stage_context_does_not_expose_manifest_writer(tmp_path):
    ctx = StageContext(
        run=RunContext(
            run_id="job1",
            task_name="mouse02",
            logger=logging.getLogger("test.no-manifest"),
            resolved_config_path=tmp_path / "resolved_config.yaml",
        ),
        data=DataContext(
            task_dir=tmp_path / "tasks" / "mouse02",
            run_dir=tmp_path / "output" / "mouse02" / "runs" / "job1",
            output_dir=tmp_path / "output" / "mouse02" / "runs" / "job1" / "stages" / "masks",
        ),
        stage_name="masks",
    )

    assert not hasattr(ctx, "manifest")
    assert not hasattr(ctx, "manifest_writer")


def test_orchestrator_builds_data_context_from_completed_manifest(tmp_path):
    config = PipelineConfig(
        task="mouse02",
        preset="pose6d",
        input=InputConfig(rgbd_dir=str(tmp_path / "tasks" / "mouse02"), multi_views_dir="/tmp/views"),
        sam2=Sam2Config(container="sam2", points=[[1, 2]], labels=[1]),
        hunyuan=HunyuanConfig(views={"front": "front.jpg"}),
        real_size=RealSizeConfig(longest_edge=0.12),
        output_dir=str(tmp_path / "output"),
        run_id="job1",
    )
    manifest = Manifest(task="mouse02", config_path="resolved_config.yaml", run_id="job1")
    manifest.mark_stage_done("masks", str(tmp_path / "output" / "mouse02" / "runs" / "job1" / "stages" / "masks"), 1.0)
    manifest.mark_stage_done("scale", str(tmp_path / "output" / "mouse02" / "runs" / "job1" / "stages" / "scale"), 1.0)
    base = StageContext(logger=logging.getLogger("test.pipeline"), job_id="job1")

    ctx = PipelineOrchestrator()._build_stage_context(
        config,
        "package",
        tmp_path / "output" / "mouse02" / "runs" / "job1" / "stages" / "package",
        manifest,
        base,
        str(tmp_path / "output" / "mouse02" / "runs" / "job1" / "resolved_config.yaml"),
    )

    assert ctx.data.input("masks").name == "masks"
    assert ctx.data.input("scale").name == "scale"
    assert ctx.data.output.name == "package"
    assert ctx.run.task_name == "mouse02"
