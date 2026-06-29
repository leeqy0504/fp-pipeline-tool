import json
import zlib
from pathlib import Path

from pipeline.config import (
    DetectionDatasetConfig,
    HunyuanConfig,
    InputConfig,
    PipelineConfig,
    RealSizeConfig,
    Sam2Config,
)
from pipeline.stages.annotation_dataset import (
    DetectionDatasetExportStage,
    MaskQaStage,
    ReviewPackStage,
)
from pipeline.stages.context import DataContext, RunContext, StageContext


def _png_bytes(width: int, height: int, pixels: list[list[int]]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        import struct

        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    import struct

    raw = bytearray()
    for row in pixels:
        raw.append(0)
        raw.extend(row)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )


def _write_box_png(path: Path, width: int, height: int, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    pixels = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(255 if x1 <= x < x2 and y1 <= y < y2 else 0)
        pixels.append(row)
    path.write_bytes(_png_bytes(width, height, pixels))


def _config(tmp_path: Path) -> PipelineConfig:
    task_dir = tmp_path / "tasks" / "mouse01"
    return PipelineConfig(
        task="mouse01",
        preset="annotation_dataset",
        input=InputConfig(
            rgbd_dir=str(task_dir),
            multi_views_dir=str(task_dir / "views"),
            first_frame=0,
        ),
        sam2=Sam2Config(container="sam2-test", points=[[10, 10]], labels=[1]),
        hunyuan=HunyuanConfig(views={"front": "front.png"}),
        real_size=RealSizeConfig(longest_edge=0.1),
        detection_dataset=DetectionDatasetConfig(class_name="mouse", class_id=0),
        output_dir=str(tmp_path / "output"),
        run_id="job1",
    )


def test_annotation_dataset_stages_export_yolo_from_masks(tmp_path):
    config = _config(tmp_path)
    task_dir = Path(config.input.rgbd_dir)
    rgb_dir = task_dir / "rgb"
    rgb_dir.mkdir(parents=True)
    for idx in range(2):
        (rgb_dir / f"{idx:05d}.png").write_bytes(_png_bytes(10, 10, [[idx] * 10 for _ in range(10)]))

    video_dir = tmp_path / "output" / "mouse01" / "runs" / "job1" / "stages" / "sam2_video_propagation"
    masks_dir = video_dir / "masks"
    masks_dir.mkdir(parents=True)
    _write_box_png(masks_dir / "00000.png", 10, 10, (2, 2, 6, 6))
    _write_box_png(masks_dir / "00001.png", 10, 10, (3, 2, 7, 6))

    qa_dir = tmp_path / "qa"
    qa_ctx = StageContext(
        run=RunContext(run_id="job1", task_name="mouse01"),
        data=DataContext(
            task_dir=task_dir,
            run_dir=tmp_path / "output" / "mouse01" / "runs" / "job1",
            output_dir=qa_dir,
            inputs={"sam2_video_propagation": video_dir},
        ),
        stage_name="mask_qa",
    )
    MaskQaStage().run(config, qa_dir, context=qa_ctx)

    report = json.loads((qa_dir / "qa_report.json").read_text())
    assert report["summary"]["accepted"] == 2
    assert report["frames"][0]["bbox_xyxy"] == [2, 2, 6, 6]
    assert (qa_dir / "review_status.json").exists()

    review_dir = tmp_path / "review"
    review_ctx = StageContext(
        run=RunContext(run_id="job1", task_name="mouse01"),
        data=DataContext(
            task_dir=task_dir,
            run_dir=tmp_path / "output" / "mouse01" / "runs" / "job1",
            output_dir=review_dir,
            inputs={"mask_qa": qa_dir},
        ),
        stage_name="review_pack",
    )
    ReviewPackStage().run(config, review_dir, context=review_ctx)
    assert (review_dir / "index.html").exists()
    assert (review_dir / "review_pack.json").exists()

    export_dir = tmp_path / "export"
    export_ctx = StageContext(
        run=RunContext(run_id="job1", task_name="mouse01"),
        data=DataContext(
            task_dir=task_dir,
            run_dir=tmp_path / "output" / "mouse01" / "runs" / "job1",
            output_dir=export_dir,
            inputs={"mask_qa": qa_dir, "review_pack": review_dir},
        ),
        stage_name="detection_dataset_export",
    )
    DetectionDatasetExportStage().run(config, export_dir, context=export_ctx)

    assert (export_dir / "images" / "00000.png").exists()
    assert (export_dir / "masks" / "00000.png").exists()
    assert (export_dir / "labels" / "00000.txt").read_text().startswith("0 ")
    annotations = json.loads((export_dir / "annotations.json").read_text())
    assert annotations["count"] == 2
    assert annotations["annotations"][0]["bbox_xyxy"] == [2, 2, 6, 6]
