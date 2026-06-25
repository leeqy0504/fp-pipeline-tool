"""Annotation dataset pipeline placeholder stages."""

from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext


class _NotImplementedAnnotationStage(BaseStage):
    name = ""
    description = ""

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        raise StageError(f"{self.name} is not implemented yet: {self.description}")


@register_stage("mask_qa")
class MaskQaStage(_NotImplementedAnnotationStage):
    name = "mask_qa"
    description = "mask quality rules and frame review status"


@register_stage("review_pack")
class ReviewPackStage(_NotImplementedAnnotationStage):
    name = "review_pack"
    description = "contact sheet and review artifacts"


@register_stage("detection_dataset_export")
class DetectionDatasetExportStage(_NotImplementedAnnotationStage):
    name = "detection_dataset_export"
    description = "YOLO dataset export from reviewed masks"
