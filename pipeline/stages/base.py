"""Base stage abstract class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.config import PipelineConfig

if TYPE_CHECKING:
    from pipeline.stages.context import StageContext


class StageError(Exception):
    """Stage execution error."""


class BaseStage(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, config: PipelineConfig, output_dir: Path,
            context: "StageContext | None" = None) -> Path:
        ...

    def check_input_path(self, path: str, description: str):
        p = Path(path)
        if not p.exists():
            raise StageError(f"{description} not found: {path}")
        return p
