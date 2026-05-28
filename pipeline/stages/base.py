"""Base stage abstract class."""

from abc import ABC, abstractmethod
from pathlib import Path

from pipeline.config import PipelineConfig


class StageError(Exception):
    """Stage execution error."""
    pass


class BaseStage(ABC):
    """Abstract base for all pipeline stages.

    Subclasses must define `name` and implement `run()`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique stage name (matches preset YAML key)."""
        ...

    @abstractmethod
    def run(self, config: PipelineConfig, output_dir: Path) -> Path:
        """Execute the stage. Return output directory path.

        Args:
            config: Full pipeline configuration.
            output_dir: Directory where this stage should write its output.

        Returns:
            Path to the stage's output directory.

        Raises:
            StageError: If stage execution fails.
        """
        ...

    def check_input_path(self, path: str, description: str):
        """Validate that an input path exists."""
        p = Path(path)
        if not p.exists():
            raise StageError(f"{description} not found: {path}")
        return p
