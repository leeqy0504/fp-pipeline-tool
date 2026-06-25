"""Scale stage: resize OBJ model to real-world dimensions."""

from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.manifest import load_manifest_for_config
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError
from pipeline.stages.context import StageContext


def parse_obj_bounds(obj_text: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Parse vertex lines from OBJ text. Returns (min_corner, max_corner)."""
    vertices = []
    for line in obj_text.splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if not vertices:
        raise StageError("OBJ file has no vertices")

    xs, ys, zs = zip(*vertices)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def scale_obj(obj_text: str, real_longest_cm: float) -> str:
    """Apply uniform scale so the OBJ's longest edge matches real_longest_cm.

    OBJ units are assumed to be cm (same as config).
    """
    min_c, max_c = parse_obj_bounds(obj_text)
    extents = [max_c[i] - min_c[i] for i in range(3)]
    current_longest = max(extents)

    if current_longest <= 0:
        raise StageError("OBJ bounding box has zero extent")

    factor = real_longest_cm / current_longest

    lines = []
    for line in obj_text.splitlines():
        if line.startswith("v "):
            parts = line.split()
            x = float(parts[1]) * factor
            y = float(parts[2]) * factor
            z = float(parts[3]) * factor
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        else:
            lines.append(line)

    return "\n".join(lines)


@register_stage("scale")
class ScaleStage(BaseStage):
    name = "scale"

    _input_obj_path: str | None = None

    def run(self, config: PipelineConfig, output_dir: Path,
            context: StageContext | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._input_obj_path:
            obj_path = Path(self._input_obj_path)
        elif context and context.data and context.data.get_input("hunyuangen"):
            obj_path = context.input("hunyuangen") / "raw.obj"
        else:
            manifest = load_manifest_for_config(config)
            hunyuan_dir = manifest.get_output_dir("hunyuangen")
            if not hunyuan_dir:
                raise StageError("No hunyuangen output found in manifest")
            obj_path = Path(hunyuan_dir) / "raw.obj"

        self.check_input_path(str(obj_path), "Input OBJ")

        obj_text = obj_path.read_text()
        scaled = scale_obj(obj_text, config.real_size.longest_edge)

        out_path = output_dir / "scaled.obj"
        out_path.write_text(scaled)
        (output_dir / "obj.obj").write_text(scaled)

        return output_dir
