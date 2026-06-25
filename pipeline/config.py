"""Configuration loading and validation."""

import os
import re
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Configuration validation error."""
    pass


@dataclass
class InputConfig:
    rgbd_dir: str
    multi_views_dir: str
    first_frame: int = 0


@dataclass
class Sam2Config:
    container: str
    checkpoint: str = "/opt/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    config_file: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    project_mount: str = "/home/try/code/fp-pipeline-tool"
    video_cli: str = "tools/sam2/sam2_video_cli.py"
    points: list[list[int]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)


@dataclass
class HunyuanConfig:
    secret_id: str = ""
    secret_key: str = ""
    region: str = "ap-guangzhou"
    model: str = "tencent/Hunyuan3D-2mv"
    subfolder: str = "hunyuan3d-dit-v2-mv"
    variant: str = "fp16"
    face_count: int = 500000
    enable_pbr: bool = False
    views: dict[str, str] = field(default_factory=dict)
    num_inference_steps: int = 50
    octree_resolution: int = 380
    num_chunks: int = 20000
    seed: int = 12345
    output_type: str = "trimesh"
    remove_background: bool = True
    conda_env: str = "hunyuan"
    project_dir: str = "/home/try/code/Hunyuan3D-2"
    python: str = ""


@dataclass
class RealSizeConfig:
    longest_edge: float  # meters (OBJ units from Hunyuan)


@dataclass
class FoundationPoseConfig:
    container: str = "foundationpose"
    workdir: str = "/home/try/code/FoundationPose"
    debug: int = 0


@dataclass
class DetectionDatasetConfig:
    class_name: str = "object"
    class_id: int = 0
    min_box_area: int = 16
    copy_images: bool = True
    preview: bool = False


@dataclass
class PipelineConfig:
    task: str
    preset: str
    input: InputConfig
    sam2: Sam2Config
    hunyuan: HunyuanConfig
    real_size: RealSizeConfig
    foundationpose: FoundationPoseConfig = field(default_factory=FoundationPoseConfig)
    detection_dataset: DetectionDatasetConfig = field(default_factory=DetectionDatasetConfig)
    output_dir: str = "output/"
    run_id: str | None = None
    pipeline_stages: list[str] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)
    registry_snapshot: dict[str, Any] = field(default_factory=dict)
    source_config_path: str | None = None


_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value):
    """Replace ${VAR} patterns with environment variable values.

    Missing env vars are left unresolved — stages that need them will
    fail at runtime with a clear error from the underlying SDK/API.
    """
    if isinstance(value, str):
        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        return _ENV_VAR_RE.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base and return a new dict."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _read_json_or_yaml(path: Path):
    import json

    if not path.exists():
        return None
    if path.suffix.lower() == ".json":
        with open(path) as f:
            return json.load(f)
    return _read_yaml(path)


def _find_project_root(config_path: Path, project_root: str | Path | None) -> Path:
    if project_root is not None:
        return Path(project_root)
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "configs").is_dir() or (candidate / "pipeline").is_dir():
            return candidate
    return Path.cwd()


def _load_named_yaml(base_dir: Path, name: str | None, required: bool = True) -> dict:
    if not name:
        return {}
    path = base_dir / f"{name}.yaml"
    if not path.exists():
        path = base_dir / f"{name}.yml"
    if not path.exists():
        if not required:
            return {}
        raise ConfigError(f"Referenced config not found: {base_dir / (name + '.yaml')}")
    return _read_yaml(path)


def _class_name_from_registry(project_root: Path, class_id: int | None) -> str | None:
    if class_id is None:
        return None
    classes = _read_json_or_yaml(project_root / "registry" / "classes.json")
    if not classes:
        return None
    if isinstance(classes, dict):
        items = classes.get("classes", [])
    else:
        items = classes
    for item in items:
        if int(item.get("class_id", -1)) == int(class_id):
            return item.get("name") or item.get("class_name")
    return None


def _load_layered_config(path: Path, project_root: Path, raw: dict) -> dict:
    """Build legacy PipelineConfig-shaped data from task.yaml style layers."""
    pipeline_name = raw.get("pipeline") or raw.get("preset")
    runtime_name = raw.get("runtime")

    pipeline_data = _load_named_yaml(project_root / "configs" / "pipelines", pipeline_name)
    runtime_data = _load_named_yaml(project_root / "configs" / "runtime", runtime_name)

    merged: dict = {}
    for algo_name in ("sam2", "hunyuan3d", "foundationpose", "yolo26"):
        merged = _deep_merge(
            merged,
            _load_named_yaml(project_root / "configs" / "algorithms", algo_name, required=False),
        )

    merged = _deep_merge(merged, runtime_data)
    merged = _deep_merge(merged, raw)

    task_id = raw.get("task_id") or raw.get("task") or path.parent.name
    merged["task"] = task_id
    merged["preset"] = pipeline_data.get("preset") or pipeline_name
    merged["pipeline_stages"] = pipeline_data.get("stages", [])
    merged["runtime"] = runtime_data.get("runtime", runtime_data)

    class_id = raw.get("class_id")
    if class_id is not None:
        det = merged.setdefault("detection_dataset", {})
        det.setdefault("class_id", class_id)
        class_name = _class_name_from_registry(project_root, int(class_id))
        if class_name:
            det.setdefault("class_name", class_name)
        merged["registry_snapshot"] = {
            "class_id": int(class_id),
            "class_name": det.get("class_name"),
            "classes_path": str(project_root / "registry" / "classes.json"),
        }

    return merged


_REQUIRED_TOP = ["task", "preset", "input", "sam2"]
_REQUIRED_INPUT = ["rgbd_dir", "multi_views_dir"]
_REQUIRED_SAM2 = ["container", "points", "labels"]
_REQUIRED_HUNYUAN = ["views"]
_REQUIRED_REAL_SIZE = ["longest_edge"]


def _validate_section(data, section_name, required_fields):
    if section_name not in data:
        raise ConfigError(
            f"Missing required field: '{section_name}'"
        )
    for field_name in required_fields:
        if field_name not in data[section_name]:
            raise ConfigError(
                f"Missing required field: '{section_name}.{field_name}'"
            )


def load_config(config_path: str, project_root: str | Path | None = None) -> PipelineConfig:
    """Load and validate a pipeline YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = _read_yaml(path)
    root = _find_project_root(path, project_root)
    if "task_id" in raw or "pipeline" in raw or path.name == "task.yaml":
        raw = _load_layered_config(path, root, raw)

    resolved = _resolve_env_vars(raw)

    for field_name in _REQUIRED_TOP:
        if field_name not in resolved:
            raise ConfigError(f"Missing required field: '{field_name}'")

    _validate_section(resolved, "input", _REQUIRED_INPUT)
    _validate_section(resolved, "sam2", _REQUIRED_SAM2)

    # Optional sections: only validate if present
    hunyuan_data = resolved.get("hunyuan", {})
    if hunyuan_data:
        _validate_section(resolved, "hunyuan", _REQUIRED_HUNYUAN)

    real_size_data = resolved.get("real_size", {})
    if real_size_data:
        _validate_section(resolved, "real_size", _REQUIRED_REAL_SIZE)

    fp_data = resolved.get("foundationpose", {})
    det_data = resolved.get("detection_dataset", {})

    return PipelineConfig(
        task=resolved["task"],
        preset=resolved["preset"],
        input=InputConfig(
            rgbd_dir=resolved["input"]["rgbd_dir"],
            multi_views_dir=resolved["input"]["multi_views_dir"],
            first_frame=resolved["input"].get("first_frame", 0),
        ),
        sam2=Sam2Config(
            container=resolved["sam2"]["container"],
            checkpoint=resolved["sam2"].get("checkpoint", "/opt/sam2/checkpoints/sam2.1_hiera_base_plus.pt"),
            config_file=resolved["sam2"].get("config_file", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
            project_mount=resolved["sam2"].get("project_mount", "/home/try/code/fp-pipeline-tool"),
            video_cli=resolved["sam2"].get("video_cli", "tools/sam2/sam2_video_cli.py"),
            points=resolved["sam2"]["points"],
            labels=resolved["sam2"]["labels"],
        ),
        hunyuan=HunyuanConfig(
            secret_id=hunyuan_data.get("secret_id", ""),
            secret_key=hunyuan_data.get("secret_key", ""),
            region=hunyuan_data.get("region", "ap-guangzhou"),
            model=hunyuan_data.get("model", "tencent/Hunyuan3D-2mv"),
            subfolder=hunyuan_data.get("subfolder", "hunyuan3d-dit-v2-mv"),
            variant=hunyuan_data.get("variant", "fp16"),
            face_count=hunyuan_data.get("face_count", 500000),
            enable_pbr=hunyuan_data.get("enable_pbr", False),
            views=hunyuan_data.get("views", {}),
            num_inference_steps=hunyuan_data.get("num_inference_steps", 50),
            octree_resolution=hunyuan_data.get("octree_resolution", 380),
            num_chunks=hunyuan_data.get("num_chunks", 20000),
            seed=hunyuan_data.get("seed", 12345),
            output_type=hunyuan_data.get("output_type", "trimesh"),
            remove_background=hunyuan_data.get("remove_background", True),
            conda_env=hunyuan_data.get("conda_env", "hunyuan"),
            project_dir=hunyuan_data.get("project_dir", "/home/try/code/Hunyuan3D-2"),
            python=hunyuan_data.get("python", ""),
        ),
        real_size=RealSizeConfig(
            longest_edge=real_size_data.get("longest_edge", 1.0),
        ),
        foundationpose=FoundationPoseConfig(
            container=fp_data.get("container", "foundationpose"),
            workdir=fp_data.get("workdir", "/home/try/code/FoundationPose"),
            debug=fp_data.get("debug", 0),
        ),
        detection_dataset=DetectionDatasetConfig(
            class_name=det_data.get("class_name", "object"),
            class_id=det_data.get("class_id", 0),
            min_box_area=det_data.get("min_box_area", 16),
            copy_images=det_data.get("copy_images", True),
            preview=det_data.get("preview", False),
        ),
        output_dir=resolved.get("output_dir", "output/"),
        run_id=resolved.get("run_id"),
        pipeline_stages=resolved.get("pipeline_stages", []),
        runtime=resolved.get("runtime", {}),
        registry_snapshot=resolved.get("registry_snapshot", {}),
        source_config_path=str(path),
    )
