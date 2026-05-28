"""Configuration loading and validation."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

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
    points: list[list[int]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)


@dataclass
class HunyuanConfig:
    secret_id: str
    secret_key: str
    region: str = "ap-guangzhou"
    model: str = "3.1"
    face_count: int = 500000
    enable_pbr: bool = False
    views: dict[str, str] = field(default_factory=dict)


@dataclass
class RealSizeConfig:
    longest_edge: float  # meters (OBJ units from Hunyuan)


@dataclass
class PipelineConfig:
    task: str
    preset: str
    input: InputConfig
    sam2: Sam2Config
    hunyuan: HunyuanConfig
    real_size: RealSizeConfig
    output_dir: str = "output/"


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


_REQUIRED_TOP = ["task", "preset", "input", "sam2"]
_REQUIRED_INPUT = ["rgbd_dir", "multi_views_dir"]
_REQUIRED_SAM2 = ["container", "points", "labels"]
_REQUIRED_HUNYUAN = ["secret_id", "secret_key", "views"]
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


def load_config(config_path: str) -> PipelineConfig:
    """Load and validate a pipeline YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

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
            points=resolved["sam2"]["points"],
            labels=resolved["sam2"]["labels"],
        ),
        hunyuan=HunyuanConfig(
            secret_id=hunyuan_data.get("secret_id", ""),
            secret_key=hunyuan_data.get("secret_key", ""),
            region=hunyuan_data.get("region", "ap-guangzhou"),
            model=hunyuan_data.get("model", "3.1"),
            face_count=hunyuan_data.get("face_count", 500000),
            enable_pbr=hunyuan_data.get("enable_pbr", False),
            views=hunyuan_data.get("views", {}),
        ),
        real_size=RealSizeConfig(
            longest_edge=real_size_data.get("longest_edge", 1.0),
        ),
        output_dir=resolved.get("output_dir", "output/"),
    )
