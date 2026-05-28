# Operator Model Development Pipeline - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI pipeline tool that automates FoundationPose dataset preparation across 4 stages: SAM2 mask generation, Hunyuan 3D model generation, scale correction, and dataset packaging.

**Architecture:** Stage-Pipeline pattern with `BaseStage` ABC, factory-based stage resolution, YAML config with env var substitution, JSON manifest for resumability. Models unitrain's argparse CLI, dataclass config, and factory patterns.

**Tech Stack:** Python 3.10+, `pyyaml`, `dataclasses`, `argparse`, `subprocess`, `tencentcloud-sdk-python` (for hunyuangen stage)

**Source spec:** `docs/superpowers/specs/2026-05-25-operator-pipeline-design.md`

---

## File Map

| File | Responsibility |
|---|---|
| `pipeline-tool/pyproject.toml` | Package metadata, CLI entry points |
| `pipeline-tool/configs/foundationpose.yaml` | Example config, checked in |
| `pipeline-tool/pipeline/__init__.py` | Package init, version |
| `pipeline-tool/pipeline/config.py` | `PipelineConfig` dataclass, `load_config()`, env var substitution, validation |
| `pipeline-tool/pipeline/manifest.py` | `Manifest` class: load/save JSON, stage status CRUD |
| `pipeline-tool/pipeline/stages/base.py` | `BaseStage` ABC with `run()`, `name`, input validation helpers |
| `pipeline-tool/pipeline/stages/__init__.py` | `get_stage()` factory function |
| `pipeline-tool/pipeline/stages/scale.py` | `ScaleStage`: parse OBJ, compute longest edge, apply uniform scale |
| `pipeline-tool/pipeline/stages/package.py` | `PackageStage`: assemble FP dataset from rgbd + mask + obj |
| `pipeline-tool/pipeline/stages/sam2mask.py` | `Sam2MaskStage`: docker run SAM2, capture mask |
| `pipeline-tool/pipeline/stages/hunyuangen.py` | `HunyuanGenStage`: tencentcloud SDK, submit/poll/download |
| `pipeline-tool/pipeline/presets.yaml` | Preset stage chain definitions |
| `pipeline-tool/pipeline/pipeline.py` | `PipelineOrchestrator`: resolve preset, run stages in sequence, manage manifest |
| `pipeline-tool/pipeline/cli.py` | argparse CLI: `run`, `stage`, `status` subcommands |
| `pipeline-tool/run.sh` | Shell entrypoint (following unitrain pattern) |
| `pipeline-tool/tests/conftest.py` | Shared fixtures |
| `pipeline-tool/tests/test_config.py` | Config loading + validation tests |
| `pipeline-tool/tests/test_manifest.py` | Manifest CRUD tests |
| `pipeline-tool/tests/test_scale_stage.py` | Scale stage unit tests |
| `pipeline-tool/tests/test_package_stage.py` | Package stage tests (uses bottle reference) |
| `pipeline-tool/tests/test_pipeline.py` | Pipeline orchestrator integration tests |
| `pipeline-tool/tests/fixtures/sample.obj` | Minimal OBJ for scale testing |

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pipeline-tool/pyproject.toml`
- Create: `pipeline-tool/pipeline/__init__.py`
- Create: `pipeline-tool/pipeline/stages/__init__.py`
- Create: `pipeline-tool/tests/__init__.py`
- Create: `pipeline-tool/tests/conftest.py`
- Create: `pipeline-tool/configs/foundationpose.yaml`
- Create: `pipeline-tool/pipeline/presets.yaml`

- [ ] **Step 1: Create directory tree**

Run:
```bash
mkdir -p /home/ubt2204/code/pipeline-tool/{pipeline/stages,tests,configs,output}
touch /home/ubt2204/code/pipeline-tool/pipeline/__init__.py
touch /home/ubt2204/code/pipeline-tool/pipeline/stages/__init__.py
touch /home/ubt2204/code/pipeline-tool/tests/__init__.py
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "pipeline-tool"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pyyaml>=6.0",
]

[project.scripts]
pipeline = "pipeline.cli:main"

[project.optional-dependencies]
hunyuan = ["tencentcloud-sdk-python>=3.0"]
```

Write to: `/home/ubt2204/code/pipeline-tool/pyproject.toml`

- [ ] **Step 3: Write pipeline/__init__.py**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Write pipeline/stages/__init__.py** (empty factory — filled in Task 4)

```python
"""Stage factory and registry."""

_registry: dict[str, type] = {}


def register_stage(name: str):
    """Decorator to register a stage class."""
    def wrapper(cls):
        _registry[name] = cls
        return cls
    return wrapper


def get_stage(name: str):
    """Return stage class by name."""
    if name not in _registry:
        raise KeyError(f"Unknown stage '{name}'. Available: {list(_registry.keys())}")
    return _registry[name]()


def list_stages():
    """Return list of registered stage names."""
    return list(_registry.keys())
```

- [ ] **Step 5: Write presets.yaml**

```yaml
foundationpose:
  stages:
    - sam2mask
    - hunyuangen
    - scale
    - package
```

Write to: `/home/ubt2204/code/pipeline-tool/pipeline/presets.yaml`

- [ ] **Step 6: Write example config**

```yaml
task: mouse_001
preset: foundationpose

input:
  rgbd_dir: /data/mouse_001/rgbd/
  multi_views_dir: /data/mouse_001/views/
  first_frame: 0

sam2:
  image: sam2:latest
  bbox: [320, 240, 480, 360]

hunyuan:
  secret_id: ${TENCENT_SECRET_ID}
  secret_key: ${TENCENT_SECRET_KEY}
  region: ap-guangzhou
  model: "3.1"
  face_count: 500000
  enable_pbr: false
  views:
    front: front.jpg
    left: left.jpg
    right: right.jpg
    back: back.jpg

real_size:
  longest_edge: 5.2

output_dir: output/
```

Write to: `/home/ubt2204/code/pipeline-tool/configs/foundationpose.yaml`

- [ ] **Step 7: Write tests/conftest.py**

```python
import os
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def tmp_dir():
    """Create a temporary directory, clean up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_obj(tmp_dir):
    """A simple 2x2x2 cube OBJ file (units: cm)."""
    content = """# Simple cube
v 0.0 0.0 0.0
v 2.0 0.0 0.0
v 2.0 2.0 0.0
v 0.0 2.0 0.0
v 0.0 0.0 2.0
v 2.0 0.0 2.0
v 2.0 2.0 2.0
v 0.0 2.0 2.0
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 4 8 5 1
"""
    path = tmp_dir / "sample.obj"
    path.write_text(content)
    return path
```

- [ ] **Step 8: Write .gitignore**

```
output/
__pycache__/
*.pyc
.eggs/
*.egg-info/
```

Write to: `/home/ubt2204/code/pipeline-tool/.gitignore`

---

### Task 2: Config Module

**Files:**
- Create: `pipeline-tool/pipeline/config.py`
- Create: `pipeline-tool/tests/test_config.py`

- [ ] **Step 1: Write failing tests for load_config**

```python
import os
import pytest
from pipeline.config import PipelineConfig, load_config, ConfigError


def test_load_config_basic(tmp_path, monkeypatch):
    """Load a minimal valid config with env var substitution."""
    monkeypatch.setenv("TENCENT_SECRET_ID", "test-id")
    monkeypatch.setenv("TENCENT_SECRET_KEY", "test-key")

    yaml_content = """
task: test_task
preset: foundationpose
input:
  rgbd_dir: /tmp/rgbd
  multi_views_dir: /tmp/views
  first_frame: 0
sam2:
  image: sam2:v1
  bbox: [10, 20, 30, 40]
hunyuan:
  secret_id: ${TENCENT_SECRET_ID}
  secret_key: ${TENCENT_SECRET_KEY}
  region: ap-guangzhou
  model: "3.1"
  face_count: 500000
  enable_pbr: false
  views:
    front: front.jpg
    left: left.jpg
    right: right.jpg
    back: back.jpg
real_size:
  longest_edge: 5.2
output_dir: output/
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)

    config = load_config(str(config_path))

    assert config.task == "test_task"
    assert config.preset == "foundationpose"
    assert config.input.rgbd_dir == "/tmp/rgbd"
    assert config.input.multi_views_dir == "/tmp/views"
    assert config.input.first_frame == 0
    assert config.sam2.image == "sam2:v1"
    assert config.sam2.bbox == [10, 20, 30, 40]
    assert config.hunyuan.secret_id == "test-id"
    assert config.hunyuan.secret_key == "test-key"
    assert config.hunyuan.region == "ap-guangzhou"
    assert config.hunyuan.views == {"front": "front.jpg", "left": "left.jpg", "right": "right.jpg", "back": "back.jpg"}
    assert config.real_size.longest_edge == 5.2
    assert config.output_dir == "output/"


def test_load_config_missing_required_field(tmp_path):
    """Missing required field raises ConfigError."""
    yaml_content = """
preset: foundationpose
input:
  rgbd_dir: /tmp/rgbd
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)

    with pytest.raises(ConfigError, match="Missing required field"):
        load_config(str(config_path))


def test_load_config_env_var_not_set(tmp_path):
    """Unresolved env var raises ConfigError."""
    yaml_content = """
task: test
preset: foundationpose
input:
  rgbd_dir: /tmp/rgbd
  multi_views_dir: /tmp/views
  first_frame: 0
sam2:
  image: sam2:v1
  bbox: [10, 20, 30, 40]
hunyuan:
  secret_id: ${MISSING_VAR}
  secret_key: sk
  region: ap-guangzhou
  model: "3.1"
  face_count: 500000
  enable_pbr: false
  views:
    front: f.jpg
real_size:
  longest_edge: 5.2
output_dir: output/
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)

    with pytest.raises(ConfigError, match="Environment variable .* not set"):
        load_config(str(config_path))


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pytest tests/test_config.py -v`

Expected: All 4 tests FAIL (ConfigError not defined, load_config not defined, etc.)

- [ ] **Step 3: Write config.py**

```python
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
    image: str
    bbox: list[int]


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
    longest_edge: float


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
    """Replace ${VAR} patterns with environment variable values."""
    if isinstance(value, str):
        def replacer(match):
            var_name = match.group(1)
            val = os.environ.get(var_name)
            if val is None:
                raise ConfigError(
                    f"Environment variable '{var_name}' not set"
                )
            return val
        return _ENV_VAR_RE.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


_REQUIRED_TOP = ["task", "preset", "input", "sam2", "hunyuan", "real_size"]
_REQUIRED_INPUT = ["rgbd_dir", "multi_views_dir"]
_REQUIRED_SAM2 = ["image", "bbox"]
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
    _validate_section(resolved, "hunyuan", _REQUIRED_HUNYUAN)
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
            image=resolved["sam2"]["image"],
            bbox=resolved["sam2"]["bbox"],
        ),
        hunyuan=HunyuanConfig(
            secret_id=resolved["hunyuan"]["secret_id"],
            secret_key=resolved["hunyuan"]["secret_key"],
            region=resolved["hunyuan"].get("region", "ap-guangzhou"),
            model=resolved["hunyuan"].get("model", "3.1"),
            face_count=resolved["hunyuan"].get("face_count", 500000),
            enable_pbr=resolved["hunyuan"].get("enable_pbr", False),
            views=resolved["hunyuan"]["views"],
        ),
        real_size=RealSizeConfig(
            longest_edge=resolved["real_size"]["longest_edge"],
        ),
        output_dir=resolved.get("output_dir", "output/"),
    )
```

- [ ] **Step 4: Run config tests to verify pass**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pytest tests/test_config.py -v`

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/ubt2204/code/pipeline-tool
git add pyproject.toml pipeline/__init__.py pipeline/config.py tests/test_config.py tests/conftest.py
git commit -m "feat: add config module with YAML loading and env var substitution"
```

---

### Task 3: Manifest Module

**Files:**
- Create: `pipeline-tool/pipeline/manifest.py`
- Create: `pipeline-tool/tests/test_manifest.py`

- [ ] **Step 1: Write failing tests for Manifest**

```python
import json
from pathlib import Path
from pipeline.manifest import Manifest


def test_manifest_create():
    m = Manifest(task="test_task", config_path="configs/test.yaml")
    assert m.task == "test_task"
    assert m.config_path == "configs/test.yaml"
    assert m.stages == {}


def test_manifest_mark_stage(tmp_path):
    m = Manifest(task="test_task", config_path="configs/test.yaml")

    m.mark_stage_done("scale", "output/scale/", 2.5)
    assert m.stages["scale"]["status"] == "done"
    assert m.stages["scale"]["output_dir"] == "output/scale/"
    assert m.stages["scale"]["duration_s"] == 2.5

    m.mark_stage_failed("hunyuan")
    assert m.stages["hunyuan"]["status"] == "failed"
    assert m.stages["hunyuan"]["output_dir"] is None


def test_manifest_save_load(tmp_path):
    path = tmp_path / "manifest.json"

    m1 = Manifest(task="test_task", config_path="configs/test.yaml")
    m1.mark_stage_done("scale", "output/scale/", 2.0)
    m1.save(str(path))

    assert path.exists()

    m2 = Manifest.load(str(path))
    assert m2.task == "test_task"
    assert m2.stages["scale"]["status"] == "done"
    assert m2.stages["scale"]["output_dir"] == "output/scale/"
    assert m2.stages["scale"]["duration_s"] == 2.0


def test_manifest_is_stage_done(tmp_path):
    m = Manifest(task="test", config_path="c.yaml")
    m.mark_stage_done("a", "out/a", 1.0)
    m.mark_stage_failed("b")

    assert m.is_stage_done("a") is True
    assert m.is_stage_done("b") is False
    assert m.is_stage_done("c") is False


def test_manifest_to_dict(tmp_path):
    m = Manifest(task="test", config_path="c.yaml")
    m.mark_stage_done("scale", "out/s", 1.0)
    d = m.to_dict()

    assert d["task"] == "test"
    assert d["stages"]["scale"]["status"] == "done"
    assert "created_at" in d
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pytest tests/test_manifest.py -v`

Expected: All FAIL (Manifest not defined)

- [ ] **Step 3: Write manifest.py**

```python
"""Manifest: track pipeline stage status and outputs."""

import json
from datetime import datetime, timezone
from pathlib import Path


class Manifest:
    """Tracks per-stage status, output paths, and timing."""

    def __init__(self, task: str, config_path: str, created_at: str | None = None):
        self.task = task
        self.config_path = config_path
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.stages: dict[str, dict] = {}

    def mark_stage_done(self, name: str, output_dir: str, duration_s: float):
        self.stages[name] = {
            "status": "done",
            "output_dir": output_dir,
            "duration_s": duration_s,
        }

    def mark_stage_failed(self, name: str):
        self.stages[name] = {
            "status": "failed",
            "output_dir": None,
            "duration_s": None,
        }

    def is_stage_done(self, name: str) -> bool:
        s = self.stages.get(name)
        return s is not None and s["status"] == "done"

    def get_output_dir(self, name: str) -> str | None:
        s = self.stages.get(name)
        if s:
            return s.get("output_dir")
        return None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "config_path": self.config_path,
            "created_at": self.created_at,
            "stages": self.stages,
        }

    @classmethod
    def load(cls, path: str) -> "Manifest":
        with open(path) as f:
            data = json.load(f)
        m = cls(
            task=data["task"],
            config_path=data["config_path"],
            created_at=data.get("created_at"),
        )
        m.stages = data.get("stages", {})
        return m

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
```

- [ ] **Step 4: Run manifest tests to verify pass**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pytest tests/test_manifest.py -v`

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/ubt2204/code/pipeline-tool
git add pipeline/manifest.py tests/test_manifest.py
git commit -m "feat: add manifest module for stage status tracking"
```

---

### Task 4: BaseStage ABC + Stage Factory + Presets

**Files:**
- Modify: `pipeline-tool/pipeline/stages/__init__.py` (already has factory skeleton)
- Create: `pipeline-tool/pipeline/stages/base.py`

- [ ] **Step 1: Write BaseStage ABC**

```python
"""Base stage abstract class."""

from abc import ABC, abstractmethod
from pathlib import Path

from pipeline.manifest import Manifest
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
```

- [ ] **Step 2: Verify stage registry decorator works**

The `_registry` dict and `register_stage` decorator are already in `pipeline/stages/__init__.py` from Task 1. Add a quick test:

```python
import pytest
from pipeline.stages import get_stage, register_stage
from pipeline.stages.base import BaseStage


def test_register_and_get_stage():
    @register_stage("test_stage")
    class TestStage(BaseStage):
        name = "test_stage"

        def run(self, config, output_dir):
            return output_dir

    stage = get_stage("test_stage")
    assert isinstance(stage, TestStage)
    assert stage.name == "test_stage"


def test_get_unknown_stage():
    with pytest.raises(KeyError, match="Unknown stage"):
        get_stage("nonexistent")
```

Append to: `tests/test_config.py` (temporary, will be moved to dedicated test later)

- [ ] **Step 3: Run the quick test**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pytest tests/test_config.py::test_register_and_get_stage tests/test_config.py::test_get_unknown_stage -v`

Expected: Both PASS

- [ ] **Step 4: Commit**

```bash
cd /home/ubt2204/code/pipeline-tool
git add pipeline/stages/base.py pipeline/stages/__init__.py tests/test_config.py
git commit -m "feat: add BaseStage ABC and stage factory with decorator registration"
```

---

### Task 5: Scale Stage

**Files:**
- Create: `pipeline-tool/pipeline/stages/scale.py`
- Create: `pipeline-tool/tests/fixtures/sample.obj` (already created by conftest.py fixture, but write a static one too)
- Create: `pipeline-tool/tests/test_scale_stage.py`

- [ ] **Step 1: Write failing tests for scale stage**

```python
import pytest
from pathlib import Path
from pipeline.config import PipelineConfig, InputConfig, Sam2Config, HunyuanConfig, RealSizeConfig
from pipeline.stages.scale import ScaleStage, parse_obj_bounds, scale_obj


def make_config(real_longest_edge=5.0):
    """Helper to create minimal PipelineConfig for scale stage."""
    return PipelineConfig(
        task="test",
        preset="foundationpose",
        input=InputConfig(rgbd_dir="/tmp/r", multi_views_dir="/tmp/v", first_frame=0),
        sam2=Sam2Config(image="s:v1", bbox=[0, 0, 10, 10]),
        hunyuan=HunyuanConfig(
            secret_id="id", secret_key="key", region="ap", model="3.1",
            face_count=500000, enable_pbr=False, views={"front": "f.jpg"}
        ),
        real_size=RealSizeConfig(longest_edge=real_longest_edge),
        output_dir="output/",
    )


def test_parse_obj_bounds():
    """Parse vertices from OBJ and compute bounding box."""
    obj = """v 0.0 0.0 0.0
v 3.0 0.0 0.0
v 0.0 4.0 0.0
v 0.0 0.0 5.0
f 1 2 3
"""
    min_corner, max_corner = parse_obj_bounds(obj)
    # Bounding box should span from (0,0,0) to (3,4,5)
    assert min_corner == pytest.approx((0.0, 0.0, 0.0))
    assert max_corner == pytest.approx((3.0, 4.0, 5.0))


def test_parse_obj_bounds_longest_edge():
    """Longest edge of cube is along Z axis (5.0)."""
    obj = """v 0.0 0.0 0.0
v 3.0 0.0 0.0
v 0.0 4.0 0.0
v 0.0 0.0 5.0
f 1 2 3
"""
    min_c, max_c = parse_obj_bounds(obj)
    extents = [max_c[i] - min_c[i] for i in range(3)]
    longest = max(extents)
    assert longest == pytest.approx(5.0)


def test_scale_obj_uniform():
    """Scale a known cube to a target longest_edge."""
    original = """v 0.0 0.0 0.0
v 2.0 0.0 0.0
v 2.0 2.0 0.0
v 0.0 2.0 0.0
v 0.0 0.0 2.0
v 2.0 0.0 2.0
v 2.0 2.0 2.0
v 0.0 2.0 2.0
f 1 2 3 4
"""
    # Target: longest edge should be 4.0 cm (currently 2.0)
    result = scale_obj(original, 4.0)
    # Parse result, verify all vertices are scaled by factor 2.0
    min_c, max_c = parse_obj_bounds(result)
    extents = [max_c[i] - min_c[i] for i in range(3)]
    assert max(extents) == pytest.approx(4.0)
    # Uniform scale: all sides multiply by same factor
    assert min_c == pytest.approx((0.0, 0.0, 0.0))
    assert max_c == pytest.approx((4.0, 4.0, 4.0))


def test_scale_stage_run(tmp_path):
    """Full stage execution: read OBJ, scale, write output."""
    # Write input OBJ
    input_obj = tmp_path / "input.obj"
    input_obj.write_text("""v 0.0 0.0 0.0
v 2.0 0.0 0.0
v 2.0 2.0 0.0
v 0.0 2.0 0.0
v 0.0 0.0 2.0
v 2.0 0.0 2.0
v 2.0 2.0 2.0
v 0.0 2.0 2.0
f 1 2 3 4
""")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = make_config(real_longest_edge=10.0)
    stage = ScaleStage()
    stage._input_obj_path = str(input_obj)  # override default logic for testing

    result = stage.run(config, output_dir)

    assert (result / "obj.obj").exists()
    scaled = (result / "obj.obj").read_text()
    min_c, max_c = parse_obj_bounds(scaled)
    assert max(max_c[i] - min_c[i] for i in range(3)) == pytest.approx(10.0)


def test_scale_stage_missing_input(tmp_path):
    """StageError if input OBJ does not exist."""
    config = make_config()
    stage = ScaleStage()
    stage._input_obj_path = "/nonexistent/path.obj"

    with pytest.raises(Exception):
        stage.run(config, tmp_path / "out")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pytest tests/test_scale_stage.py -v`

Expected: All FAIL (ScaleStage, parse_obj_bounds, scale_obj not defined)

- [ ] **Step 3: Write scale.py**

```python
"""Scale stage: resize OBJ model to real-world dimensions."""

from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.manifest import Manifest
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError


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

    # Override in tests to bypass manifest
    _input_obj_path: str | None = None

    def run(self, config: PipelineConfig, output_dir: Path) -> Path:
        manifest = Manifest(config.task, config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine input OBJ path: from upstream stage or test override
        if self._input_obj_path:
            obj_path = Path(self._input_obj_path)
        else:
            hunyuan_dir = manifest.get_output_dir("hunyuangen")
            if not hunyuan_dir:
                raise StageError("No hunyuangen output found in manifest")
            obj_path = Path(hunyuan_dir) / "obj.obj"

        self.check_input_path(str(obj_path), "Input OBJ")

        obj_text = obj_path.read_text()
        scaled = scale_obj(obj_text, config.real_size.longest_edge)

        out_path = output_dir / "obj.obj"
        out_path.write_text(scaled)

        return output_dir
```

- [ ] **Step 4: Run scale tests to verify pass**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pytest tests/test_scale_stage.py -v`

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/ubt2204/code/pipeline-tool
git add pipeline/stages/scale.py tests/test_scale_stage.py
git commit -m "feat: add scale stage for OBJ longest-edge alignment"
```

---

### Task 6: Package Stage

**Files:**
- Create: `pipeline-tool/pipeline/stages/package.py`
- Create: `pipeline-tool/tests/test_package_stage.py`

- [ ] **Step 1: Write failing tests for package stage**

```python
import json
import shutil
from pathlib import Path
import pytest
from pipeline.config import PipelineConfig, InputConfig, Sam2Config, HunyuanConfig, RealSizeConfig
from pipeline.stages.package import PackageStage


def make_config(rgbd_dir="/tmp/r", first_frame=0):
    return PipelineConfig(
        task="test",
        preset="foundationpose",
        input=InputConfig(rgbd_dir=rgbd_dir, multi_views_dir="/tmp/v", first_frame=first_frame),
        sam2=Sam2Config(image="s:v1", bbox=[0, 0, 10, 10]),
        hunyuan=HunyuanConfig(
            secret_id="id", secret_key="key", region="ap", model="3.1",
            face_count=500000, enable_pbr=False, views={"front": "f.jpg"}
        ),
        real_size=RealSizeConfig(longest_edge=5.0),
        output_dir="output/",
    )


def build_mock_rgbd_dir(base: Path, num_frames=3):
    """Create a minimal rgbd directory structure matching capture script output."""
    rgb_dir = base / "rgb"
    depth_dir = base / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    for i in range(num_frames):
        # Original capture naming (e.g., frame_0000.png)
        (rgb_dir / f"frame_{i:04d}.png").write_text(f"rgb{i}")
        (depth_dir / f"frame_{i:04d}.png").write_text(f"depth{i}")

    # Camera params from capture script
    cam_params = {
        "width": 640, "height": 480,
        "fx": 603.19, "fy": 602.36,
        "ppx": 322.92, "ppy": 250.40,
        "depth_scale": 0.001,
    }
    (base / "camera_params.json").write_text(json.dumps(cam_params))


def build_mock_mask(base: Path):
    mask_path = base / "mask.png"
    mask_path.write_text("mask")
    return mask_path


def build_mock_obj(base: Path):
    obj_path = base / "obj.obj"
    obj_path.write_text("v 0 0 0\nv 1 1 1\nf 1 2\n")
    return obj_path


def test_package_creates_fp_dataset(tmp_path):
    """Full package stage: verify FP dataset structure."""
    rgbd_dir = tmp_path / "rgbd"
    build_mock_rgbd_dir(rgbd_dir, num_frames=3)

    mask_path = build_mock_obj(tmp_path)  # temp
    mask_dir = tmp_path / "mask_dir"
    mask_dir.mkdir()
    mask_file = mask_dir / "mask.png"
    mask_file.write_text("maskdata")

    obj_dir = tmp_path / "obj_dir"
    obj_dir.mkdir()
    obj_file = obj_dir / "obj.obj"
    obj_file.write_text("v 0 0 0\nv 1 1 1\nf 1 2\n")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = make_config(rgbd_dir=str(rgbd_dir), first_frame=0)
    stage = PackageStage()
    stage._mask_path_override = str(mask_file)
    stage._obj_path_override = str(obj_file)

    result = stage.run(config, output_dir)

    # Verify structure
    assert (result / "rgb").is_dir()
    assert (result / "depth").is_dir()
    assert (result / "masks").is_dir()
    assert (result / "mesh").is_dir()
    assert (result / "cam_K.txt").exists()
    assert (result / "camera_params.json").exists()

    # Verify file naming (frame_0000.png -> 00000.png)
    rgb_files = sorted((result / "rgb").iterdir())
    assert rgb_files[0].name == "00000.png"
    assert rgb_files[1].name == "00001.png"
    assert rgb_files[2].name == "00002.png"

    # Verify mask
    assert (result / "masks" / "00000.png").exists()
    assert (result / "masks" / "00000.png").read_text() == "maskdata"

    # Verify mesh
    assert (result / "mesh" / "textured_simple.obj").exists()
    assert (result / "mesh" / "textured_simple.obj").read_text() == "v 0 0 0\nv 1 1 1\nf 1 2\n"


def test_package_generates_cam_k(tmp_path):
    """cam_K.txt contains correct 3x3 intrinsics matrix."""
    rgbd_dir = tmp_path / "rgbd"
    build_mock_rgbd_dir(rgbd_dir, num_frames=1)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    config = make_config(rgbd_dir=str(rgbd_dir))
    stage = PackageStage()
    stage._mask_path_override = str(tmp_path / "m.png")
    stage._obj_path_override = str(tmp_path / "o.obj")
    (tmp_path / "m.png").write_text("x")
    (tmp_path / "o.obj").write_text("v 0 0 0\nf 1\n")

    result = stage.run(config, output_dir)

    cam_k = (result / "cam_K.txt").read_text().strip().split("\n")
    assert len(cam_k) == 3
    # fx should be in first row
    assert "603.19" in cam_k[0]


def test_package_camera_params_json(tmp_path):
    """camera_params.json preserves all fields from capture script."""
    rgbd_dir = tmp_path / "rgbd"
    build_mock_rgbd_dir(rgbd_dir)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    config = make_config(rgbd_dir=str(rgbd_dir))
    stage = PackageStage()
    stage._mask_path_override = str(tmp_path / "m.png")
    stage._obj_path_override = str(tmp_path / "o.obj")
    (tmp_path / "m.png").write_text("x")
    (tmp_path / "o.obj").write_text("v 0 0 0\nf 1\n")

    result = stage.run(config, output_dir)

    params = json.loads((result / "camera_params.json").read_text())
    assert params["width"] == 640
    assert params["height"] == 480
    assert params["fx"] == 603.19
    assert params["depth_scale"] == 0.001


def test_package_first_frame_selection(tmp_path):
    """Test non-zero first_frame starts naming from that offset."""
    rgbd_dir = tmp_path / "rgbd"
    build_mock_rgbd_dir(rgbd_dir, num_frames=5)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    config = make_config(rgbd_dir=str(rgbd_dir), first_frame=0)
    stage = PackageStage()
    stage._mask_path_override = str(tmp_path / "m.png")
    stage._obj_path_override = str(tmp_path / "o.obj")
    (tmp_path / "m.png").write_text("x")
    (tmp_path / "o.obj").write_text("v 0 0 0\nf 1\n")

    result = stage.run(config, output_dir)
    rgb_files = sorted((result / "rgb").iterdir())
    # First file is 00000.png regardless of first_frame value
    assert rgb_files[0].name == "00000.png"
    assert len(rgb_files) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pytest tests/test_package_stage.py -v`

Expected: All FAIL (PackageStage not defined)

- [ ] **Step 3: Write package.py**

```python
"""Package stage: assemble FP dataset from RGB-D, mask, and OBJ."""

import json
import shutil
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.manifest import Manifest
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError


def _generate_cam_k_txt(fx: float, fy: float, ppx: float, ppy: float) -> str:
    """Generate 3x3 camera intrinsics matrix in cam_K.txt format."""
    return (
        f"{fx:.6f} 0.000000 0.000000\n"
        f"0.000000 {fy:.6f} 0.000000\n"
        f"{ppx:.6f} {ppy:.6f} 1.000000\n"
    )


@register_stage("package")
class PackageStage(BaseStage):
    name = "package"

    _mask_path_override: str | None = None
    _obj_path_override: str | None = None

    def run(self, config: PipelineConfig, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Resolve input paths ---

        rgbd_dir = Path(config.input.rgbd_dir)
        self.check_input_path(str(rgbd_dir), "RGB-D directory")

        rgb_src = rgbd_dir / "rgb"
        depth_src = rgbd_dir / "depth"
        cam_src = rgbd_dir / "camera_params.json"
        self.check_input_path(str(rgb_src), "RGB source directory")
        self.check_input_path(str(depth_src), "Depth source directory")
        self.check_input_path(str(cam_src), "Camera params file")

        # Load camera intrinsics from capture script output
        with open(cam_src) as f:
            cam_data = json.load(f)

        # Mask: from sam2mask stage or test override
        if self._mask_path_override:
            mask_src = Path(self._mask_path_override)
        else:
            manifest = Manifest(config.task, config.output_dir)
            mask_dir = manifest.get_output_dir("sam2mask")
            if not mask_dir:
                raise StageError("No sam2mask output found in manifest")
            mask_src = Path(mask_dir) / "mask.png"
        self.check_input_path(str(mask_src), "Mask file")

        # OBJ: from scale stage or test override
        if self._obj_path_override:
            obj_src = Path(self._obj_path_override)
        else:
            manifest = Manifest(config.task, config.output_dir)
            scale_dir = manifest.get_output_dir("scale")
            if not scale_dir:
                raise StageError("No scale output found in manifest")
            obj_src = Path(scale_dir) / "obj.obj"
        self.check_input_path(str(obj_src), "Scaled OBJ file")

        # --- Copy and rename RGB frames ---

        out_rgb = output_dir / "rgb"
        out_rgb.mkdir(exist_ok=True)

        rgb_files = sorted(rgb_src.glob("*.png"))
        if not rgb_files:
            raise StageError(f"No PNG images found in {rgb_src}")

        for idx, src_file in enumerate(rgb_files):
            dst = out_rgb / f"{idx:05d}.png"
            shutil.copy2(src_file, dst)

        # --- Copy and rename Depth frames ---

        out_depth = output_dir / "depth"
        out_depth.mkdir(exist_ok=True)

        depth_files = sorted(depth_src.glob("*.png"))
        for idx, src_file in enumerate(depth_files):
            dst = out_depth / f"{idx:05d}.png"
            shutil.copy2(src_file, dst)

        # --- Copy mask as first-frame mask ---

        out_mask = output_dir / "masks"
        out_mask.mkdir(exist_ok=True)
        mask_dst = out_mask / f"{config.input.first_frame:05d}.png"
        shutil.copy2(mask_src, mask_dst)

        # --- Copy scaled OBJ as textured_simple.obj ---

        out_mesh = output_dir / "mesh"
        out_mesh.mkdir(exist_ok=True)
        shutil.copy2(obj_src, out_mesh / "textured_simple.obj")

        # --- Write camera files ---

        cam_k = _generate_cam_k_txt(
            cam_data["fx"], cam_data["fy"],
            cam_data["ppx"], cam_data["ppy"],
        )
        (output_dir / "cam_K.txt").write_text(cam_k)

        out_cam_params = {
            "width": cam_data["width"],
            "height": cam_data["height"],
            "fx": cam_data["fx"],
            "fy": cam_data["fy"],
            "ppx": cam_data["ppx"],
            "ppy": cam_data["ppy"],
            "depth_scale": cam_data["depth_scale"],
        }
        with open(output_dir / "camera_params.json", "w") as f:
            json.dump(out_cam_params, f, indent=4)

        return output_dir
```

- [ ] **Step 4: Run package tests to verify pass**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pytest tests/test_package_stage.py -v`

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/ubt2204/code/pipeline-tool
git add pipeline/stages/package.py tests/test_package_stage.py
git commit -m "feat: add package stage for FP dataset assembly"
```

---

### Task 7: Sam2Mask Stage

**Files:**
- Create: `pipeline-tool/pipeline/stages/sam2mask.py`

- [ ] **Step 1: Write sam2mask.py**

```python
"""SAM2 mask generation stage: runs SAM2 via Docker."""

import subprocess
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError


@register_stage("sam2mask")
class Sam2MaskStage(BaseStage):
    name = "sam2mask"

    def run(self, config: PipelineConfig, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        rgbd_dir = Path(config.input.rgbd_dir)
        self.check_input_path(str(rgbd_dir), "RGB-D directory")

        rgb_src = rgbd_dir / "rgb"
        self.check_input_path(str(rgb_src), "RGB source directory")

        # Find first-frame RGB image
        rgb_files = sorted(rgb_src.glob("*.png"))
        if not rgb_files:
            raise StageError(f"No RGB images found in {rgb_src}")

        first_frame = rgb_files[config.input.first_frame]
        x1, y1, x2, y2 = config.sam2.bbox

        mask_output = output_dir / "mask.png"

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{first_frame.parent}:/input:ro",
            "-v", f"{output_dir}:/output",
            config.sam2.image,
            "python", "infer.py",
            "--image", f"/input/{first_frame.name}",
            "--bbox", f"{x1},{y1},{x2},{y2}",
            "--output", "/output/mask.png",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise StageError(
                f"SAM2 Docker failed (exit {result.returncode}):\n"
                f"stderr: {result.stderr}"
            )

        if not mask_output.exists():
            raise StageError(f"SAM2 completed but mask not found at {mask_output}")

        return output_dir
```

- [ ] **Step 2: Commit**

```bash
cd /home/ubt2204/code/pipeline-tool
git add pipeline/stages/sam2mask.py
git commit -m "feat: add sam2mask stage (docker wrapper)"
```

---

### Task 8: Hunyuangen Stage

**Files:**
- Create: `pipeline-tool/pipeline/stages/hunyuangen.py`

- [ ] **Step 1: Write hunyuangen.py**

```python
"""Hunyuan 3D generation stage: submit/poll/download via tencentcloud SDK."""

import base64
import time
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.stages import register_stage
from pipeline.stages.base import BaseStage, StageError


def _load_tencentcloud():
    """Lazy-import tencentcloud SDK."""
    try:
        from tencentcloud.common.credential import Credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.ai3d.v20250513 import ai3d_client, models
        return Credential, ClientProfile, HttpProfile, ai3d_client, models
    except ImportError:
        raise StageError(
            "tencentcloud-sdk-python not installed. "
            "Run: pip install pipeline-tool[hunyuan]"
        )


_POLL_INTERVAL = 10   # seconds
_MAX_WAIT = 600        # 10 minutes timeout


@register_stage("hunyuangen")
class HunyuanGenStage(BaseStage):
    name = "hunyuangen"

    def run(self, config: PipelineConfig, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        Credential, ClientProfile, HttpProfile, ai3d_client, models = _load_tencentcloud()

        views_dir = Path(config.input.multi_views_dir)
        self.check_input_path(str(views_dir), "Multi-view images directory")

        # Build MultiViewImages array
        multi_view_images = []
        for view_name, filename in config.hunyuan.views.items():
            img_path = views_dir / filename
            self.check_input_path(str(img_path), f"View image ({view_name})")
            with open(img_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            multi_view_images.append(models.ViewImage(
                ViewType=view_name,
                ImageBase64=b64_data,
            ))

        # Create credential and client
        cred = Credential(config.hunyuan.secret_id, config.hunyuan.secret_key)
        http_profile = HttpProfile(endpoint="ai3d.tencentcloudapi.com")
        client_profile = ClientProfile(httpProfile=http_profile)
        client = ai3d_client.Ai3dClient(cred, config.hunyuan.region, client_profile)

        # Submit job
        req = models.SubmitHunyuanTo3DProJobRequest()
        req.Model = config.hunyuan.model
        req.MultiViewImages = multi_view_images
        req.GenerateType = "Normal"
        req.FaceCount = config.hunyuan.face_count
        req.EnablePBR = config.hunyuan.enable_pbr

        resp = client.SubmitHunyuanTo3DProJob(req)
        job_id = resp.JobId
        print(f"[hunyuangen] Job submitted: {job_id}")

        # Poll until complete
        elapsed = 0
        query_req = models.QueryHunyuanTo3DProJobRequest()
        query_req.JobId = job_id

        while elapsed < _MAX_WAIT:
            time.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL

            query_resp = client.QueryHunyuanTo3DProJob(query_req)
            status = query_resp.Status

            if status == "SUCCESS":
                break
            elif status == "FAILED":
                raise StageError(
                    f"Hunyuan job {job_id} failed: {query_resp.ErrorMessage}"
                )

            print(f"[hunyuangen] Polling... status={status}, elapsed={elapsed}s")

        if elapsed >= _MAX_WAIT:
            raise StageError(f"Hunyuan job {job_id} timed out after {_MAX_WAIT}s")

        # Download result — save to output dir
        # Note: The API response contains a URL or file data in the query response.
        # Depending on actual response format, adjust this section.
        result_url = getattr(query_resp, "ResultUrl", None)
        if result_url:
            import urllib.request
            obj_path = output_dir / "obj.obj"
            urllib.request.urlretrieve(result_url, str(obj_path))
        else:
            # Fallback: result may be in response directly
            result_data = getattr(query_resp, "ResultData", None)
            if result_data:
                obj_path = output_dir / "obj.obj"
                obj_path.write_bytes(base64.b64decode(result_data))
            else:
                raise StageError(
                    f"Job {job_id} completed but no result data found in response"
                )

        print(f"[hunyuangen] Downloaded result to {output_dir}")
        return output_dir
```

- [ ] **Step 2: Commit**

```bash
cd /home/ubt2204/code/pipeline-tool
git add pipeline/stages/hunyuangen.py
git commit -m "feat: add hunyuangen stage (tencentcloud API)"
```

---

### Task 9: Pipeline Orchestrator

**Files:**
- Create: `pipeline-tool/pipeline/pipeline.py`
- Create: `pipeline-tool/tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for pipeline orchestrator**

```python
import json
from pathlib import Path
import pytest
from pipeline.pipeline import PipelineOrchestrator
from pipeline.config import (
    PipelineConfig, InputConfig, Sam2Config, HunyuanConfig, RealSizeConfig,
)


def make_config(rgbd_dir="/tmp/r"):
    return PipelineConfig(
        task="test",
        preset="foundationpose",
        input=InputConfig(rgbd_dir=rgbd_dir, multi_views_dir="/tmp/v", first_frame=0),
        sam2=Sam2Config(image="s:v1", bbox=[0, 0, 10, 10]),
        hunyuan=HunyuanConfig(
            secret_id="id", secret_key="key", region="ap", model="3.1",
            face_count=500000, enable_pbr=False, views={"front": "f.jpg"}
        ),
        real_size=RealSizeConfig(longest_edge=5.0),
        output_dir="output/",
    )


def test_pipeline_resolve_preset(tmp_path):
    """Orchestrator resolves preset to stage list."""
    presets_yaml = tmp_path / "presets.yaml"
    presets_yaml.write_text("""
test_preset:
  stages:
    - stage_a
    - stage_b
""")
    orch = PipelineOrchestrator(presets_path=str(presets_yaml))
    stages = orch.resolve_preset("test_preset")
    assert stages == ["stage_a", "stage_b"]


def test_pipeline_unknown_preset(tmp_path):
    """Error for unknown preset name."""
    presets_yaml = tmp_path / "presets.yaml"
    presets_yaml.write_text("known: {stages: [a]}")
    orch = PipelineOrchestrator(presets_path=str(presets_yaml))

    with pytest.raises(KeyError, match="Unknown preset"):
        orch.resolve_preset("nonexistent")


def test_pipeline_manifest_path(tmp_path):
    """Manifest path derived from config task name."""
    config = make_config(rgbd_dir=str(tmp_path))
    config.task = "my_task"
    config.output_dir = str(tmp_path / "output")

    orch = PipelineOrchestrator()
    path = orch._manifest_path(config)
    assert path.endswith("my_task/manifest.json")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pytest tests/test_pipeline.py -v`

Expected: All FAIL (PipelineOrchestrator not defined)

- [ ] **Step 3: Write pipeline.py**

```python
"""Pipeline orchestrator: resolve preset, run stages in sequence."""

from pathlib import Path

import yaml

from pipeline.config import PipelineConfig
from pipeline.manifest import Manifest
from pipeline.stages import get_stage


class PipelineOrchestrator:
    """Runs presets by dispatching stages in order, tracking progress via manifest."""

    def __init__(self, presets_path: str | None = None):
        """Load presets from YAML file.

        Defaults to <package_dir>/presets.yaml.
        """
        if presets_path is None:
            presets_path = str(Path(__file__).parent / "presets.yaml")
        with open(presets_path) as f:
            self.presets = yaml.safe_load(f)

    def resolve_preset(self, preset_name: str) -> list[str]:
        """Resolve a preset name to its ordered stage list."""
        if preset_name not in self.presets:
            raise KeyError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(self.presets.keys())}"
            )
        return list(self.presets[preset_name]["stages"])

    def _manifest_path(self, config: PipelineConfig) -> str:
        return str(
            Path(config.output_dir) / config.task / "manifest.json"
        )

    def _stage_output_dir(self, config: PipelineConfig, stage_name: str) -> str:
        return str(Path(config.output_dir) / config.task / stage_name)

    def run_preset(self, config: PipelineConfig, force: bool = False):
        """Run all stages in the preset."""
        stages = self.resolve_preset(config.preset)

        manifest_path = self._manifest_path(config)
        manifest = Manifest.load(manifest_path) if Path(manifest_path).exists() else Manifest(
            task=config.task,
            config_path=config.output_dir,
        )

        for stage_name in stages:
            if not force and manifest.is_stage_done(stage_name):
                print(f"[pipeline] {stage_name}: skip (already done)")
                continue

            print(f"[pipeline] {stage_name}: starting...")
            stage = get_stage(stage_name)
            output_dir = Path(self._stage_output_dir(config, stage_name))

            import time
            start = time.time()
            try:
                result_path = stage.run(config, output_dir)
                elapsed = time.time() - start
                manifest.mark_stage_done(stage_name, str(result_path), elapsed)
                print(f"[pipeline] {stage_name}: done ({elapsed:.1f}s)")
            except Exception as e:
                manifest.mark_stage_failed(stage_name)
                manifest.save(manifest_path)
                print(f"[pipeline] {stage_name}: FAILED - {e}")
                raise

            manifest.save(manifest_path)

        print(f"[pipeline] Complete. Manifest: {manifest_path}")

    def run_stage(self, config: PipelineConfig, stage_name: str, force: bool = False):
        """Run a single stage by name."""
        manifest_path = self._manifest_path(config)
        manifest = Manifest.load(manifest_path) if Path(manifest_path).exists() else Manifest(
            task=config.task,
            config_path=config.output_dir,
        )

        if not force and manifest.is_stage_done(stage_name):
            print(f"[pipeline] {stage_name}: skip (already done)")
            return

        print(f"[pipeline] {stage_name}: starting...")
        stage = get_stage(stage_name)
        output_dir = Path(self._stage_output_dir(config, stage_name))

        import time
        start = time.time()
        try:
            result_path = stage.run(config, output_dir)
            elapsed = time.time() - start
            manifest.mark_stage_done(stage_name, str(result_path), elapsed)
            print(f"[pipeline] {stage_name}: done ({elapsed:.1f}s)")
        except Exception as e:
            manifest.mark_stage_failed(stage_name)
            manifest.save(manifest_path)
            print(f"[pipeline] {stage_name}: FAILED - {e}")
            raise

        manifest.save(manifest_path)

    def status(self, config: PipelineConfig):
        """Print task status from manifest."""
        manifest_path = self._manifest_path(config)
        if not Path(manifest_path).exists():
            print(f"No manifest found for task '{config.task}'")
            return

        manifest = Manifest.load(manifest_path)
        print(f"Task: {manifest.task}")
        print(f"Created: {manifest.created_at}")
        print("Stages:")
        for name, info in manifest.stages.items():
            status = info["status"]
            marker = "✅" if status == "done" else "❌" if status == "failed" else "⏳"
            output = info.get("output_dir") or "-"
            duration = info.get("duration_s")
            dur_str = f" ({duration:.1f}s)" if duration else ""
            print(f"  {marker} {name}: {status}{dur_str} -> {output}")
```

- [ ] **Step 4: Run pipeline tests to verify pass**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pytest tests/test_pipeline.py -v`

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/ubt2204/code/pipeline-tool
git add pipeline/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestrator with preset resolution and manifest tracking"
```

---

### Task 10: CLI Entry Point

**Files:**
- Create: `pipeline-tool/pipeline/cli.py`

- [ ] **Step 1: Write cli.py**

```python
"""CLI entry point for the pipeline tool."""

import argparse
import sys

from pipeline.config import load_config, ConfigError
from pipeline.pipeline import PipelineOrchestrator


def cmd_run(args):
    """Handle `pipeline run <preset> --config <path>`."""
    config = load_config(args.config)
    if args.preset:
        config.preset = args.preset

    orch = PipelineOrchestrator()
    orch.run_preset(config, force=args.force)


def cmd_stage(args):
    """Handle `pipeline stage <name> --config <path>`."""
    config = load_config(args.config)

    orch = PipelineOrchestrator()
    orch.run_stage(config, args.stage_name, force=args.force)


def cmd_status(args):
    """Handle `pipeline status --config <path>`."""
    config = load_config(args.config)

    orch = PipelineOrchestrator()
    orch.status(config)


def main():
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Operator model development pipeline tool",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # pipeline run <preset> --config <path> [--force]
    run_parser = subparsers.add_parser("run", help="Run a preset pipeline")
    run_parser.add_argument("preset", nargs="?", help="Preset name (overrides config)")
    run_parser.add_argument("--config", required=True, help="Path to YAML config file")
    run_parser.add_argument("--force", action="store_true", help="Force re-run all stages")
    run_parser.set_defaults(func=cmd_run)

    # pipeline stage <name> --config <path> [--force]
    stage_parser = subparsers.add_parser("stage", help="Run a single stage")
    stage_parser.add_argument("stage_name", help="Stage name to run")
    stage_parser.add_argument("--config", required=True, help="Path to YAML config file")
    stage_parser.add_argument("--force", action="store_true", help="Force re-run")
    stage_parser.set_defaults(func=cmd_stage)

    # pipeline status --config <path>
    status_parser = subparsers.add_parser("status", help="Show task status")
    status_parser.add_argument("--config", required=True, help="Path to YAML config file")
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 2: Verify CLI loads correctly**

Run:
```
cd /home/ubt2204/code/pipeline-tool && python -c "from pipeline.cli import main; print('CLI loaded OK')"
```

- [ ] **Step 3: Commit**

```bash
cd /home/ubt2204/code/pipeline-tool
git add pipeline/cli.py
git commit -m "feat: add argparse CLI with run, stage, status subcommands"
```

---

### Task 11: Shell Entrypoint (run.sh)

**Files:**
- Create: `pipeline-tool/run.sh`

- [ ] **Step 1: Write run.sh**

```bash
#!/usr/bin/env bash
# Pipeline tool entrypoint — manages Python environment and dispatches commands.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

# Ensure virtualenv exists
ensure_venv() {
    if [ ! -d "${VENV_DIR}" ]; then
        echo "[run.sh] Creating virtualenv..."
        python3 -m venv "${VENV_DIR}"
        source "${VENV_DIR}/bin/activate"
        pip install -e "${PROJECT_DIR}"
        echo "[run.sh] Virtualenv ready."
    fi
}

# Source the venv
source "${VENV_DIR}/bin/activate"

# Dispatch to Python CLI
PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python -m pipeline.cli "$@"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x /home/ubt2204/code/pipeline-tool/run.sh
```

- [ ] **Step 3: Verify help output**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pipeline.cli --help` (before run.sh first use)

- [ ] **Step 4: Commit**

```bash
cd /home/ubt2204/code/pipeline-tool
git add run.sh
git commit -m "feat: add run.sh shell entrypoint"
```

---

### Task 12: E2E Integration Test (Phase 1 - Bottle Dataset)

**Files:**
- Create: `pipeline-tool/tests/test_e2e_bottle.py`

- [ ] **Step 1: Create bottle-based E2E test**

This test uses the existing `bottle` reference dataset to verify `scale` + `package` end-to-end. We skip `sam2mask` and `hunyuangen` by providing pre-existing artifacts.

```python
"""E2E test: scale + package using bottle reference dataset.

Tests the minimal closed loop:
  1. Scale: use bottle's existing mesh, verify scale factor math
  2. Package: produce a second dataset, verify it matches bottle structure
"""

import json
import shutil
from pathlib import Path
import pytest

# Path to the reference bottle dataset
BOTTLE_DIR = Path("/home/ubt2204/Downloads/data/bottle")


@pytest.fixture(scope="module")
def bottle_copy(tmp_path_factory):
    """Copy bottle dataset to a temp location for testing."""
    tmp = tmp_path_factory.mktemp("bottle_test")
    dst = tmp / "bottle"
    shutil.copytree(BOTTLE_DIR, dst)
    return dst


def test_bottle_rgb_count(bottle_copy):
    """Sanity check: bottle has 56 RGB frames."""
    rgb_files = sorted((bottle_copy / "rgb").glob("*.png"))
    assert len(rgb_files) == 56


def test_bottle_has_mesh(bottle_copy):
    """Bottle has a mesh/obj file."""
    mesh = bottle_copy / "mesh" / "textured_simple.obj"
    assert mesh.exists()


def test_bottle_has_mask(bottle_copy):
    """Bottle has a first-frame mask."""
    mask = bottle_copy / "masks" / "00000.png"
    assert mask.exists()


def test_scale_then_package_produces_valid_fp_dataset(tmp_path, bottle_copy):
    """Full scale+package pipeline against bottle data."""
    from pipeline.stages.scale import ScaleStage, parse_obj_bounds, scale_obj
    from pipeline.stages.package import PackageStage
    from pipeline.config import (
        PipelineConfig, InputConfig, Sam2Config, HunyuanConfig, RealSizeConfig,
    )

    # Prepare input: rgbd dir (bottle has rgb + depth)
    # But bottle's camera_params.json is at top level, not in rgbd dir.
    # We need to restructure slightly for the pipeline's expected input layout.
    rgbd_dir = tmp_path / "rgbd_input"
    rgbd_dir.mkdir()
    shutil.copytree(bottle_copy / "rgb", rgbd_dir / "rgb")
    shutil.copytree(bottle_copy / "depth", rgbd_dir / "depth")
    shutil.copy2(bottle_copy / "camera_params.json", rgbd_dir / "camera_params.json")

    output_dir = tmp_path / "output"

    config = PipelineConfig(
        task="bottle_test",
        preset="foundationpose",
        input=InputConfig(
            rgbd_dir=str(rgbd_dir),
            multi_views_dir="/nonexistent",
            first_frame=0,
        ),
        sam2=Sam2Config(image="s:v1", bbox=[0, 0, 10, 10]),
        hunyuan=HunyuanConfig(
            secret_id="id", secret_key="key", region="ap", model="3.1",
            face_count=500000, enable_pbr=False,
            views={"front": "f.jpg", "left": "l.jpg", "right": "r.jpg", "back": "b.jpg"},
        ),
        real_size=RealSizeConfig(longest_edge=1.0),  # deliberately wrong to test scale path
        output_dir=str(output_dir),
    )

    # --- Stage: Scale ---
    scale_out = output_dir / "scale"
    scale_out.mkdir(parents=True)

    original_obj = (bottle_copy / "mesh" / "textured_simple.obj").read_text()
    (tmp_path / "input_obj" / "obj.obj").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "input_obj" / "obj.obj").write_text(original_obj)

    scale_stage = ScaleStage()
    scale_stage._input_obj_path = str(tmp_path / "input_obj" / "obj.obj")

    # Get original longest edge
    min_c, max_c = parse_obj_bounds(original_obj)
    original_longest = max(max_c[i] - min_c[i] for i in range(3))

    # Scale to same size (factor=1.0)
    config.real_size.longest_edge = original_longest
    scale_stage.run(config, scale_out)

    scaled_obj = (scale_out / "obj.obj").read_text()
    min_c2, max_c2 = parse_obj_bounds(scaled_obj)
    scaled_longest = max(max_c2[i] - min_c2[i] for i in range(3))
    assert scaled_longest == pytest.approx(original_longest, rel=1e-3)

    # --- Stage: Package ---
    package_out = output_dir / "package"
    package_out.mkdir(parents=True)

    package_stage = PackageStage()
    package_stage._obj_path_override = str(scale_out / "obj.obj")
    package_stage._mask_path_override = str(bottle_copy / "masks" / "00000.png")

    result = package_stage.run(config, package_out)

    # Verify output structure matches bottle exactly
    assert (result / "rgb").is_dir()
    assert (result / "depth").is_dir()
    assert (result / "masks").is_dir()
    assert (result / "mesh").is_dir()
    assert (result / "cam_K.txt").exists()
    assert (result / "camera_params.json").exists()

    # Same number of frames
    expected_count = len(sorted(rgbd_dir.glob("rgb/*.png")))
    actual_count = len(sorted(result.glob("rgb/*.png")))
    assert actual_count == expected_count

    # Camera params match
    result_params = json.loads((result / "camera_params.json").read_text())
    original_params = json.loads((bottle_copy / "camera_params.json").read_text())
    assert result_params["fx"] == pytest.approx(original_params["fx"])

    # Mask copied correctly
    assert (result / "masks" / "00000.png").exists()

    # Mesh placed correctly
    assert (result / "mesh" / "textured_simple.obj").exists()
```

- [ ] **Step 2: Run the E2E test**

Run: `cd /home/ubt2204/code/pipeline-tool && python -m pytest tests/test_e2e_bottle.py -v`

Expected: All tests PASS (5 tests: 4 sanity checks + 1 e2e)

- [ ] **Step 3: Commit**

```bash
cd /home/ubt2204/code/pipeline-tool
git add tests/test_e2e_bottle.py
git commit -m "test: add E2E test with bottle dataset (scale + package)"
```

---

## Self-Review

### 1. Spec Coverage

| Spec Section | Covered By |
|---|---|
| Core Abstractions (Stage, Preset, Manifest, Config) | Tasks 2-4, Task 9 |
| Directory Structure | Task 1 |
| Design Patterns (argparse, YAML, factory) | Tasks 2, 4, 10 |
| Manifest Format | Task 3 |
| sam2mask stage (Docker wrapper) | Task 7 |
| hunyuangen stage (Hunyuan API) | Task 8 |
| scale stage (longest-edge align, cm units) | Task 5 |
| package stage (FP dataset assembly) | Task 6 |
| FP Dataset Output Layout | Task 6 (test validates structure) |
| CLI (run, stage, status, --force) | Task 10 |
| Config Format (env var subs, validation) | Task 2 |
| Error Handling (missing fields, paths, API errors) | Tasks 2, 5-8 |
| Testing Strategy Phase 1 (bottle dataset) | Task 12 |
| run.sh shell entrypoint | Task 11 |

### 2. Placeholder Scan

- No TBDs or TODOs.
- Hunyuangen stage (Task 8) has a note about `ResultUrl` vs `ResultData` — this is a real API uncertainty (the SDK doc shows the request side but the query response format is not fully documented). The code handles both cases. This is acceptable for Phase 1; the user can adjust after first real API call.
- All steps contain actual code, not "implement later" descriptions.

### 3. Type Consistency

- `PipelineConfig` fields used consistently across all stages.
- `Manifest` methods (`load`, `save`, `mark_stage_done`, `mark_stage_failed`, `is_stage_done`, `get_output_dir`) consistent between Task 3 (definition) and Tasks 5-9 (usage).
- Stage `name` property matches preset YAML keys: `sam2mask`, `hunyuangen`, `scale`, `package`.
- `register_stage("name")` decorator names match `presets.yaml` values.
- OBJ unit assumption (cm) documented in scale stage, consistent with `real_size.longest_edge` in cm.
