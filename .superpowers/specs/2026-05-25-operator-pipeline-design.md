# Operator Model Development Pipeline - Design

## Overview

A CLI pipeline tool that automates data preparation for operator model training pipelines. Phase 1 targets FoundationPose dataset generation. Phase 2 extends to object detection/tracking datasets.

The tool orchestrates multiple stages (SAM2 mask generation, 3D model generation via Hunyuan API, scale correction, dataset packaging) into a single command, tracking progress through a manifest file for resumability.

## Architecture

```
┌─ first frame RGB + bbox ────→ sam2mask ──→ mask.png ───────────────┐
│                  (docker run sam2)                                  │
│                                                                     ↓
├─ multi-view images + API key → hunyuangen ──→ obj ──→ scale ──→ obj│
│                                    (HTTP API)   (local)             │
│                                                                     │
└─ RGB-D frames + intrinsics ────────────────────────────→ package ───┤
                                                                      │
                                                                      ↓
                                                             FP dataset
```

### Core Abstractions

| Concept | Role |
|---|---|
| **Stage** | Single processing unit: `run(input, manifest) -> output`. Stateless. Abstracted as `BaseStage` ABC. |
| **Preset** | Named sequence of stages defined in `presets.yaml`. e.g., `foundationpose` = `sam2mask -> hunyuangen -> scale -> package`. |
| **Manifest** | JSON file tracking per-stage status, output paths, and timing. Enables resume (skip done stages) and debugging. |
| **Config** | Single YAML input defining all task parameters. Validated before execution. |

### Directory Structure

```
pipeline-tool/
├── run.sh                 # Shell orchestrator (follows unitrain pattern)
├── pyproject.toml
├── configs/               # Example YAML configs
│   └── foundationpose.yaml
├── pipeline/
│   ├── __init__.py
│   ├── cli.py             # argparse CLI entry point
│   ├── config.py          # PipelineConfig dataclass + load_config()
│   ├── manifest.py        # Manifest read/write
│   ├── pipeline.py        # Pipeline orchestrator (run preset, stage dispatch)
│   ├── stages/
│   │   ├── __init__.py    # get_stage() factory
│   │   ├── base.py        # BaseStage ABC
│   │   ├── sam2mask.py    # SAM2 Docker wrapper
│   │   ├── hunyuangen.py  # Hunyuan 3D API client
│   │   ├── scale.py       # OBJ scale correction
│   │   └── package.py     # FP dataset packager
│   └── presets.yaml       # Preset stage chains
└── output/                # Task outputs (gitignored)
```

### Design Patterns (from unitrain)

- argparse CLI (not Click)
- YAML config with nested structure
- Factory pattern for stage resolution (`get_stage(name) -> BaseStage`)
- Dataclass for config with validation on load
- Shell entrypoint (`run.sh`) for environment management

### Manifest Format

```json
{
  "task": "mouse_001",
  "config_path": "configs/mouse_001.yaml",
  "created_at": "2026-05-25T14:30:00",
  "stages": {
    "sam2mask":    {"status": "done",    "output_dir": "output/mouse_001/sam2mask/",  "duration_s": 5},
    "hunyuangen":  {"status": "done",    "output_dir": "output/mouse_001/hunyuan/",    "duration_s": 45},
    "scale":       {"status": "done",    "output_dir": "output/mouse_001/scaled/",      "duration_s": 2},
    "package":     {"status": "pending", "output_dir": null,                           "duration_s": null}
  }
}
```

## Stages

### sam2mask

- **Input**: First-frame RGB image, bounding box coordinates (from config)
- **Method**: `docker run --rm -v ... sam2:latest python infer.py --image <rgb> --bbox <x1,y1,x2,y2> --output <mask.png>`
- **Output**: Single mask PNG (same resolution as RGB)

### hunyuangen

- **Input**: Multi-view images directory, view-angle mapping (from config), API credentials
- **SDK**: `tencentcloud-sdk-python` (Ai3dClient, apiVersion `2025-05-13`)
- **Method**:
  1. Load multi-view images, encode as Base64
  2. Call `SubmitHunyuanTo3DProJob` with `MultiViewImages` array, `GenerateType: Normal`, `Model: 3.1`
  3. Poll `QueryHunyuanTo3DProJob` by `JobId` until status complete
  4. Download result `.obj` file
- **Output**: `.obj` file (textured, arbitrary scale)
- **Config fields**:

```yaml
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
```

### scale

- **Input**: `.obj` from hunyuangen, real-world longest-edge length (cm) from config
- **Method**: Parse OBJ, compute bounding box (OBJ units = cm, same as config), find longest edge, compute `scale_factor = real_longest / current_longest`, apply uniform scale
- **Output**: Scaled `.obj` file (units: cm)

### package

- **Input**: RGB-D frames directory, mask PNG, camera intrinsics (from RGB-D capture script), scaled `.obj`
- **Method**: Copy/rename files into FP dataset layout. Rename obj to `textured_simple.obj`. Generate `cam_K.txt` and `camera_params.json` from intrinsics.
- **Output**: FP-ready dataset directory

## FP Dataset Output Layout

Aligns with existing datasets (e.g., `bottle`):

```
fp_dataset/
├── rgb/                        # 00000.png, 00001.png, ...
├── depth/                      # 00000.png, 00001.png, ...
├── masks/
│   └── 00000.png               # First-frame mask
├── mesh/
│   └── textured_simple.obj     # Scaled 3D model
├── cam_K.txt                   # 3x3 intrinsics matrix
└── camera_params.json          # {width, height, fx, fy, ppx, ppy, depth_scale}
```

## CLI

```
# Run full preset pipeline
pipeline run foundationpose --config configs/mouse_001.yaml

# Run individual stage (debugging)
pipeline stage sam2mask --config configs/mouse_001.yaml
pipeline stage hunyuangen --config configs/mouse_001.yaml
pipeline stage scale --config configs/mouse_001.yaml
pipeline stage package --config configs/mouse_001.yaml

# Force re-run a stage even if already done
pipeline stage hunyuangen --config configs/mouse_001.yaml --force

# Check task status
pipeline status --config configs/mouse_001.yaml
```

## Config Format

```yaml
task: mouse_001
preset: foundationpose

input:
  rgbd_dir: /data/mouse_001/rgbd/
  multi_views_dir: /data/mouse_001/views/
  first_frame: 0

sam2:
  image: sam2:latest
  bbox: [320, 240, 480, 360]    # x1, y1, x2, y2

hunyuan:
  secret_id: ${TENCENT_SECRET_ID}
  secret_key: ${TENCENT_SECRET_KEY}
  region: ap-guangzhou
  model: "3.1"
  face_count: 500000
  enable_pbr: false
  views:                          # 文件名按角度命名
    front: front.jpg
    left: left.jpg
    right: right.jpg
    back: back.jpg

real_size:
  longest_edge: 5.2  # cm, object's longest side measured physically

output_dir: output/
```

Environment variable references (`${VAR}`) resolved at load time.

## Error Handling

| Scenario | Behavior |
|---|---|
| Config missing required fields | Validate before execution, list all missing fields |
| Input path does not exist | Stage checks input before starting, reports exact path |
| Hunyuan API timeout/error | Manifest marked `failed`, resume will re-run this stage |
| SAM2 Docker unavailable | Immediate stop, prompt to check image |
| Stage output already exists | Skip (idempotent by default), `--force` to re-run |

## Testing Strategy

- **Phase 1 (minimal closed loop)**: E2E test using existing `bottle` dataset — skip `sam2mask` and `hunyuangen`, test `scale` + `package` only, verify output matches bottle format
- **Phase 2**: Add integration test with mock Hunyuan API
- **Phase 3**: Full E2E with real SAM2 Docker + Hunyuan API

## Future Extension

For object detection/tracking datasets (Phase 2):
- Reference existing detection dataset formats (COCO, YOLO)
- Add new stages: `convert_to_coco`, `convert_to_yolo`
- Add new presets: `detection`, `tracking`
- Stage interface designed to support this without breaking changes

---

## Spec Self-Review

- Placeholder scan: No TBDs or TODOs.
- Internal consistency: All 4 stages have defined inputs/outputs/methods. Manifest format matches pipeline.py's orchestration model. Config YAML fields cover all stage inputs.
- Scope check: Focused on FP data prep pipeline. Detection/tracking extension is noted as future phase but not scoped here.
- Ambiguity check: All field names have explicit types and semantics. Manifest states are enumerated (pending/done/failed). FP dataset layout pinned to bottle reference set.
