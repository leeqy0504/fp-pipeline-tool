"""Pipeline orchestrator: resolve preset, run stages in sequence."""

import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from pipeline.config import PipelineConfig
from pipeline.manifest import Manifest
from pipeline.stages import get_stage

if TYPE_CHECKING:
    from pipeline.stages.context import StageContext


class PipelineOrchestrator:
    """Runs presets by dispatching stages in order, tracking progress via manifest."""

    def __init__(self, presets_path: str | None = None):
        if presets_path is None:
            presets_path = str(Path(__file__).parent / "presets.yaml")
        with open(presets_path) as f:
            self.presets = yaml.safe_load(f)

    def resolve_preset(self, preset_name: str) -> list[str]:
        if preset_name not in self.presets:
            raise KeyError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(self.presets.keys())}"
            )
        return list(self.presets[preset_name]["stages"])

    def _manifest_path(self, config: PipelineConfig) -> str:
        return str(Path(config.output_dir) / config.task / "manifest.json")

    def _stage_output_dir(self, config: PipelineConfig, stage_name: str) -> str:
        return str(Path(config.output_dir) / config.task / stage_name)

    def run_preset(self, config: PipelineConfig, force: bool = False,
                   context: "StageContext | None" = None):
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

            start = time.time()
            try:
                result_path = stage.run(config, output_dir, context=context)
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

    def run_stage(self, config: PipelineConfig, stage_name: str, force: bool = False,
                  context: "StageContext | None" = None):
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

        start = time.time()
        try:
            result_path = stage.run(config, output_dir, context=context)
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
