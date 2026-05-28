"""CLI entry point for the pipeline tool."""

import argparse
import sys
from pathlib import Path

import yaml

from pipeline.config import load_config, ConfigError
from pipeline.pipeline import PipelineOrchestrator


def cmd_run(args):
    config = load_config(args.config)
    if args.preset:
        config.preset = args.preset

    orch = PipelineOrchestrator()
    orch.run_preset(config, force=args.force)


def cmd_stage(args):
    config = load_config(args.config)

    orch = PipelineOrchestrator()
    orch.run_stage(config, args.stage_name, force=args.force)


def cmd_status(args):
    config = load_config(args.config)

    orch = PipelineOrchestrator()
    orch.status(config)


def cmd_setup(args):
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    task_name = args.task

    task_dir = config_path.parent.parent / "tasks" / task_name
    if not task_dir.is_dir():
        print(f"Error: Task directory not found: {task_dir}", file=sys.stderr)
        sys.exit(1)

    raw["task"] = task_name
    raw["input"]["rgbd_dir"] = f"./tasks/{task_name}/"
    raw["input"]["multi_views_dir"] = f"./tasks/{task_name}/views/"

    # 从 dataset_info.json 更新 sam2 points/labels 和 real_size
    for ds_path in [task_dir / "dataset_info.json"]:
        if ds_path.exists():
            import json
            with open(ds_path) as f:
                ds = json.load(f)
            sam2_data = ds.get("sam2_points", {})
            pts = sam2_data.get("points", [])
            lbls = sam2_data.get("labels", [])
            if pts and lbls and len(pts) == len(lbls):
                raw["sam2"]["points"] = pts
                raw["sam2"]["labels"] = lbls
                print(f"  sam2 pts   -> {pts} (from {ds_path})")
            else:
                print(f"  [warn] dataset_info.json found but sam2_points invalid, keeping config defaults")
            # real_size
            rs = ds.get("real_size")
            if rs:
                raw["real_size"] = rs
                print(f"  real_size  -> {rs} (from {ds_path})")
            break

    with open(config_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Config updated: task={task_name}")
    print(f"  rgbd_dir   -> ./tasks/{task_name}/")
    print(f"  views_dir  -> ./tasks/{task_name}/views/")


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

    # pipeline setup --config <path> --task <name>
    setup_parser = subparsers.add_parser("setup", help="Update config with a new task")
    setup_parser.add_argument("--config", required=True, help="Path to YAML config file")
    setup_parser.add_argument("--task", required=True, help="Task name")
    setup_parser.set_defaults(func=cmd_setup)

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


if __name__ == "__main__":
    main()
