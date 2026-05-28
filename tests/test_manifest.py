import json
from pathlib import Path
from pipeline.manifest import Manifest


def test_manifest_create():
    m = Manifest(task="test_task", config_path="configs/test.yaml")
    assert m.task == "test_task"
    assert m.config_path == "configs/test.yaml"
    assert m.stages == {}


def test_manifest_mark_stage():
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


def test_manifest_is_stage_done():
    m = Manifest(task="test", config_path="c.yaml")
    m.mark_stage_done("a", "out/a", 1.0)
    m.mark_stage_failed("b")

    assert m.is_stage_done("a") is True
    assert m.is_stage_done("b") is False
    assert m.is_stage_done("c") is False


def test_manifest_to_dict():
    m = Manifest(task="test", config_path="c.yaml")
    m.mark_stage_done("scale", "out/s", 1.0)
    d = m.to_dict()

    assert d["task"] == "test"
    assert d["stages"]["scale"]["status"] == "done"
    assert "created_at" in d
