from pipeline.config import (
    HunyuanConfig,
    InputConfig,
    PipelineConfig,
    RealSizeConfig,
    Sam2Config,
)
from pipeline.stages.hunyuangen import HunyuanGenStage


class FakeResponse:
    status_code = 200
    content = b"glb-data"
    text = ""


def make_config(views_dir):
    return PipelineConfig(
        task="test",
        preset="foundationpose",
        input=InputConfig(rgbd_dir="/tmp/r", multi_views_dir=str(views_dir), first_frame=0),
        sam2=Sam2Config(container="sam2-test", points=[[0, 0]], labels=[1]),
        hunyuan=HunyuanConfig(
            views={"front": "front.jpg"},
            api_host="127.0.0.1",
            api_port=8081,
            api_timeout=30,
        ),
        real_size=RealSizeConfig(longest_edge=1.0),
        output_dir="output/",
    )


def test_hunyuangen_calls_local_api_and_writes_raw_obj(tmp_path, monkeypatch):
    views_dir = tmp_path / "views"
    views_dir.mkdir()
    (views_dir / "front.jpg").write_bytes(b"image")

    calls = {}

    def fake_post(url, json, timeout):
        calls["url"] = url
        calls["payload"] = json
        calls["timeout"] = timeout
        return FakeResponse()

    def fake_convert(glb_path, obj_path):
        assert glb_path.name == "result.glb"
        obj_path.write_text("v 0 0 0\n")

    monkeypatch.setattr("pipeline.stages.hunyuangen.requests.post", fake_post)
    monkeypatch.setattr("pipeline.stages.hunyuangen._convert_glb_to_obj", fake_convert)

    result = HunyuanGenStage().run(make_config(views_dir), tmp_path / "out")

    assert result == tmp_path / "out"
    assert calls["url"] == "http://127.0.0.1:8081/generate"
    assert calls["timeout"] == 30
    assert "image" in calls["payload"]
    assert (result / "result.glb").read_bytes() == b"glb-data"
    assert (result / "raw.obj").read_text() == "v 0 0 0\n"
