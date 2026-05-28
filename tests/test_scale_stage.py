import pytest
from pathlib import Path
from pipeline.config import PipelineConfig, InputConfig, Sam2Config, HunyuanConfig, RealSizeConfig
from pipeline.stages.scale import ScaleStage, parse_obj_bounds, scale_obj


def make_config(real_longest_edge=5.0):
    return PipelineConfig(
        task="test",
        preset="foundationpose",
        input=InputConfig(rgbd_dir="/tmp/r", multi_views_dir="/tmp/v", first_frame=0),
        sam2=Sam2Config(container="sam2-test", points=[[0, 0], [10, 10]], labels=[1, 1]),
        hunyuan=HunyuanConfig(
            secret_id="id", secret_key="key", region="ap", model="3.1",
            face_count=500000, enable_pbr=False, views={"front": "f.jpg"}
        ),
        real_size=RealSizeConfig(longest_edge=real_longest_edge),
        output_dir="output/",
    )


def test_parse_obj_bounds():
    obj = """v 0.0 0.0 0.0
v 3.0 0.0 0.0
v 0.0 4.0 0.0
v 0.0 0.0 5.0
f 1 2 3
"""
    min_corner, max_corner = parse_obj_bounds(obj)
    assert min_corner == pytest.approx((0.0, 0.0, 0.0))
    assert max_corner == pytest.approx((3.0, 4.0, 5.0))


def test_parse_obj_bounds_longest_edge():
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
    result = scale_obj(original, 4.0)
    min_c, max_c = parse_obj_bounds(result)
    extents = [max_c[i] - min_c[i] for i in range(3)]
    assert max(extents) == pytest.approx(4.0)
    assert min_c == pytest.approx((0.0, 0.0, 0.0))
    assert max_c == pytest.approx((4.0, 4.0, 4.0))


def test_scale_stage_run(tmp_path):
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
    stage._input_obj_path = str(input_obj)

    result = stage.run(config, output_dir)

    assert (result / "obj.obj").exists()
    scaled = (result / "obj.obj").read_text()
    min_c, max_c = parse_obj_bounds(scaled)
    assert max(max_c[i] - min_c[i] for i in range(3)) == pytest.approx(10.0)
