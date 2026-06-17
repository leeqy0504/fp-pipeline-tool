from pipeline.config import (
    HunyuanConfig,
    InputConfig,
    PipelineConfig,
    RealSizeConfig,
    Sam2Config,
)
from pipeline.stages.hunyuangen import HunyuanGenStage


class FakeImageObj:
    mode = "RGBA"

    def convert(self, mode):
        self.mode = mode
        return self


class FakeImage:
    opened = []

    @staticmethod
    def open(path):
        FakeImage.opened.append(str(path))
        return FakeImageObj()


class FakeTorch:
    @staticmethod
    def manual_seed(seed):
        return f"seed:{seed}"


class FakeMesh:
    def export(self, path):
        with open(path, "w") as f:
            f.write("mesh")


class FakePipeline:
    loaded = {}
    called = {}

    @classmethod
    def from_pretrained(cls, model, subfolder=None, variant=None):
        cls.loaded = {"model": model, "subfolder": subfolder, "variant": variant}
        return cls()

    def __call__(self, **kwargs):
        FakePipeline.called = kwargs
        return [FakeMesh()]


class FakeBackgroundRemover:
    def __call__(self, image):
        return image


def make_config(views_dir):
    return PipelineConfig(
        task="test",
        preset="foundationpose",
        input=InputConfig(rgbd_dir="/tmp/r", multi_views_dir=str(views_dir), first_frame=0),
        sam2=Sam2Config(container="sam2-test", points=[[0, 0]], labels=[1]),
        hunyuan=HunyuanConfig(
            views={"front": "front.jpg", "left": "left.jpg", "back": "back.jpg"},
            num_inference_steps=12,
            octree_resolution=128,
            num_chunks=512,
            seed=99,
        ),
        real_size=RealSizeConfig(longest_edge=1.0),
        output_dir="output/",
    )


def test_hunyuangen_runs_local_multiview_model_and_exports_mesh(tmp_path, monkeypatch):
    views_dir = tmp_path / "views"
    views_dir.mkdir()
    for name in ("front.jpg", "left.jpg", "back.jpg"):
        (views_dir / name).write_bytes(b"image")

    monkeypatch.setattr(
        "pipeline.stages.hunyuangen._load_hunyuan3d",
        lambda: (FakeTorch, FakeImage, FakeBackgroundRemover, FakePipeline),
    )

    result = HunyuanGenStage().run(make_config(views_dir), tmp_path / "out")

    assert result == tmp_path / "out"
    assert (result / "result.glb").read_text() == "mesh"
    assert (result / "raw.obj").read_text() == "mesh"
    assert FakePipeline.loaded == {
        "model": "tencent/Hunyuan3D-2mv",
        "subfolder": "hunyuan3d-dit-v2-mv",
        "variant": "fp16",
    }
    assert set(FakePipeline.called["image"].keys()) == {"front", "left", "back"}
    assert FakePipeline.called["num_inference_steps"] == 12
    assert FakePipeline.called["octree_resolution"] == 128
    assert FakePipeline.called["num_chunks"] == 512
    assert FakePipeline.called["generator"] == "seed:99"
    assert FakePipeline.called["output_type"] == "trimesh"
