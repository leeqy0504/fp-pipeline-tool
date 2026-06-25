import importlib.util
from pathlib import Path


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "sam2" / "sam2_video_cli.py"
    spec = importlib.util.spec_from_file_location("sam2_video_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sam2_video_cli_imports_without_sam2_dependencies():
    module = _load_cli_module()

    assert module.__name__ == "sam2_video_cli"


def test_sam2_video_cli_detects_prompt_mask_mode(tmp_path):
    module = _load_cli_module()
    prompt_mask = tmp_path / "00000.png"
    prompt_mask.write_bytes(b"mask")

    assert module._should_use_prompt_mask(str(prompt_mask)) is True
    assert module._should_use_prompt_mask("") is False


def test_sam2_video_cli_uses_prompt_mask_when_predictor_supports_it(tmp_path):
    module = _load_cli_module()
    prompt_mask = tmp_path / "00000.png"
    prompt_mask.write_bytes(b"mask")
    calls = []

    class FakeMask:
        def __gt__(self, other):
            return [[False, True]]

    fake_mask = FakeMask()

    class FakePredictor:
        def add_new_mask(self, **kwargs):
            calls.append(("mask", kwargs))

        def add_new_points_or_box(self, **kwargs):
            calls.append(("points", kwargs))

    class FakeCv2:
        IMREAD_GRAYSCALE = 0

        @staticmethod
        def imread(path, flags):
            return fake_mask

    prompt_mode = module._add_initial_prompt(
        predictor=FakePredictor(),
        state=object(),
        first_frame=0,
        points=[[10, 20]],
        labels=[1],
        prompt_mask=str(prompt_mask),
        cv2=FakeCv2,
    )

    assert prompt_mode == "mask"
    assert calls[0][0] == "mask"


def test_sam2_video_cli_falls_back_to_points_without_mask_api(tmp_path):
    module = _load_cli_module()
    prompt_mask = tmp_path / "00000.png"
    prompt_mask.write_bytes(b"mask")
    calls = []
    fake_mask = [[0, 255]]

    class FakePredictor:
        def add_new_points_or_box(self, **kwargs):
            calls.append(("points", kwargs))

    class FakeCv2:
        IMREAD_GRAYSCALE = 0

        @staticmethod
        def imread(path, flags):
            return fake_mask

    prompt_mode = module._add_initial_prompt(
        predictor=FakePredictor(),
        state=object(),
        first_frame=0,
        points=[[10, 20]],
        labels=[1],
        prompt_mask=str(prompt_mask),
        cv2=FakeCv2,
    )

    assert prompt_mode == "points"
    assert calls[0][0] == "points"
