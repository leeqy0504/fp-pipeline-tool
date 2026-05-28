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
