"""Worker process for running local Hunyuan3D generation in its own Python env."""

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, "/home/try/code/Hunyuan3D-2")

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


def load_images(payload: dict) -> dict:
    views_dir = Path(payload["views_dir"])
    images = {}
    rembg = None

    for view_name, filename in payload["views"].items():
        image_path = views_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"View image ({view_name}) not found: {image_path}")

        image = Image.open(image_path)
        if payload["remove_background"] and image.mode == "RGB":
            if rembg is None:
                rembg = BackgroundRemover()
            image = rembg(image)
        else:
            image = image.convert("RGBA")
        images[view_name] = image

    if "front" not in images:
        raise ValueError("No 'front' view found in payload")
    return images


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python hunyuan_worker.py payload.json", file=sys.stderr)
        return 2

    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading views from {payload['views_dir']}", flush=True)
    images = load_images(payload)
    print(f"Loaded views: {', '.join(images.keys())}", flush=True)
    print(f"Loading model: {payload['model']}", flush=True)

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        payload["model"],
        subfolder=payload["subfolder"],
        variant=payload["variant"],
    )

    start_time = time.time()
    mesh = pipeline(
        image=images,
        num_inference_steps=payload["num_inference_steps"],
        octree_resolution=payload["octree_resolution"],
        num_chunks=payload["num_chunks"],
        generator=torch.manual_seed(payload["seed"]),
        output_type=payload["output_type"],
    )[0]
    print(f"Generation completed in {time.time() - start_time:.1f}s", flush=True)

    glb_path = output_dir / "result.glb"
    obj_path = output_dir / "raw.obj"
    mesh.export(str(glb_path))
    mesh.export(str(obj_path))
    print(f"Saved GLB: {glb_path}", flush=True)
    print(f"Saved OBJ: {obj_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
