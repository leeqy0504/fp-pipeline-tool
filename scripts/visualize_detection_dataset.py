"""Visualize and export a generated detection dataset.

Usage:
    python scripts/visualize_detection_dataset.py detection_dataset
"""

import argparse
import base64
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _resolve_dataset_path(dataset_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = dataset_dir / path.name
    if candidate.exists():
        return candidate
    images_candidate = dataset_dir / "images" / path.name
    if images_candidate.exists():
        return images_candidate
    return path


def _draw_preview(image_path: Path, output_path: Path, box: list[int], label: str):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box

    color = (255, 40, 40)
    width = max(2, image.width // 320)
    for offset in range(width):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)

    font = ImageFont.load_default()
    text = f"{label} {x2 - x1}x{y2 - y1}"
    text_box = draw.textbbox((x1, y1), text, font=font)
    text_h = text_box[3] - text_box[1]
    text_w = text_box[2] - text_box[0]
    bg_y1 = max(0, y1 - text_h - 6)
    draw.rectangle((x1, bg_y1, min(image.width - 1, x1 + text_w + 8), y1), fill=color)
    draw.text((x1 + 4, bg_y1 + 3), text, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _write_labelme_json(image_path: Path, output_path: Path, box: list[int], label: str):
    image = Image.open(image_path)
    x1, y1, x2, y2 = box
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
            }
        ],
        "imagePath": image_path.name,
        "imageData": image_data,
        "imageHeight": image.height,
        "imageWidth": image.width,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def _write_contact_sheet(preview_paths: list[Path], output_path: Path, thumb_width: int = 240, columns: int = 5):
    thumbs: list[Image.Image] = []
    for path in preview_paths:
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / image.width
        thumb_height = int(image.height * ratio)
        image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        thumbs.append(image)

    if not thumbs:
        return

    rows = (len(thumbs) + columns - 1) // columns
    thumb_height = thumbs[0].height
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), (245, 245, 245))
    for idx, thumb in enumerate(thumbs):
        x = (idx % columns) * thumb_width
        y = (idx // columns) * thumb_height
        sheet.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Create preview images and LabelMe JSON from detection annotations.")
    parser.add_argument("dataset_dir", help="Dataset directory containing annotations.json")
    parser.add_argument("--preview-dir", default="preview", help="Preview output subdirectory")
    parser.add_argument("--labelme-dir", default="labelme", help="LabelMe JSON output subdirectory")
    parser.add_argument("--contact-sheet", default="contact_sheet.png", help="Contact sheet filename")
    parser.add_argument("--contact-columns", type=int, default=5, help="Contact sheet column count")
    parser.add_argument("--no-preview", action="store_true", help="Skip preview image generation")
    parser.add_argument("--no-labelme", action="store_true", help="Skip LabelMe JSON generation")
    parser.add_argument("--no-contact-sheet", action="store_true", help="Skip contact sheet generation")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    annotations_path = dataset_dir / "annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"annotations.json not found: {annotations_path}")

    data = json.loads(annotations_path.read_text())
    annotations = data.get("annotations", [])
    if not annotations:
        raise RuntimeError(f"No annotations found in {annotations_path}")

    preview_dir = dataset_dir / args.preview_dir
    labelme_dir = dataset_dir / args.labelme_dir
    preview_paths: list[Path] = []
    count = 0
    for ann in annotations:
        image_path = _resolve_dataset_path(dataset_dir, ann["image"])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for annotation: {ann['image']}")

        stem = image_path.stem
        label = ann.get("class_name", data.get("class_name", "object"))
        box = ann["bbox_xyxy"]

        if not args.no_preview:
            preview_path = preview_dir / f"{stem}.png"
            _draw_preview(image_path, preview_path, box, label)
            preview_paths.append(preview_path)
        if not args.no_labelme:
            _write_labelme_json(image_path, labelme_dir / f"{stem}.json", box, label)
        count += 1

    contact_sheet_path = dataset_dir / args.contact_sheet
    if not args.no_preview and not args.no_contact_sheet:
        _write_contact_sheet(preview_paths, contact_sheet_path, columns=args.contact_columns)

    outputs = []
    if not args.no_preview:
        outputs.append(str(preview_dir))
    if not args.no_preview and not args.no_contact_sheet:
        outputs.append(str(contact_sheet_path))
    if not args.no_labelme:
        outputs.append(str(labelme_dir))
    print(f"Generated {count} annotation visualizations -> {', '.join(outputs)}")


if __name__ == "__main__":
    main()
