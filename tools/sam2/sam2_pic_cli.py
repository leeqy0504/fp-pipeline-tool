import argparse
import json
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def main():
    parser = argparse.ArgumentParser(description="SAM2 单张图片分割 CLI")
    parser.add_argument("--image",      required=True,  help="输入图片路径")
    parser.add_argument("--points",     required=True,  help='点坐标，格式: "x1,y1 x2,y2"')
    parser.add_argument("--labels",     required=True,  help='点标签，格式: "1 0"')
    parser.add_argument("--output",     required=True,  help="输出 mask 路径（.png）")
    parser.add_argument("--checkpoint", default="/opt/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--config",     default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    args = parser.parse_args()

    # 解析点和标签
    points = [list(map(int, p.split(","))) for p in args.points.strip().split()]
    labels = list(map(int, args.labels.strip().split()))

    assert len(points) == len(labels), "points 和 labels 数量必须一致"

    # 加载模型
    predictor = SAM2ImagePredictor(
        build_sam2(args.config, args.checkpoint)
    )

    # 读取图片
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"图片读取失败，请检查路径：{args.image}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 推理
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=False
        )

    # 保存 mask
    mask = masks[0].astype(np.uint8) * 255
    cv2.imwrite(args.output, mask)

    # 输出结果供 pipeline 解析
    print(json.dumps({
        "score":             float(scores[0]),
        "mask_path":         args.output,
        "foreground_pixels": int(masks[0].sum()),
        "mask_shape":        list(mask.shape),
    }))


if __name__ == "__main__":
    main()