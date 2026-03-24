"""Inference CLI — run detection on images and save annotated results.

Usage::

    python -m src.pipelines.inference \\
        --image   path/to/image.jpg \\
        --weights experiments/exp01/checkpoints/best.pth \\
        --output  output/results/

    # Multiple images at once
    python -m src.pipelines.inference \\
        --image img1.jpg img2.jpg img3.jpg \\
        --weights best.pth

    # Adjust thresholds
    python -m src.pipelines.inference \\
        --image photo.jpg \\
        --weights best.pth \\
        --conf-thresh 0.4 \\
        --nms-thresh  0.45
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.inference.predictor import Predictor
from src.utils.visualization import draw_detections

# COCO 80-class names (0-indexed)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run object detection inference on images",
    )
    parser.add_argument(
        "--image", "-i", nargs="+", required=True,
        help="Path(s) to input image(s)",
    )
    parser.add_argument(
        "--weights", "-w", type=str, required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output/results",
        help="Directory to save annotated images (default: output/results)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=80,
        help="Number of foreground classes (default: 80 for COCO)",
    )
    parser.add_argument(
        "--image-size", type=int, default=640,
        help="Model input resolution (default: 640)",
    )
    parser.add_argument(
        "--conf-thresh", type=float, default=0.5,
        help="Confidence score threshold (default: 0.5)",
    )
    parser.add_argument(
        "--nms-thresh", type=float, default=0.5,
        help="NMS IoU threshold (default: 0.5)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (default: cuda)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build predictor ──────────────────────────────────────────────────
    print(f"Loading model from {args.weights} ...")
    predictor = Predictor(
        checkpoint_path=args.weights,
        num_classes=args.num_classes,
        device=args.device,
        image_size=args.image_size,
        conf_threshold=args.conf_thresh,
        nms_threshold=args.nms_thresh,
    )
    print("Model loaded.\n")

    # ── Run inference on each image ──────────────────────────────────────
    class_names = COCO_CLASSES if args.num_classes == 80 else None

    for img_path in args.image:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"[SKIP] File not found: {img_path}")
            continue

        print(f"Processing {img_path} ...")
        result = predictor.predict(str(img_path))

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

        print(f"  Detections: {len(boxes)}")
        for box, score, label in zip(boxes, scores, labels):
            name = class_names[int(label)] if class_names else str(int(label))
            print(f"    {name:>20s}  {score:.3f}  {box.astype(int).tolist()}")

        # Draw and save
        image_bgr = cv2.imread(str(img_path))
        annotated = draw_detections(
            image=image_bgr,
            boxes=boxes,
            labels=labels,
            scores=scores,
            class_names=class_names,
            score_threshold=0.0,  # already filtered by predictor
        )

        save_path = output_dir / f"{img_path.stem}_det{img_path.suffix}"
        cv2.imwrite(str(save_path), annotated)
        print(f"  Saved → {save_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
