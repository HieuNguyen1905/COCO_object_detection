"""Inference CLI — run detection on images and save annotated results.

Usage:

    # Single image with best checkpoint
    python -m src.pipelines.inference --image path/to/image.jpg

    # Multiple images
    python -m src.pipelines.inference --image img1.jpg img2.jpg img3.jpg

    # Use specific checkpoint
    python -m src.pipelines.inference --image photo.jpg --weights outputs/last.pth

    # Adjust thresholds
    python -m src.pipelines.inference \\
        --image photo.jpg \\
        --conf-thresh 0.3 \\
        --nms-thresh 0.4

    # Process all images in folder
    python -m src.pipelines.inference --image path/to/folder/ --output results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from src.inference import Predictor
from src.utils import draw_detections, load_config

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
        description="Run Faster R-CNN object detection inference on images",
    )
    parser.add_argument(
        "--image", "-i", nargs="+", required=True,
        help="Path(s) to input image(s) or folder",
    )
    parser.add_argument(
        "--weights", "-w", type=str, default=None,
        help="Path to model checkpoint (.pth). Default: outputs/best.pth",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Directory to save annotated images (default: from config)",
    )
    parser.add_argument(
        "--conf-thresh", type=float, default=None,
        help="Confidence score threshold (default: from config)",
    )
    parser.add_argument(
        "--nms-thresh", type=float, default=None,
        help="NMS IoU threshold (default: from config)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for inference (default: auto-detect cuda/cpu)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save annotated images, only print detections",
    )
    return parser.parse_args()


def get_image_paths(sources: list[str]) -> list[Path]:
    """Expand sources to list of image paths (support folders)."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    paths = []

    for source in sources:
        path = Path(source)
        if path.is_dir():
            # Add all images in folder
            for ext in image_extensions:
                paths.extend(path.glob(f"*{ext}"))
                paths.extend(path.glob(f"*{ext.upper()}"))
        elif path.is_file():
            if path.suffix.lower() in image_extensions:
                paths.append(path)
            else:
                print(f"[SKIP] Not an image: {path}")
        else:
            print(f"[SKIP] File not found: {path}")

    return sorted(set(paths))


def main() -> None:
    args = parse_args()

    # Load config
    config = load_config("configs/configs.yaml")

    # Determine weights path
    weights_path = args.weights
    if weights_path is None:
        weights_path = Path(config["BASE_OUTPUT"]) / "best.pth"
        if not weights_path.exists():
            weights_path = Path(config["BASE_OUTPUT"]) / "last.pth"
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"No checkpoint found. Please specify --weights or train a model first."
                )
        print(f"Using checkpoint: {weights_path}")

    # Determine device
    device = args.device
    if device is None:
        device_config = config.get("DEVICE", "auto")
        if device_config == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_config
    print(f"Using device: {device}")

    # Get number of classes from config
    num_classes = config.get("NUM_CLASSES", 81)

    # Get thresholds from config or CLI args
    conf_threshold = args.conf_thresh if args.conf_thresh is not None else config.get("CONF_THRESHOLD", 0.5)
    nms_threshold = args.nms_thresh if args.nms_thresh is not None else config.get("NMS_THRESHOLD", 0.5)

    # Get output directory
    output_dir = Path(args.output) if args.output else Path(config.get("INFERENCE_OUTPUT", "outputs/inference"))
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build predictor using config
    print(f"\nLoading model from {weights_path} ...")
    predictor = Predictor(
        checkpoint_path=weights_path,
        config=config,
        device=device,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )
    print("Model loaded.\n")

    # Get all image paths
    image_paths = get_image_paths(args.image)
    if not image_paths:
        print("No valid images found.")
        return

    print(f"Found {len(image_paths)} image(s) to process.\n")

    # Run inference on each image
    class_names = COCO_CLASSES

    for img_path in image_paths:
        print(f"Processing {img_path.name} ...")
        result = predictor.predict(str(img_path))

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

        print(f"  Detections: {len(boxes)}")
        if len(boxes) > 0:
            for box, score, label in zip(boxes, scores, labels):
                label_idx = int(label)
                name = class_names[label_idx] if 0 <= label_idx < len(class_names) else f"class_{label_idx}"
                x1, y1, x2, y2 = box.astype(int)
                print(f"    {name:>20s}  {score:.3f}  [{x1}, {y1}, {x2}, {y2}]")

        # Draw and save
        if not args.no_save:
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                print(f"  [ERROR] Cannot read image: {img_path}")
                continue

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
        else:
            print()

    print("Done.")


if __name__ == "__main__":
    main()
