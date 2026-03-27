"""Predictor: load a trained Faster R-CNN model and run inference on images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.ops import nms

from src.models import build_model
from src.utils import get_inference_transforms


class Predictor:
    """End-to-end inference pipeline for Faster R-CNN object detection.

    1. Load model weights from a checkpoint.pth file.
    2. Preprocess input image using validation transforms.
    3. Run model forward pass.
    4. Apply confidence thresholding + NMS.
    5. Return filtered detections.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the saved .pth file (checkpoint with "model" key).
    config : dict, optional
        Configuration dictionary. If provided, parameters will be taken from config.
        Individual parameters can override config values.
    num_classes : int, optional
        Number of classes including background (e.g., 81 for COCO).
        If None and config provided, uses config["NUM_CLASSES"].
    device : str | torch.device, optional
        Target device for inference. If None and config provided, uses config["DEVICE"].
    image_size : int, optional
        Input resolution. If None and config provided, uses config["IMAGE_SIZE"].
    conf_threshold : float, optional
        Minimum confidence score to keep a detection. If None and config provided, uses config["CONF_THRESHOLD"].
    nms_threshold : float, optional
        IoU threshold for Non-Maximum Suppression. If None and config provided, uses config["NMS_THRESHOLD"].
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: dict = None,
        num_classes: int = None,
        device: str | torch.device = None,
        image_size: int = None,
        conf_threshold: float = None,
        nms_threshold: float = None,
    ) -> None:
        # Use config values if provided, otherwise use explicit params or defaults
        if config is not None:
            num_classes = num_classes or config.get("NUM_CLASSES", 81)
            device = device or config.get("DEVICE", "cuda")
            image_size = image_size or config.get("IMAGE_SIZE", 640)
            conf_threshold = conf_threshold or config.get("CONF_THRESHOLD", 0.5)
            nms_threshold = nms_threshold or config.get("NMS_THRESHOLD", 0.5)
        else:
            num_classes = num_classes or 81
            device = device or "cuda"
            image_size = image_size or 640
            conf_threshold = conf_threshold or 0.5
            nms_threshold = nms_threshold or 0.5

        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.image_size = image_size
        self.num_classes = num_classes
        self.config = config

        # Build model using config if available, otherwise use legacy method
        if config is not None:
            self.model = build_model(config, num_classes)
        else:
            # Legacy compatibility: create minimal config
            legacy_config = {
                "BASE_MODEL": "fasterrcnn_resnet50_fpn",
                "PRETRAINED": False,
                "NUM_CLASSES": num_classes,
                "TRAINABLE_BACKBONE_LAYERS": 3
            }
            self.model = build_model(legacy_config, num_classes)

        self._load_weights(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Inference transform (no bbox_params, no augmentations)
        self.transform = get_inference_transforms(image_size)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict(
        self,
        source: str | Path | np.ndarray,
    ) -> dict[str, Any]:
        """Run detection on a single image.

        Parameters
        ----------
        source : str | Path | np.ndarray
            Either a file path or a BGR / RGB numpy array (H, W, 3).

        Returns
        -------
        dict with keys:
            - boxes  : np.ndarray, shape (K, 4) — [xmin, ymin, xmax, ymax]
            - scores : np.ndarray, shape (K,)
            - labels : np.ndarray, shape (K,) (0-based class ids, background removed)
        """
        image = self._load_image(source)  # RGB uint8 (H, W, 3)
        orig_h, orig_w = image.shape[:2]
        tensor = self._preprocess(image)  # (C, H, W) tensor

        with torch.no_grad():
            outputs: list[dict[str, Tensor]] = self.model([tensor])

        result = outputs[0]
        boxes = result["boxes"]  # (N, 4)
        scores = result["scores"]  # (N,)
        labels = result["labels"]  # (N,)

        # Confidence filter
        keep = scores >= self.conf_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # NMS
        if boxes.numel() > 0:
            nms_keep = nms(boxes, scores, self.nms_threshold)
            boxes = boxes[nms_keep]
            scores = scores[nms_keep]
            labels = labels[nms_keep]

        # Scale boxes back to original image size
        boxes = self._rescale_boxes(boxes, self.image_size, orig_h, orig_w)

        # Labels: Faster R-CNN uses 1-based indexing (1=first class) → shift to 0-based
        labels = labels - 1

        return {
            "boxes": boxes.cpu().numpy(),
            "scores": scores.cpu().numpy(),
            "labels": labels.cpu().numpy(),
        }

    def predict_batch(
        self,
        sources: list[str | Path | np.ndarray],
    ) -> list[dict[str, Any]]:
        """Run detection on a batch of images."""
        return [self.predict(s) for s in sources]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_image(source: str | Path | np.ndarray) -> np.ndarray:
        """Read an image and return as RGB uint8 numpy array."""
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isinstance(source, np.ndarray):
            # Assume already RGB (or grayscale)
            return source
        raise TypeError(f"Unsupported source type: {type(source)}")

    def _preprocess(self, image: np.ndarray) -> Tensor:
        """Apply transforms and return a (C, H, W) tensor on device."""
        # Image is already RGB from _load_image
        # Albumentations expects RGB for inference transforms
        transformed = self.transform(image=image)
        tensor = transformed["image"].to(self.device)

        # Faster R-CNN expects float in range [0, 1]
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0

        return tensor

    def _load_weights(self, path: str | Path) -> None:
        """Load model weights from checkpoint file."""
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            # Try different keys
            if "model" in ckpt:
                state = ckpt["model"]
            elif "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            else:
                state = ckpt
        else:
            state = ckpt

        self.model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint from {path}")
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            print(f"  Epoch: {ckpt['epoch']}")
        if isinstance(ckpt, dict) and "mAP" in ckpt:
            print(f"  mAP: {ckpt['mAP']:.4f}")

    @staticmethod
    def _rescale_boxes(
        boxes: Tensor,
        model_size: int,
        orig_h: int,
        orig_w: int,
    ) -> Tensor:
        """Rescale boxes from padded model input space back to original image.

        The preprocessing uses LongestMaxSize + PadIfNeeded, so we reverse that:
          1. Compute the scale factor used by LongestMaxSize.
          2. Compute the padding offsets.
          3. Subtract padding, then divide by scale.
        """
        scale = model_size / max(orig_h, orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        pad_top = (model_size - new_h) // 2
        pad_left = (model_size - new_w) // 2

        boxes = boxes.clone().float()
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        boxes /= scale

        # Clamp to image boundaries
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)

        return boxes
