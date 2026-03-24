"""Predictor: load a trained model and run inference on images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torchvision.ops import nms

from src.models.model import ObjectDetector


class Predictor:
    """End-to-end inference pipeline for object detection.

    1. Load model weights from a ``.pth`` checkpoint.
    2. Preprocess an input image (resize, normalize, to tensor).
    3. Run the model forward pass.
    4. Apply confidence thresholding + NMS.
    5. Return filtered detections.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the saved ``.pth`` file (must contain ``model_state_dict``
        or be a raw ``state_dict``).
    num_classes : int
        Number of foreground classes the model was trained on.
    device : str | torch.device
        Target device for inference.
    image_size : int
        Input resolution the model expects.
    mean / std : list[float]
        ImageNet normalisation constants.
    conf_threshold : float
        Minimum confidence score to keep a detection.
    nms_threshold : float
        IoU threshold for Non-Maximum Suppression.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        num_classes: int = 80,
        device: str | torch.device = "cuda",
        image_size: int = 640,
        mean: list[float] = (0.485, 0.456, 0.406),
        std: list[float] = (0.229, 0.224, 0.225),
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.image_size = image_size
        self.mean = list(mean)
        self.std = list(std)

        # Build model and load weights
        self.model = ObjectDetector(num_classes=num_classes, pretrained_backbone=False)
        self._load_weights(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Inference transform (no augmentations, no bbox_params needed)
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=114,
            ),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

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
            Either a file path or a BGR / RGB numpy array ``(H, W, 3)``.

        Returns
        -------
        dict with keys:
            - ``boxes``  : np.ndarray, shape ``(K, 4)`` — ``[xmin, ymin, xmax, ymax]``
            - ``scores`` : np.ndarray, shape ``(K,)``
            - ``labels`` : np.ndarray, shape ``(K,)``  (0-based class ids)
        """
        image = self._load_image(source)            # RGB uint8 (H, W, 3)
        orig_h, orig_w = image.shape[:2]
        tensor = self._preprocess(image)             # (1, C, H, W)

        with torch.no_grad():
            outputs: list[dict[str, Tensor]] = self.model([tensor.squeeze(0)])

        result = outputs[0]
        boxes = result["boxes"]     # (N, 4)
        scores = result["scores"]   # (N,)
        labels = result["labels"]   # (N,)

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

        # Labels: Faster R-CNN adds +1 for background → shift back to 0-based
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
            return source
        raise TypeError(f"Unsupported source type: {type(source)}")

    def _preprocess(self, image: np.ndarray) -> Tensor:
        """Apply transforms and return a ``(1, C, H, W)`` tensor on device."""
        transformed = self.transform(image=image)
        tensor = transformed["image"].unsqueeze(0).to(self.device)
        return tensor

    def _load_weights(self, path: str | Path) -> None:
        """Load model weights from a checkpoint file."""
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        self.model.load_state_dict(state, strict=True)

    @staticmethod
    def _rescale_boxes(
        boxes: Tensor,
        model_size: int,
        orig_h: int,
        orig_w: int,
    ) -> Tensor:
        """Rescale boxes from padded model input space back to original image.

        The preprocessing uses ``LongestMaxSize`` + ``PadIfNeeded``, so we
        reverse that mapping:
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
