"""Detection metrics using torchmetrics MeanAveragePrecision."""

from __future__ import annotations

from typing import Any

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class DetectionMetrics:
    """Wraps ``torchmetrics.detection.MeanAveragePrecision`` for convenience.

    Usage::

        metrics = DetectionMetrics(iou_thresholds=[0.5, 0.75])
        for images, targets in val_loader:
            preds = model(images)
            metrics.update(preds, targets)
        results = metrics.compute()
        print(results)     # {'map': ..., 'map_50': ..., 'map_75': ..., ...}
        metrics.reset()
    """

    def __init__(
        self,
        iou_type: str = "bbox",
        iou_thresholds: list[float] | None = None,
    ) -> None:
        self.metric = MeanAveragePrecision(
            iou_type=iou_type,
            iou_thresholds=iou_thresholds,
        )

    def update(
        self,
        preds: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> None:
        """Accumulate a batch of predictions and ground truths.

        Parameters
        ----------
        preds : list[dict]
            Each dict must contain:
              - ``boxes``  : ``FloatTensor[N, 4]`` (xyxy)
              - ``scores`` : ``FloatTensor[N]``
              - ``labels`` : ``Int64Tensor[N]``
        targets : list[dict]
            Each dict must contain:
              - ``boxes``  : ``FloatTensor[M, 4]`` (xyxy)
              - ``labels`` : ``Int64Tensor[M]``
        """
        self.metric.update(preds, targets)

    def compute(self) -> dict[str, Any]:
        """Compute all mAP variants and return as a plain dict.

        Keys include ``map``, ``map_50``, ``map_75``, ``map_small``,
        ``map_medium``, ``map_large``, ``mar_1``, ``mar_10``, ``mar_100``,
        among others.
        """
        raw = self.metric.compute()
        # Convert single-element tensors to Python floats for readability
        return {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v
                for k, v in raw.items()}

    def reset(self) -> None:
        """Clear internal state for a new evaluation epoch."""
        self.metric.reset()

    def to(self, device: torch.device | str) -> "DetectionMetrics":
        """Move internal metric state to *device*."""
        self.metric = self.metric.to(device)
        return self
