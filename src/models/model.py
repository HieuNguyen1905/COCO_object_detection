"""Faster R-CNN based Object Detector with configurable backbone and num_classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn, Tensor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class ObjectDetector(nn.Module):
    """Wrapper around torchvision Faster R-CNN.

    * **Training** : ``forward(images, targets)`` returns a ``dict`` of losses
      (``loss_classifier``, ``loss_box_reg``, ``loss_objectness``,
      ``loss_rpn_box_reg``).
    * **Inference**: ``forward(images)`` returns a ``list[dict]`` where each
      dict contains ``boxes``, ``scores``, and ``labels``.

    Parameters
    ----------
    num_classes : int
        Number of foreground classes (background is added automatically by
        Faster R-CNN, so pass the raw count, e.g. 80 for COCO).
    backbone_name : str
        Currently only ``"resnet50"`` is supported (matches
        ``fasterrcnn_resnet50_fpn``).
    pretrained_backbone : bool
        Whether to initialise the backbone with ImageNet-pretrained weights.
    pretrained_weights : str | None
        Path to a full-model checkpoint for fine-tuning.  Loaded **after**
        the classification head is replaced.
    freeze_bn : bool
        If ``True``, freezes all BatchNorm layers in the backbone.
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        pretrained_weights: str | None = None,
        freeze_bn: bool = False,
    ) -> None:
        super().__init__() # Bắt buộc phải có để nhận diên neural network

        # --- build base model ---------------------------------------------------
        if backbone_name == "resnet50":
            weights = (
                FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                if pretrained_backbone
                else None
            )
            self.model = fasterrcnn_resnet50_fpn(weights=weights)
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "
                "Currently only 'resnet50' is implemented."
            )

        # --- replace the classification head with the correct num_classes --------
        # Faster R-CNN expects num_classes = foreground + 1 (background)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes + 1
        )

        # --- optionally freeze BatchNorm layers ---------------------------------
        if freeze_bn:
            self._freeze_bn()

        # --- optionally load full-model weights for fine-tuning ------------------
        if pretrained_weights is not None:
            state = torch.load(pretrained_weights, map_location="cpu")
            # Support checkpoints saved as {"model_state_dict": ...}
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            self.load_state_dict(state, strict=False)

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def forward(
        self,
        images: list[Tensor],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Forward pass.

        Parameters
        ----------
        images : list[Tensor]
            List of ``(C, H, W)`` tensors (un-batched).
        targets : list[dict], optional
            Required during training.  Each dict must contain ``boxes``
            ``(FloatTensor[N, 4])`` and ``labels`` ``(Int64Tensor[N])``.

        Returns
        -------
        dict[str, Tensor]
            Loss dict when ``self.traini    ng is True``.
        list[dict[str, Tensor]]
            Predictions (``boxes``, ``scores``, ``labels``) otherwise.
        """
        return self.model(images, targets)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _freeze_bn(self) -> None:
        """Set all BatchNorm layers to eval mode so running stats stay fixed."""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    @classmethod
    def from_config(cls, config_path: str | Path) -> "ObjectDetector":
        """Instantiate from a YAML config file (``configs/model.yaml``).

        Example YAML::

            backbone:
              name: resnet50
              pretrained: true
              freeze_bn: false
            head:
              num_classes: 80
            weights:
              pretrained_weights: null
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        backbone_cfg = cfg["backbone"]
        head_cfg = cfg["head"]
        weights_cfg = cfg.get("weights", {})

        return cls(
            num_classes=head_cfg["num_classes"],
            backbone_name=backbone_cfg["name"],
            pretrained_backbone=backbone_cfg["pretrained"],
            pretrained_weights=weights_cfg.get("pretrained_weights"),
            freeze_bn=backbone_cfg.get("freeze_bn", False),
        )
