from __future__ import annotations
import logging
import os
import random
from typing import Any
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class COCODetectionDataset(Dataset):

    def __init__(self, img_dir, ann_file, transforms=None,):
        super().__init__()
        self.img_dir = img_dir
        self.transforms = transforms

        # Load COCO API
        coco = COCO(ann_file)

        # original_id -> 0-based index
        cat_ids = sorted(coco.getCatIds())
        self.cat_id_to_label: dict[int, int] = {
            cat_id: idx for idx, cat_id in enumerate(cat_ids)
        }
        self.num_classes = len(cat_ids)

       # Pre-extract image metadata and annotations to avoid COCO API overhead in __getitem__
        self._img_ids: list[int] = []
        self._file_names: list[str] = []
        self._img_sizes: list[tuple[int, int]] = []  # (width, height)
        self._annotations: list[list[dict]] = []      # per-image list of anns

        for img_id in sorted(coco.getImgIds()):
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False) # training chỉ cần ann không crowd (đông đúc, chồng nhau)
            anns = coco.loadAnns(ann_ids) # list of dicts with keys: 'bbox', 'category_id', 'area', 'iscrowd', ...
            if len(anns) == 0:
                continue  # skip images without annotations

            img_info = coco.loadImgs(img_id)[0] # dict with keys: 'file_name', 'width', 'height', ...
            self._img_ids.append(img_id)
            self._file_names.append(img_info["file_name"])
            self._img_sizes.append((img_info["width"], img_info["height"]))
            # Keep only the fields we need (bbox, category_id, area, iscrowd)
            self._annotations.append([{"bbox": ann["bbox"], "category_id": ann["category_id"], "area": ann["area"],
                                        "iscrowd": ann.get("iscrowd", 0),} for ann in anns])
        del coco

    def __len__(self) -> int:
        return len(self._img_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        # Retry with a random index if the image is missing or corrupted
        for _attempt in range(10):
            img_id = self._img_ids[index]
            file_name = self._file_names[index]
            img_w, img_h = self._img_sizes[index]
            img_path = os.path.join(self.img_dir, file_name)

            # Read image as RGB numpy array (H, W, C)
            image = cv2.imread(img_path)
            if image is not None:
                break
            logger.warning("Skipping missing/corrupted image: %s", img_path)
            index = random.randint(0, len(self) - 1)
        else:
            raise FileNotFoundError(
                f"Could not load any image after 10 retries (last: {img_path})"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse pre-extracted annotations
        anns = self._annotations[index]

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd: list[int] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO format: [x, y, width, height]
            if w <= 0 or h <= 0:
                continue
            # Convert to pascal_voc: [xmin, ymin, xmax, ymax]
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h

            # Clamp to image boundaries
            xmin = max(0.0, xmin)
            ymin = max(0.0, ymin)
            xmax = min(float(img_w), xmax)
            ymax = min(float(img_h), ymax)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.cat_id_to_label[ann["category_id"]])
            areas.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

        # Apply albumentations transforms
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels,
            )
            image = transformed["image"]  # already a tensor (C, H, W)
            boxes = transformed["bboxes"]
            labels = transformed["labels"]
            # ToTensorV2 outputs uint8, need float [0, 1] for Faster R-CNN
            image = image.float() / 255.0
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Build target dict
        target: dict[str, Any] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": img_id,
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.bool),
        }
        # return (image, labels, boxes)
        return image, target
    
def collate_fn( batch: list[tuple[torch.Tensor, dict[str, Any]]],) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    """Custom collate function for detection.

    Returns images as a list of tensors (variable-size targets prevent stacking)
    and targets as a list of dicts.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)
