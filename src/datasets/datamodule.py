from __future__ import annotations
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from src.datasets.dataset import COCODetectionDataset, collate_fn
from src.datasets.transforms import get_train_transforms, get_val_transforms

class DataModule:

    def __init__(self, config_path: str | Path):
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        data_config = config["data"]
        loader_config = config["dataloader"]
        image_config = config["image"]

        image_size = image_config["image_size"]
        mean = image_config["mean"]
        std = image_config["std"]

        # Build augmentation pipelines
        train_transforms = get_train_transforms(image_size, mean, std)
        val_transforms = get_val_transforms(image_size, mean, std)

        # Build datasets
        self.train_dataset = COCODetectionDataset(
            img_dir=data_config["train_images"],
            ann_file=data_config["train_json"],
            transforms=train_transforms,
        )
        self.val_dataset = COCODetectionDataset(
            img_dir=data_config["val_images"],
            ann_file=data_config["val_json"],
            transforms=val_transforms,
        )

        # Build dataloaders
        num_workers = loader_config["num_workers"]
        # Only pin memory when a CUDA device is actually available
        pin_memory = loader_config["pin_memory"] and torch.cuda.is_available()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=loader_config["batch_size"],
            shuffle=loader_config["shuffle_train"],
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=loader_config["batch_size"],
            shuffle=loader_config["shuffle_val"],
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )

    def num_classes(self) -> int:
        return self.train_dataset.num_classes
