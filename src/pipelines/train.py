import torch
from src.utils.config import load_config
from src.datasets.datamodule import DataModule
from src.models.bbox_regressor import build_model
from src.training.trainer import Trainer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from src.datasets.dataset import COCODetectionDataset
from src.datasets.dataset import collate_fn
def main():
    config = load_config("configs/configs.yaml")
    
    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=config["IMAGE_SIZE"]),
            A.PadIfNeeded(
                min_height=config["IMAGE_SIZE"],
                min_width=config["IMAGE_SIZE"],
                border_mode=0,
                fill=114,
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            ToTensorV2(),  # Faster R-CNN tự normalize, chỉ cần convert to tensor
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=1.0,
            min_visibility=0.2,
        ),
    )
    val_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=config["IMAGE_SIZE"]),
            A.PadIfNeeded(
                min_height=config["IMAGE_SIZE"],
                min_width=config["IMAGE_SIZE"],
                border_mode=0,
                fill=114,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=1.0,
            min_visibility=0.2,
        ),
    )
    train_ds = COCODetectionDataset(config["TRAIN_IMAGES"], config["TRAIN_JSON"], transforms=train_transforms)
    val_ds = COCODetectionDataset(config["VAL_IMAGES"], config["VAL_JSON"], transforms=val_transforms)

    dm = DataModule(train_ds, val_ds, batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"], pin_memory=config["PIN_MEMORY"], collate_fn=collate_fn)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()


    # num_classes + 1 for background
    model = build_model(num_classes=dm.num_classes + 1, pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, optimizer, train_loader, device, val_loader=val_loader, save_dir=config["BASE_OUTPUT"])
    trainer.fit(config["NUM_EPOCHS"])


if __name__ == "__main__":
    main()