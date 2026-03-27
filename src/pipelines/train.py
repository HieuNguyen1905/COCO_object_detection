import torch

from src.datasets import COCODetectionDataset, DataModule, collate_fn
from src.models import build_model
from src.training import Trainer, build_optimizer
from src.utils import load_config, get_train_transforms, get_val_transforms


def main():
    config = load_config("configs/configs.yaml")

    train_transforms = get_train_transforms(config["IMAGE_SIZE"])
    val_transforms = get_val_transforms(config["IMAGE_SIZE"])

    train_ds = COCODetectionDataset(config["TRAIN_IMAGES"], config["TRAIN_JSON"], transforms=train_transforms)
    val_ds = COCODetectionDataset(config["VAL_IMAGES"], config["VAL_JSON"], transforms=val_transforms)

    dm = DataModule(train_ds, val_ds, batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"], pin_memory=config["PIN_MEMORY"], collate_fn=collate_fn)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # num_classes from config or dataset + 1 for background
    num_classes = config.get("NUM_CLASSES", dm.num_classes + 1)
    base_model = config.get("BASE_MODEL", "fasterrcnn_resnet50_fpn")
    model = build_model(num_classes=num_classes, pretrained=config.get("PRETRAINED", True), base_model=base_model)

    # Device handling
    device_config = config.get("DEVICE", "auto")
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)

    model.to(device)

    optimizer = build_optimizer(model, config)

    trainer = Trainer(
        model,
        optimizer,
        train_loader,
        device,
        num_classes=num_classes,
        val_loader=val_loader,
        save_dir=config["BASE_OUTPUT"],
        log_interval=config.get("LOG_INTERVAL", 10),
        iou_threshold=config.get("IOU_THRESHOLD", 0.5),
        ap_interpolation_points=config.get("AP_INTERPOLATION_POINTS", 11),
    )
    trainer.fit(config["NUM_EPOCHS"])


if __name__ == "__main__":
    main()