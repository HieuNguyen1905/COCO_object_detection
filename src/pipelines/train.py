import torch
from src.utils.config import load_config
from src.datasets.coco_dataset import COCODataset
from src.datasets.datamodule import DataModule
from src.models.build import build_model
from src.training.optimizer import build_optimizer
from src.training.trainer import Trainer

def main():
    cfg = load_config("configs/train.yaml")

    train_ds = COCODataset(...)
    val_ds = COCODataset(...)

    dm = DataModule(train_ds, val_ds, batch_size=4, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

    model = build_model(cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = build_optimizer(model, cfg["optimizer"])

    trainer = Trainer(model, optimizer, dm.train_loader(), device)
    trainer.fit(cfg["epochs"])

if __name__ == "__main__":
    main()