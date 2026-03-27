import os

import torch
from tqdm import tqdm

from src.utils import evaluate_map


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        device,
        num_classes,
        scheduler=None,
        val_loader=None,
        save_dir=None,
        log_interval=10,
        iou_threshold=0.5,
        ap_interpolation_points=11,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.iou_threshold = iou_threshold
        self.ap_interpolation_points = ap_interpolation_points
        self.best_map = 0.0

        self.model.to(self.device)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        for step, (images, targets) in pbar:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if step % self.log_interval == 0:
                pbar.set_description(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None, None

        print("\nValidating...")

        # Tính validation loss - cần model ở train mode để lấy loss_dict
        self.model.train()
        total_loss = 0

        for images, targets in tqdm(self.val_loader, desc="Val Loss"):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())

            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)

        # Calculate mAP - cần model ở eval mode để lấy predictions
        self.model.eval()
        mAP = evaluate_map(
            self.model,
            self.val_loader,
            self.device,
            self.num_classes,
            iou_threshold=self.iou_threshold,
            num_points=self.ap_interpolation_points,
        )

        return avg_loss, mAP

    def save_checkpoint(self, epoch, mAP, is_best=False):
        if self.save_dir is None:
            return

        os.makedirs(self.save_dir, exist_ok=True)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "mAP": mAP,
        }

        # Always save last checkpoint
        last_path = os.path.join(self.save_dir, "last.pth")
        torch.save(checkpoint, last_path)

        # Save best checkpoint if mAP improved
        if is_best:
            best_path = os.path.join(self.save_dir, "best.pth")
            torch.save(checkpoint, best_path)
            print(f"Best model saved with mAP: {mAP:.4f}")

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, mAP = self.validate()

            if self.scheduler:
                self.scheduler.step()

            print(f"\nEpoch {epoch}:")
            print(f"Train Loss: {train_loss:.4f}")

            if val_loss is not None:
                print(f"Val Loss: {val_loss:.4f}")
                print(f"mAP: {mAP:.4f}")

                # Check if this is the best model
                is_best = mAP > self.best_map
                if is_best:
                    self.best_map = mAP

                self.save_checkpoint(epoch, mAP, is_best)
            else:
                self.save_checkpoint(epoch, 0.0, False)