import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        device,
        scheduler=None,
        val_loader=None,
        save_dir=None,
        log_interval=10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.log_interval = log_interval

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
            return None

        self.model.eval()
        total_loss = 0

        for images, targets in self.val_loader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())

            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch):
        if self.save_dir is None:
            return

        path = f"{self.save_dir}/checkpoint_epoch_{epoch}.pth"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            path,
        )

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()

            if self.scheduler:
                self.scheduler.step()

            print(f"\nEpoch {epoch}:")
            print(f"Train Loss: {train_loss:.4f}")

            if val_loss is not None:
                print(f"Val Loss: {val_loss:.4f}")

            self.save_checkpoint(epoch)