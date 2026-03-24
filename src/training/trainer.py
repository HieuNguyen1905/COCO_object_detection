import torch

class Trainer:
    def __init__(self, model, optimizer, train_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device

    def train_one_epoch(self):
        self.model.train()

        for images, targets in self.train_loader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k,v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit(self, epochs):
        for epoch in range(epochs):
            self.train_one_epoch()
            print(f"Epoch {epoch} done")