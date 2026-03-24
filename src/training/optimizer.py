import torch

def build_optimizer(model, cfg):
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=0.9
    )