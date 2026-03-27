import torch


def build_optimizer(model, config):

    optimizer_type = config["OPTIMIZER"].lower()
    lr = config["LEARNING_RATE"]

    if optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config["MOMENTUM"],
        )
    elif optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")