from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


def build_scheduler(optimizer, config):

    scheduler_type = config["SCHEDULER"].lower()

    if scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=config["STEP_SIZE"],
            gamma=config["GAMMA"],
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config["NUM_EPOCHS"],
        )
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")