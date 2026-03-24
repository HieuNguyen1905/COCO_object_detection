from torch.optim.lr_scheduler import StepLR

def build_scheduler(optimizer):
    return StepLR(optimizer, step_size=3, gamma=0.1)