from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.optim as optim


def get_optimizer(parameters,model):
    if parameters["opt"] == "adam":
        return optim.Adam(model.parameters(), lr=parameters["lr"], weight_decay=parameters["weight_decay"])

def get_scheduler(parameters,optimizer):
    warmup_epochs = parameters["warmup_epochs"]
    main_epochs = parameters["epochs"] - warmup_epochs
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=main_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return scheduler