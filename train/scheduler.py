from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.optim as optim


def get_optimizer(opt,lr,weight_decay,model):
    if opt == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(warmup_epochs,epochs,optimizer):
    main_epochs = epochs - warmup_epochs
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=main_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return scheduler