import torch.nn as nn

def get_loss_function(name):
    name = name.lower()
    if name == "bcewithlogitsloss":
        return nn.BCEWithLogitsLoss(reduction='mean')
    elif name == "crossentropyloss":
        return nn.CrossEntropyLoss()
    elif name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")