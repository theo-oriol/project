import torch 
import torch.nn as nn
import os 

os.environ["XFORMERS_AVAILABLE"] = "False"



class dinov2_vitl14_reg(nn.Module):
    def __init__(self,output_dim=1):
        super().__init__()
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        self.backbone = dino

        for param in self.backbone.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024//3),
            nn.ReLU(inplace=True),
            nn.Linear(1024//3, output_dim)  
        )

    def forward(self, x):
        x = self.backbone(x)
        feat = nn.functional.normalize(x, dim=1,eps=1e-8).detach()
        logits = self.classifier(x) 
        return logits, feat


class dinov2_vitl14(nn.Module):
    def __init__(self,output_dim=1):
        super().__init__()
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.backbone = dino

        for param in self.backbone.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024//3),
            nn.ReLU(inplace=True),
            nn.Linear(1024//3, output_dim)  
        )

    def forward(self, x):
        x = self.backbone(x)
        feat = nn.functional.normalize(x, dim=1,eps=1e-8).detach()
        logits = self.classifier(x) 
        return logits, feat