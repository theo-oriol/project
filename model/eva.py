import timm
import torch.nn as nn

class eva02L(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        base_model= timm.create_model('eva02_large_patch14_448.mim_in22k_ft_in22k', pretrained=True)
        
        self.backbone = base_model

        # Replace classifier with identity (to get features only)
        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)

        for param in self.backbone.parameters():
            param.requires_grad = True


        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features//3),
            nn.ReLU(inplace=True),
            nn.Linear(in_features//3, num_classes)  
        )
        
    def forward(self, x):
        x = self.backbone(x)
        feat = nn.functional.normalize(x, dim=1,eps=1e-8).detach()
        logits = self.classifier(x) 
        return logits, feat
        