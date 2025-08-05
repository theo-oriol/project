
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.transforms.functional import pad
import os
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


class Classifier(nn.Module):
    def __init__(self, output_dim=1):
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
        feat = nn.functional.normalize(x, dim=1, eps=1e-8).detach()
        logits = self.classifier(x)
        return logits, feat

    def get_patch_tokens(self, x):
        tokens = self.backbone.get_intermediate_layers(x, n=1)[0]  
        return tokens[:, 1:, :]  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier()
checkpoint = torch.load("/home/oriol@newcefe.newage.fr/JeanZay/results_whole/results/dino_10_000_5_0/model", map_location="cpu")
model.load_state_dict(checkpoint)
model.to(device)


class NewPad:
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        pad_w = (max_wh - w) // 2
        pad_h = (max_wh - h) // 2
        padding = (pad_w, pad_h, max_wh - w - pad_w, max_wh - h - pad_h)
        return pad(img, padding)

transform = T.Compose([
    NewPad(),
    T.Resize(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

input_dir = "/home/oriol@newcefe.newage.fr/Datasets/whole_bird"

image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

path = image_paths[6]  
img_pil = Image.open(path).convert("RGB")

with torch.no_grad():
    x = transform(img_pil).unsqueeze(0).to(device) 
    patch_tokens = model.get_patch_tokens(x) 
    patch_tokens = patch_tokens.squeeze(0)        


def patch_pca_rgb(patch_tokens, h, w, threshold_percentile=70):
    pca = PCA(n_components=3)
    components = pca.fit_transform(patch_tokens.cpu().numpy())  # [N, 3]

    # Threshold first PCA component
    first_component = components[:, 0]
    threshold = np.percentile(first_component, threshold_percentile)
    foreground_mask = first_component < np.percentile(first_component, 30)
    # Normalize all components to [0, 1]
    components = (components - components.min(0)) / (components.ptp(0) + 1e-5)

    # Reshape to grid
    rgb = components.reshape(h, w, 3)
    mask = foreground_mask.reshape(h, w)

    # Zero out background
    rgb[~mask] = 0  # or set to gray if preferred

    return rgb

pca_rgb = patch_pca_rgb(patch_tokens, h=15, w=17)


p = path.split("/")[-1].split(".")[0]  # Extract filename without extension

img = Image.fromarray((pca_rgb * 255).astype(np.uint8)).resize((224, 224), resample=Image.NEAREST)

original = Image.open(path).convert("RGB")
original = NewPad()(original).resize((224, 224))
overlay = Image.blend(original, img, alpha=0.5)
overlay.save(f"observation/pca_output/{p}_pca.png")