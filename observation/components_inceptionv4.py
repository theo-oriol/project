import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.transforms.functional import pad
from PIL import Image
from sklearn.decomposition import PCA
import timm

# -------- Model Definition --------

class InceptionV4Classifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        base_model = timm.create_model('inception_v4', pretrained=True)
        
        self.backbone = base_model
        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)

        for param in self.backbone.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 3),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 3, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        feat = nn.functional.normalize(x, dim=1, eps=1e-8).detach()
        logits = self.classifier(x)
        return logits, feat

    def get_patch_tokens(self, x):
        features = self.backbone.forward_features(x)  # [B, C, H, W]
        B, C, H, W = features.shape
        tokens = features.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, D]
        return tokens, H, W

# -------- Preprocessing --------

class NewPad:
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        pad_w = (max_wh - w) // 2
        pad_h = (max_wh - h) // 2
        padding = (pad_w, pad_h, max_wh - w - pad_w, max_wh - h - pad_h)
        return pad(img, padding, fill=0)

transform = T.Compose([
    NewPad(),
    T.Resize(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# -------- PCA Visualization Function --------

def patch_pca_rgb(patch_tokens, h, w, threshold_percentile=30, keep_low=True):
    pca = PCA(n_components=3)
    components = pca.fit_transform(patch_tokens.cpu().numpy())  # [N, 3]

    # Use first component to remove background
    first_component = components[:, 0]
    threshold = np.percentile(first_component, threshold_percentile)

    foreground_mask = first_component < threshold if keep_low else first_component > threshold

    components = (components - components.min(0)) / (components.ptp(0) + 1e-5)
    rgb = components.reshape(h, w, 3)
    mask = foreground_mask.reshape(h, w)

    rgb[~mask] = 0  # Set background to black
    return rgb

# -------- Load Model --------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionV4Classifier()
checkpoint = torch.load("/home/oriol@newcefe.newage.fr/JeanZay/results_total/results/inception_10_000_5_3/model", map_location="cpu")
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# -------- Load Image --------

input_dir = "/home/oriol@newcefe.newage.fr/Datasets/whole_bird"
image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
img_path = image_paths[0]  # change index to pick another image
img_pil = Image.open(img_path).convert("RGB")

# -------- Run PCA Visualization --------

with torch.no_grad():
    x = transform(img_pil).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    patch_tokens, h, w = model.get_patch_tokens(x)  # [1, N, D], h × w = N
    patch_tokens = patch_tokens.squeeze(0)          # [N, D]

pca_rgb = patch_pca_rgb(patch_tokens, h, w, threshold_percentile=30, keep_low=True)

# -------- Create Overlay --------

p = os.path.splitext(os.path.basename(img_path))[0]
os.makedirs("observation/pca_output_incept", exist_ok=True)

# Resize PCA result back to image size
pca_img = Image.fromarray((pca_rgb * 255).astype(np.uint8)).resize((224, 224), resample=Image.NEAREST)
original = NewPad()(img_pil).resize((224, 224))

overlay = Image.blend(original, pca_img, alpha=0.5)
overlay.save(f"observation/pca_output_incept/{p}_pca.png")
print(f"✅ Saved PCA visualization to observation/pca_output/{p}_pca.png")
