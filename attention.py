import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms.functional import pad
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(1, '/home/oriol@newcefe.newage.fr/Models/dinov2')
from dinov2.models.vision_transformer import vit_large


class Classifier(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()
        dino = vit_large(
            patch_size=14,
            img_size=526,
            init_values=1.0,
            block_chunks=0
        )
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
    

patch_size = 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dir = "/home/oriol@newcefe.newage.fr/Datasets/whole_bird"
output_dir = "./attention_outputs"
os.makedirs(output_dir, exist_ok=True)

model = Classifier()
checkpoint = torch.load("/home/oriol@newcefe.newage.fr/Models/project/results/grad_3_2/model", map_location="cpu")
model.load_state_dict(checkpoint)
model = model.backbone
model.eval().to(device)
for p in model.parameters():
    p.requires_grad = False

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

image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])


mean_attn_per_head = None       # [heads, 224, 224]
mean_of_means = None            # [224, 224]
count = 0
last_img_np = None

for img_path in tqdm(image_paths, desc="Processing images"):
    try:
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        w, h = img_tensor.shape[2] - img_tensor.shape[2] % patch_size, img_tensor.shape[3] - img_tensor.shape[3] % patch_size
        img_tensor = img_tensor[:, :, :w, :h]

        w_featmap = img_tensor.shape[-2] // patch_size
        h_featmap = img_tensor.shape[-1] // patch_size

        attn = model.get_last_self_attention(img_tensor)  # [1, heads, tokens, tokens]
        nh = attn.shape[1]

        # Extract CLS-to-patch attention
        attn = attn[0, :, 0, 1:].reshape(nh, -1)
        attn[:, 13] = 0  # optional: suppress outlier pixel
        attn[:, 2] = 0
        # print(torch.argmax(attn[0])) 
        attn = attn.reshape(nh, w_featmap, h_featmap)  # [heads, 16, 16]

        # Upsample to [heads, 224, 224]
        attn_upsampled = F.interpolate(attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

        # Per-image mean over heads: [224, 224]
        per_image_mean = np.mean(attn_upsampled, axis=0)

        # Accumulate
        if mean_attn_per_head is None:
            mean_attn_per_head = attn_upsampled.copy()
            mean_of_means = per_image_mean.copy()
        else:
            mean_attn_per_head += attn_upsampled
            mean_of_means += per_image_mean

        count += 1
        last_img_np = np.array(NewPad()(img_pil).resize((224, 224)))

    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")

# ---- Average Across All Images ----
if count > 0:
    mean_attn_per_head /= count        # shape [heads, 224, 224]
    mean_of_means /= count             # shape [224, 224]

    # Save NumPy arrays
    np.save(os.path.join(output_dir, "mean_attention_per_head.npy"), mean_attn_per_head)
    np.save(os.path.join(output_dir, "mean_of_means_attention.npy"), mean_of_means)

    # ---- Save Overlays ----
    # 1. Mean of heads
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(last_img_np)
    ax.imshow(mean_of_means, cmap="inferno", alpha=0.5)
    ax.axis("off")
    fig.savefig(os.path.join(output_dir, "global-mean-of-means-overlay.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # 2. Each head
    for j in range(mean_attn_per_head.shape[0]):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        ax.imshow(last_img_np)
        ax.imshow(mean_attn_per_head[j], cmap="inferno", alpha=0.5)
        ax.axis("off")
        fname = os.path.join(output_dir, f"global-mean-head{j}-overlay.png")
        fig.savefig(fname, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"✅ Saved {mean_attn_per_head.shape[0]} per-head mean overlays and mean-of-means overlay.")
else:
    print("❌ No images processed.")