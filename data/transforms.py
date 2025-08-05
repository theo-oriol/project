import numpy as np 
from torchvision.transforms.functional import pad
from torchvision import transforms

def get_padding(image):    
    """
    Add padding to an image to make it a square 
    """
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        return pad(img, get_padding(img), self.fill, self.padding_mode)


class ApplyTransform:
    """
    resize, dataaugmentation and normalisation 
    """
    def __init__(self,model,img_size=224):
        self.resize = transforms.Compose([
            NewPad(),
            transforms.Resize(img_size),
        ])

        self.augment = transforms.Compose([       
            transforms.RandomApply([ transforms.RandomResizedCrop(img_size, scale=(0.6, 1.2), ratio=(0.75, 1.33))],p=0.1),
            transforms.RandomApply([ transforms.RandomHorizontalFlip()],p=0.1),
            transforms.RandomApply([ transforms.RandomVerticalFlip()],p=0.1),
            transforms.RandomApply([ transforms.RandomRotation(degrees=5)],p=0.1),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                ),
            ], p=0.1),
            transforms.RandomApply([ transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],p=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.1),
            transforms.RandomApply([
                transforms.RandomPosterize(bits=4)
            ], p=0.1),         
            transforms.ToTensor(),
        ])
            
        if model in ['dinov2_vitl14_reg', 'dinov2_vitl14']: 
            self.normalise = transforms.Compose([
                lambda x: 255.0 * x[:3],
                transforms.Normalize(
                        mean=(123.675, 116.28, 103.53),
                        std=(58.395, 57.12, 57.375),
                ),
            ])
        elif model == 'CNN' : 
            self.normalise = transforms.Compose([
                transforms.Normalize(
                        mean=([0.5]*3),
                        std=([0.5]*3),
                ),
            ])
        elif model == 'inceptionv4':
            self.normalise = transforms.Compose([
                transforms.Normalize(
                        mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5),
                ),
            ])
        else : raise ValueError(f"Unknown model : {model}")

