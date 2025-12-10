import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T

class PerceptualExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]  
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.model = vgg.to(device)

        self.tf = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def get_feat(self, img_gray):
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        t = self.tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(t)
        return feat.squeeze().cpu().numpy()

def perceptual_distance(f1, f2):
    return np.linalg.norm(f1 - f2)
