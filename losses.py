# losses.py
import torch
import torch.nn as nn
import torchvision.models as models

class ContentLoss(nn.Module):
    def __init__(self, weight_path, layer="conv5_4"):
        super().__init__()
        from torchvision import models
        vgg = models.vgg19()
        vgg.load_state_dict(torch.load(weight_path))
        self.features = nn.Sequential(*list(vgg.features.children())[:36]).eval()
        self.features = self.features.cuda()
        for p in self.features.parameters():
            p.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        # [B, 1, H, W] => [B, 3, H, W] for VGG
        if sr.shape[1] == 1:
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)
        # Now pass to VGG
        with torch.cuda.amp.autocast(enabled=False):  # Or torch.amp.autocast(enabled=False)
            sr_features = self.features(sr.float())
            hr_features = self.features(hr.float())
        return self.criterion(sr_features, hr_features)

 


class PixelLoss(nn.L1Loss):
    pass

class AdversarialLoss(nn.BCELoss):
    pass
