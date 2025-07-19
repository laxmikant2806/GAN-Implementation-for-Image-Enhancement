# models/discriminator.py
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_ch=1, base_nf=64, img_size=128):
        super().__init__()
        layers = []
        nf = base_nf
        def disc_block(in_f, out_f, stride=1, bn=True):
            blk = [nn.Conv2d(in_f, out_f, 3, stride, 1)]
            if bn:
                blk.append(nn.BatchNorm2d(out_f))
            blk.append(nn.LeakyReLU(0.2, inplace=True))
            return blk

        layers += disc_block(in_ch, nf, stride=1, bn=False)
        layers += disc_block(nf, nf, stride=2)          # 64×64
        layers += disc_block(nf, nf*2, stride=1)
        layers += disc_block(nf*2, nf*2, stride=2)      # 32×32
        layers += disc_block(nf*2, nf*4, stride=1)
        layers += disc_block(nf*4, nf*4, stride=2)      # 16×16
        layers += disc_block(nf*4, nf*8, stride=1)
        layers += disc_block(nf*8, nf*8, stride=2)      # 8×8

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((img_size//16)*(img_size//16)*nf*8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)
