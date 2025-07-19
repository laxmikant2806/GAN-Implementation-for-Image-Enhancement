import torch
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast  # Use cuda AMP without device_type
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.generator import Generator
from models.discriminator import Discriminator
from losses import ContentLoss, PixelLoss, AdversarialLoss
from dataset import PairedImageDataset
import config as C
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
import os

def save_samples(generator, val_loader, epoch, tag="val"):
    generator.eval()
    with torch.no_grad():
        lr, hr = next(iter(val_loader))
        lr, hr = lr.cuda(), hr.cuda()
        sr = generator(lr)
        # Upsample low-resolution input to match high-resolution dimensions
        lr_upsampled = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        # Concatenate: all tensors now [batch, channels, H, W]
        grid = torch.cat([lr_upsampled, sr, hr], dim=0)
        # Make sure the log directory exists
        os.makedirs(C.LOG_DIR, exist_ok=True)
        save_image(grid, f"{C.LOG_DIR}/{tag}_epoch_{epoch}.png", nrow=lr.size(0))
    generator.train()


def main():
    train_set = PairedImageDataset(C.DATA_DIR, C.UPSCALE_FACTOR, C.PATCH_SIZE_HR, split='train')
    val_set   = PairedImageDataset(C.DATA_DIR, C.UPSCALE_FACTOR, C.PATCH_SIZE_HR, split='val')

    t_loader  = DataLoader(train_set, C.BATCH_SIZE, shuffle=True,
                           num_workers=C.NUM_WORKERS, pin_memory=True)
    v_loader  = DataLoader(val_set,  C.BATCH_SIZE, shuffle=False,
                           num_workers=C.NUM_WORKERS, pin_memory=True)

    G = Generator(upscale=C.UPSCALE_FACTOR).cuda()
    D = Discriminator(img_size=C.PATCH_SIZE_HR).cuda()

    pixel_loss    = PixelLoss()
    content_loss  = ContentLoss(C.VGG_WEIGHTS)
    adv_loss      = AdversarialLoss()
    opt_G = Adam(G.parameters(), lr=C.LR_INITIAL, betas=C.BETAS)
    opt_D = Adam(D.parameters(), lr=C.LR_INITIAL, betas=C.BETAS)
    
    scaler = GradScaler(enabled=(C.PRECISION == "amp"))
    BCE_Y_real = torch.ones((C.BATCH_SIZE, 1), device="cuda")
    BCE_Y_fake = torch.zeros((C.BATCH_SIZE, 1), device="cuda")

    # ----- Pretraining: SRResNet -----
    print("Starting SRResNet pretraining...")
    for epoch in range(C.TOTAL_EPOCHS_RR):
        loop = tqdm(t_loader, desc=f"Pretrain Epoch {epoch+1}/{C.TOTAL_EPOCHS_RR}", unit="batch")
        for lr, hr in loop:
            lr, hr = lr.cuda(), hr.cuda()
            opt_G.zero_grad(set_to_none=True)

            with autocast(enabled=(C.PRECISION == "amp")):
                sr = G(lr)
                loss_pixel = pixel_loss(sr, hr)
                with autocast(enabled=False):
                    vgg_loss = content_loss(sr.float(), hr.float())
                loss = loss_pixel + 0.006 * vgg_loss

            scaler.scale(loss).backward()
            scaler.step(opt_G)
            scaler.update()

            # Show current loss in progress bar
            loop.set_postfix(loss=loss.item())

        if epoch % C.SAVE_INTERVAL == 0:
            torch.save(G.state_dict(), C.CHECKPOINT_DIR / f"srresnet_{epoch}.pth")
            save_samples(G, v_loader, epoch, tag="rr")


    # ----- SRGAN Adversarial Fine-Tuning -----
    print("Starting SRGAN adversarial fine-tuning...")
    for epoch in range(C.TOTAL_EPOCHS_GAN):
        loop = tqdm(t_loader, desc=f"GAN Epoch {epoch+1}/{C.TOTAL_EPOCHS_GAN}", unit="batch")
        for lr, hr in loop:
            lr, hr = lr.cuda(), hr.cuda()
            opt_D.zero_grad(set_to_none=True)
            with torch.no_grad():
                sr = G(lr)
            D_real = D(hr)
            D_fake = D(sr.detach())
            loss_D = (adv_loss(D_real, BCE_Y_real[:len(D_real)]) +
                    adv_loss(D_fake, BCE_Y_fake[:len(D_fake)])) / 2
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            scaler.update()

            opt_G.zero_grad(set_to_none=True)
            with autocast(enabled=(C.PRECISION == "amp")):
                sr = G(lr)
                D_fake = D(sr)
                with autocast(enabled=False):
                    vgg_loss = content_loss(sr.float(), hr.float())
                loss_G = (1e-2 * adv_loss(D_fake, BCE_Y_real[:len(D_fake)]) +
                        0.006 * vgg_loss +
                        pixel_loss(sr, hr))
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

            # Show both generator and discriminator loss in progress bar
            loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

        if epoch % C.SAVE_INTERVAL == 0:
            os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
            print(f"Epoch {epoch}: Loss G: {loss_G.item()}, Loss D: {loss_D.item()}")
            torch.save(G.state_dict(), C.CHECKPOINT_DIR / f"srgan_{epoch}.pth")
            save_samples(G, v_loader, epoch, tag="gan")


if __name__ == "__main__":
    main()
