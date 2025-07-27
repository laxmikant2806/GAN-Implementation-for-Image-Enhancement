import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import time
import math
import os

from models.generator import Generator
from models.discriminator import Discriminator
from losses import ContentLoss, PixelLoss, AdversarialLoss
from dataset import PairedImageDataset
import config as C

# PSNR Calculation Function
def psnr(img1, img2, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

# SSIM Implementation
class SSIM:
    """Structural Similarity Index Measure"""
    def __init__(self, window_size=11):
        self.window_size = window_size
        self.window = self.create_window(window_size)
    
    def gaussian(self, window_size, sigma=1.5):
        gauss = torch.Tensor([
            math.exp(-(x - window_size//2)**2 / (2*sigma**2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size):
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window
    
    def __call__(self, img1, img2):
        if img1.is_cuda:
            self.window = self.window.cuda()
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2) - mu1_mu2
        
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()

# Model Evaluation Function
def evaluate_model(generator, val_loader, ssim_metric, device='cuda'):
    """Comprehensive model evaluation"""
    generator.eval()
    total_psnr = 0
    total_ssim = 0
    total_samples = 0
    
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = generator(lr)
            
            # Calculate metrics for each image in batch
            for i in range(lr.size(0)):
                psnr_val = psnr(sr[i:i+1], hr[i:i+1])
                ssim_val = ssim_metric(sr[i:i+1], hr[i:i+1])
                
                total_psnr += psnr_val.item()
                total_ssim += ssim_val.item()
                total_samples += 1
    
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    
    generator.train()
    return avg_psnr, avg_ssim

# Adaptive Loss Weights Function
def get_adaptive_weights(epoch, total_epochs):
    """Calculate adaptive loss weights based on training progress"""
    # Increase adversarial loss weight gradually
    adv_weight = min(1e-2 * (epoch / 50), 1e-2)
    # Decrease content loss weight over time
    content_weight = max(0.006 * (1 - epoch / total_epochs), 0.001)
    return adv_weight, content_weight

# Optimized Save Samples Function
def save_samples(generator, val_loader, epoch, tag="val"):
    generator.eval()
    with torch.no_grad():
        lr, hr = next(iter(val_loader))
        lr, hr = lr.cuda(non_blocking=True), hr.cuda(non_blocking=True)
        sr = generator(lr)
        # Upsample low-resolution input to match high-resolution dimensions
        lr_upsampled = F.interpolate(lr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        # Concatenate: all tensors now [batch, channels, H, W]
        grid = torch.cat([lr_upsampled, sr, hr], dim=0)
        # Make sure the log directory exists
        os.makedirs(C.LOG_DIR, exist_ok=True)
        save_image(grid, f"{C.LOG_DIR}/{tag}_epoch_{epoch}.png", nrow=lr.size(0))
    generator.train()

# Main Training Function
def main():

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Initialize datasets
    train_set = PairedImageDataset(C.DATA_DIR, C.UPSCALE_FACTOR, C.PATCH_SIZE_HR, split='train')
    val_set = PairedImageDataset(C.DATA_DIR, C.UPSCALE_FACTOR, C.PATCH_SIZE_HR, split='val')

    # Optimized DataLoaders
    t_loader = DataLoader(
        train_set, 
        C.BATCH_SIZE, 
        shuffle=True,
        num_workers=C.NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    v_loader = DataLoader(
        val_set, 
        C.BATCH_SIZE, 
        shuffle=False,
        num_workers=max(1, C.NUM_WORKERS//2), 
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize models
    G = Generator(upscale=C.UPSCALE_FACTOR).cuda()
    D = Discriminator(img_size=C.PATCH_SIZE_HR).cuda()

    # Try to use torch.compile for faster training (PyTorch 2.0+)
    try:
        G = torch.compile(G)
        D = torch.compile(D)
        print("Using torch.compile for faster training")
    except:
        print("torch.compile not available, using standard models")

    # Initialize losses and optimizers
    pixel_loss = PixelLoss()
    content_loss = ContentLoss(C.VGG_WEIGHTS)
    adv_loss = AdversarialLoss()
    
    opt_G = Adam(G.parameters(), lr=C.LR_INITIAL, betas=C.BETAS, weight_decay=1e-5)
    opt_D = Adam(D.parameters(), lr=C.LR_INITIAL, betas=C.BETAS, weight_decay=1e-5)
    
    # Learning rate schedulers
    scheduler_G = StepLR(opt_G, step_size=30, gamma=0.5)
    scheduler_D = StepLR(opt_D, step_size=30, gamma=0.5)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=(C.PRECISION == "amp"))
    
    # BCE targets for discriminator
    BCE_Y_real = torch.ones((C.BATCH_SIZE, 1), device="cuda")
    BCE_Y_fake = torch.zeros((C.BATCH_SIZE, 1), device="cuda")
    
    # Initialize evaluation metrics
    ssim_metric = SSIM()
    
    # Ensure directories exist
    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(C.LOG_DIR, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # ----- Phase 1: SRResNet Pretraining -----
    print("Starting SRResNet pretraining...")
    for epoch in range(C.TOTAL_EPOCHS_RR):
        loop = tqdm(t_loader, desc=f"Pretrain Epoch {epoch+1}/{C.TOTAL_EPOCHS_RR}")
        
        for lr, hr in loop:
            lr, hr = lr.cuda(non_blocking=True), hr.cuda(non_blocking=True)
            opt_G.zero_grad(set_to_none=True)

            with autocast(enabled=(C.PRECISION == "amp")):
                sr = G(lr)
                loss_pixel = pixel_loss(sr, hr)
                # Disable autocast for VGG feature extraction
                with autocast(enabled=False):
                    vgg_loss = content_loss(sr.float(), hr.float())
                loss = loss_pixel + 0.006 * vgg_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            scaler.step(opt_G)
            scaler.update()
            
            # Update progress bar
            loop.set_postfix(loss=loss.item())
        
        # Step scheduler
        scheduler_G.step()
        
        # Evaluate every 10 epochs during pretraining
        if epoch % 10 == 0 and epoch > 0:
            psnr_val, ssim_val = evaluate_model(G, v_loader, ssim_metric)
            elapsed_time = (time.time() - start_time) / 3600
            print(f"Pretrain Epoch {epoch}: PSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.4f}, Time: {elapsed_time:.2f}h")

        # Save checkpoints
        if epoch % C.SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': G.state_dict(),
                'optimizer_G_state_dict': opt_G.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
            }, C.CHECKPOINT_DIR / f"srresnet_{epoch}.pth")
            save_samples(G, v_loader, epoch, tag="rr")

    # Evaluate after pretraining
    print("Evaluating pretrained SRResNet...")
    psnr_pretrain, ssim_pretrain = evaluate_model(G, v_loader, ssim_metric)
    print(f"SRResNet Results - PSNR: {psnr_pretrain:.2f}dB, SSIM: {ssim_pretrain:.4f}")

    # ----- Phase 2: SRGAN Adversarial Fine-Tuning -----
    print("Starting SRGAN adversarial fine-tuning...")
    for epoch in range(C.TOTAL_EPOCHS_GAN):
        loop = tqdm(t_loader, desc=f"GAN Epoch {epoch+1}/{C.TOTAL_EPOCHS_GAN}")
        
        for batch_idx, (lr, hr) in enumerate(loop):
            lr, hr = lr.cuda(non_blocking=True), hr.cuda(non_blocking=True)
            
            # Train Discriminator (every 2 iterations to prevent overpowering generator)
            if batch_idx % 2 == 0:
                opt_D.zero_grad(set_to_none=True)
                with torch.no_grad():
                    sr = G(lr)
                
                D_real = D(hr)
                D_fake = D(sr.detach())
                loss_D = (adv_loss(D_real, BCE_Y_real[:len(D_real)]) +
                         adv_loss(D_fake, BCE_Y_fake[:len(D_fake)])) / 2
                
                scaler.scale(loss_D).backward()
                scaler.unscale_(opt_D)
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                scaler.step(opt_D)
                scaler.update()

            # Train Generator
            opt_G.zero_grad(set_to_none=True)
            with autocast(enabled=(C.PRECISION == "amp")):
                sr = G(lr)
                D_fake = D(sr)
                
                # Get adaptive weights
                adv_weight, content_weight = get_adaptive_weights(epoch, C.TOTAL_EPOCHS_GAN)
                
                with autocast(enabled=False):
                    vgg_loss = content_loss(sr.float(), hr.float())
                
                loss_G = (adv_weight * adv_loss(D_fake, BCE_Y_real[:len(D_fake)]) +
                         content_weight * vgg_loss +
                         pixel_loss(sr, hr))
            
            scaler.scale(loss_G).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            scaler.step(opt_G)
            scaler.update()
            
            # Update progress bar
            if batch_idx % 2 == 0:
                loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
            else:
                loop.set_postfix(loss_G=loss_G.item())

        # Step schedulers
        scheduler_G.step()
        scheduler_D.step()
        
        # Evaluate every 5 epochs during GAN training
        if epoch % 5 == 0:
            psnr_val, ssim_val = evaluate_model(G, v_loader, ssim_metric)
            elapsed_time = (time.time() - start_time) / 3600
            print(f"GAN Epoch {epoch}: PSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.4f}, Time: {elapsed_time:.2f}h")

        # Save checkpoints
        if epoch % C.SAVE_INTERVAL == 0:
            # Get current metrics for saving
            current_psnr, current_ssim = evaluate_model(G, v_loader, ssim_metric) if epoch % 5 == 0 else (None, None)
            
            torch.save({
                'epoch': epoch,
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'optimizer_G_state_dict': opt_G.state_dict(),
                'optimizer_D_state_dict': opt_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'psnr': current_psnr,
                'ssim': current_ssim,
            }, C.CHECKPOINT_DIR / f"srgan_{epoch}.pth")
            
            save_samples(G, v_loader, epoch, tag="gan")

    # Final evaluation
    print("Final evaluation...")
    final_psnr, final_ssim = evaluate_model(G, v_loader, ssim_metric)
    total_time = (time.time() - start_time) / 3600
    print(f"Training Complete!")
    print(f"Final Results - PSNR: {final_psnr:.2f}dB, SSIM: {final_ssim:.4f}")
    print(f"Total Training Time: {total_time:.2f} hours")

    # Save final model
    torch.save({
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'final_psnr': final_psnr,
        'final_ssim': final_ssim,
        'total_training_time_hours': total_time,
        'training_complete': True
    }, C.CHECKPOINT_DIR / "srgan_final.pth")

if __name__ == "__main__":
    main()
