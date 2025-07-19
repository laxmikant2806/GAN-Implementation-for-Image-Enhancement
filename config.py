# config.py
from pathlib import Path

# ------------------ Paths ------------------
ROOT_DIR      = Path(__file__).resolve().parent
DATA_DIR      = ROOT_DIR / "data"
LOG_DIR       = ROOT_DIR / "logs"
CHECKPOINT_DIR= ROOT_DIR / "checkpoints"
VGG_WEIGHTS   = ROOT_DIR / "weights" / "vgg19-dcbb9e9d.pth"

# ------------------ Data -------------------
UPSCALE_FACTOR   = 4                 # 2×, 4×, or 8×
PATCH_SIZE_HR    = 128               # HR patch width/height
NORMALIZE_MEAN   = [0.5]             # For single-channel CT slices
NORMALIZE_STD    = [0.5]

# ------------------ Training ---------------
TOTAL_EPOCHS_RR  = 100               # SRResNet warm-up
TOTAL_EPOCHS_GAN = 200               # Adversarial fine-tuning
BATCH_SIZE       = 16
LR_INITIAL       = 1e-4
LR_MIN           = 1e-6
BETAS            = (0.9, 0.999)
PRECISION        = "amp"             # "amp" or "fp32"

# ------------------ Evaluation -------------
SAVE_INTERVAL      = 1               # epochs
VAL_INTERVAL       = 1
NUM_WORKERS        = 8
