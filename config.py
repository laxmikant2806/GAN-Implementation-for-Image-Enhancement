from pathlib import Path

# ------------------ Paths ------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
VGG_WEIGHTS = ROOT_DIR / "weights" / "vgg19-dcbb9e9d.pth"

# ------------------ Data -------------------
UPSCALE_FACTOR = 4
PATCH_SIZE_HR = 128
NORMALIZE_MEAN = [0.5]
NORMALIZE_STD = [0.5]

# ------------------ Training ---------------
TOTAL_EPOCHS_RR = 50       # Reduced for faster experimentation
TOTAL_EPOCHS_GAN = 100     # Reduced for faster experimentation
BATCH_SIZE = 32            # Increased for better GPU utilization
LR_INITIAL = 1e-4
LR_MIN = 1e-6
BETAS = (0.9, 0.999)
PRECISION = "amp"          # Use mixed precision

# ------------------ Evaluation -------------
SAVE_INTERVAL = 10         # Save every 10 epochs
VAL_INTERVAL = 5           # Evaluate every 5 epochs
NUM_WORKERS = 8            # Optimize based on your CPU cores
