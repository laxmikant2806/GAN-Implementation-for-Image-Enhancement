# inference.py
import argparse, cv2, torch
from torchvision.transforms.functional import normalize
from models.generator import Generator
import config as C

def upscale(img_path, weight_path, out_path):
    net = Generator(upscale=C.UPSCALE_FACTOR)
    net.load_state_dict(torch.load(weight_path, map_location="cpu"))
    net.eval()

    lr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype("float32")/255.
    if lr.ndim == 2: lr = lr[...,None]
    lr_t = torch.from_numpy(lr.transpose(2,0,1))[None]
    lr_t = normalize(lr_t, C.NORMALIZE_MEAN, C.NORMALIZE_STD)

    with torch.no_grad():
        sr_t = net(lr_t)
    sr = sr_t.squeeze().clamp(0,1).cpu().numpy().transpose(1,2,0)*255.
    cv2.imwrite(out_path, sr.astype("uint8"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--weights", default="checkpoints/srgan_best.pth")
    args = p.parse_args()
    upscale(args.input, args.weights, args.output)
