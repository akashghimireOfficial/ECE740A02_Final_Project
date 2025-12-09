import torch
import argparse
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import  random

from model import build_model
from dataloader.oxford_dataloader import get_oxford_loaders
from dataloader.tumor_coco_dataloader import get_tumor_loaders

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------------------------------------
def normalize_img(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() + 1e-8)
    return (img * 255).astype(np.uint8)


def normalize_mask(mask_tensor):
    """Convert mask tensor to RGB visualization"""
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = np.nan_to_num(mask)
    
    # Create RGB visualization of mask
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Assign different colors to different classes
    unique_classes = np.unique(mask)
    colors = plt.cm.get_cmap('tab10', len(unique_classes))
    
    for idx, cls in enumerate(unique_classes):
        color = (np.array(colors(idx)[:3]) * 255).astype(np.uint8)
        mask_rgb[mask == cls] = color
    
    return mask_rgb


def simple_cam(act):
    act = act.squeeze(0)          # C,H,W
    cam = act.mean(dim=0).relu()  # H,W
    cam = cam.cpu().numpy()
    return cam


def cam_to_overlay(image, cam):
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_uint8 = cv2.resize(cam_uint8, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return (0.4 * heatmap + 0.6 * image).astype(np.uint8)


# --------------------------------------------------------
class MultiLayerHook:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.activations = {}

        for name, layer in layers.items():
            layer.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def fn(module, inp, out):
            self.activations[name] = out.detach()
        return fn

    def extract(self, img):
        self.model(img)
        return {k: simple_cam(v) for k, v in self.activations.items()}


# --------------------------------------------------------
def load_model(args, num_classes, device):
    model = build_model(args.model, num_classes)
    aug_folder = "aug" if args.aug else "noaug"
    ckpt = f"saved_model/{args.dataset_name}/{args.model}/{aug_folder}/epoch_{args.epoch}.pth"
    print(f"[INFO] Loading model: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()


# --------------------------------------------------------
def get_fixed_indices(total, args):
    if args.sample_ids:
        return sorted(list(set(args.sample_ids)))
    return list(range(min(args.num_samples, total)))


# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="tumor")
    parser.add_argument("--model", type=str, default="resnet", choices=["small", "resnet"])
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sample_ids", type=int, nargs="*", default=None)
    args = parser.parse_args()

    set_seed()
    
    if "cuda" in args.device and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using: {device}")

    # ------------------------------
    if args.dataset_name == "oxford":
        _, val_dl, _ = get_oxford_loaders(root="data", batch_size=1, aug=args.aug)
        num_classes = 3
    else:
        _, val_dl, _ = get_tumor_loaders(root="data/tumor", batch=1, img_size=256, aug=args.aug)
        num_classes = 2

    total_images = sum(1 for _ in val_dl)
    sample_indices = get_fixed_indices(total_images, args)
    print(f"[INFO] Will extract heatmaps for indices: {sample_indices}")

    model = load_model(args, num_classes, device)

    # ------------------------------
    # LAYER CHOICES
    if args.model == "resnet":
        layers = {
            "enc3": model.layer3[-1].conv2,
            "enc4": model.layer4[-1].conv2,
            "dec3": model.up3.conv.conv[0],
            "dec1": model.up1.conv.conv[0],
        }
    else:
        layers = {
            "enc2": model.e2.conv,
            "enc4": model.e4.conv,
            "bottleneck": model.b,
            "dec2": model.d2.conv,
            "dec4": model.d4.conv,
        }

    hooker = MultiLayerHook(model, layers)

    # save root
    out_root = f"results/heatmap/{args.dataset_name}/{args.model}_{'aug' if args.aug else 'noaug'}_ep{args.epoch}"
    os.makedirs(out_root, exist_ok=True)

    # --------------------------------------------------------
    global_idx = 0
    pbar = tqdm(sample_indices, desc="Heatmaps")

    for imgs, masks in val_dl:
        if global_idx in sample_indices:

            img_np = normalize_img(imgs[0])
            gt_mask = normalize_mask(masks[0])
            cams = hooker.extract(imgs.to(device))

            save_dir = f"{out_root}/sample_{global_idx:04d}"
            os.makedirs(save_dir, exist_ok=True)

            for name, cam in cams.items():
                overlay = cam_to_overlay(img_np, cam)
                
                # Combine: Original | Overlay | GT
                combined = np.hstack([img_np, overlay, gt_mask])
                
                cv2.imwrite(f"{save_dir}/{name}.png",
                            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            pbar.update(1)

        global_idx += 1

    pbar.close()
    print(f"[INFO] Heatmaps saved to {out_root}")


if __name__ == "__main__":
    main()