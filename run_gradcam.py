import torch
import argparse
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from model import build_model
from dataloader.oxford_dataloader import get_oxford_loaders
from dataloader.tumor_coco_dataloader import get_tumor_loaders
from gradcam import GradCAM_ResNetUNet


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_img(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = np.nan_to_num(img)
    img = (img - img.min()) / (img.max() + 1e-8)
    return (img * 255).astype(np.uint8)


def overlay_cam(image, cam):
    cam = np.asarray(cam).squeeze()
    cam = np.nan_to_num(cam)

    if cam.ndim != 2:
        cam = cam[..., 0]

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_uint8 = cv2.resize(cam_uint8, (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return (0.4 * heatmap + 0.6 * image).astype(np.uint8)


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


# ---------------------------------------------------------
def load_model(args, num_classes, device):
    model = build_model(args.model, num_classes)
    aug_folder = "aug" if args.aug else "noaug"
    ckpt_path = f"saved_model/{args.dataset_name}/{args.model}/{aug_folder}/epoch_{args.epoch}.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model


# ---------------------------------------------------------
def get_fixed_indices(total, args):
    """deterministic selection of samples"""
    if args.sample_ids:
        return sorted(list(set(args.sample_ids)))

    # fallback: first N samples
    return list(range(min(args.num_samples, total)))


# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="tumor", choices=["tumor", "oxford"])
    parser.add_argument("--model", type=str, default="resnet", choices=["small", "resnet"])
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sample_ids", type=int, nargs="*", default=None)
    args = parser.parse_args()

    set_seed()

    # device
    if "cuda" in args.device and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] Using device: {device}")

    # dataset
    if args.dataset_name == "oxford":
        _, val_dl, _ = get_oxford_loaders(root="data", batch_size=1, aug=args.aug)
        num_classes = 3
    else:
        _, val_dl, _ = get_tumor_loaders(root="data/tumor", batch=1, img_size=256, aug=args.aug)
        num_classes = 2

    # count total validation images
    total_images = sum(1 for _ in val_dl)
    sample_indices = get_fixed_indices(total_images, args)
    print(f"[INFO] Will visualize sample indices: {sample_indices}")

    # model
    model = load_model(args, num_classes, device)
    cam_extractor = GradCAM_ResNetUNet(model)

    # output folder
    save_dir = f"results/gradcam/{args.dataset_name}/{args.model}_{'aug' if args.aug else 'noaug'}_ep{args.epoch}"
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------
    # iterate deterministically
    # ---------------------------------------------
    global_idx = 0
    pbar = tqdm(sample_indices, desc="GradCAM")

    for imgs, masks in val_dl:
        if global_idx in sample_indices:

            img_tensor = imgs.to(device)
            cam = cam_extractor(img_tensor)
            if torch.is_tensor(cam):
                cam = cam.cpu().numpy()

            # Get original image
            img_np = normalize_img(imgs[0])
            
            # Get overlay
            overlay = overlay_cam(img_np, cam)
            
            # Get GT mask
            gt_mask = normalize_mask(masks[0])

            # Concatenate horizontally: Original | Overlay | GT
            combined = np.hstack([img_np, overlay, gt_mask])

            base = f"sample_{global_idx:04d}"
            cv2.imwrite(f"{save_dir}/{base}.png", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            pbar.update(1)

        global_idx += 1

    pbar.close()
    print(f"[INFO] Saved results in: {save_dir}")


if __name__ == "__main__":
    main()