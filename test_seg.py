import torch
import torch.nn.functional as F
import os
from os.path import join
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import json
import random

from dataloader.oxford_dataloader import get_oxford_loaders
from dataloader.tumor_coco_dataloader import get_tumor_loaders
from model import build_model


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_metric(pred, target, eps=1e-6):
    pred = pred.argmax(dim=1)
    target = target.squeeze(1)

    num_classes = int(max(pred.max(), target.max()).item()) + 1
    dice_scores = []

    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice = (2 * inter + eps) / (union + eps)
        dice_scores.append(dice.item())

    return sum(dice_scores) / len(dice_scores)


def iou_metric(pred, target, eps=1e-6):
    pred = pred.argmax(dim=1)
    target = target.squeeze(1)

    num_classes = int(max(pred.max(), target.max()).item()) + 1
    iou_scores = []

    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        union = (p + t - (p * t)).sum()
        iou = (inter + eps) / (union + eps)
        iou_scores.append(iou.item())

    return sum(iou_scores) / len(iou_scores)


def save_overlay(img, mask, pred, save_path):
    img_np = img.permute(1,2,0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() + 1e-6)

    mask_np = mask.cpu().numpy()
    pred_np = pred.cpu().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(14, 4))

    ax[0].imshow(img_np)
    ax[0].axis("off")

    ax[1].imshow(mask_np, cmap="jet", alpha=0.7)
    ax[1].axis("off")

    ax[2].imshow(pred_np, cmap="jet", alpha=0.7)
    ax[2].axis("off")

    ax[3].imshow(img_np)
    ax[3].imshow(pred_np, cmap="jet", alpha=0.4)
    ax[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def main():

    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["oxford", "tumor"])
    parser.add_argument("--model", type=str, default="small",
                        choices=["small", "resnet"])
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--max_vis", type=int, default=20)
    args = parser.parse_args()

    set_seed()

    torch.cuda.set_device(args.gpu_id)

    if args.dataset_name == "oxford":
        _, _, test_dl = get_oxford_loaders("data", args.batch_size, aug=args.aug)
        num_classes = 3
    else:
        _, _, test_dl = get_tumor_loaders(
            root="data/tumor",
            batch=args.batch_size,
            img_size=args.img_size,
            aug=args.aug
        )
        num_classes = 2

    aug_folder = "aug" if args.aug else "noaug"
    ckpt_path = join("saved_model", args.dataset_name, args.model, aug_folder,
                     f"epoch_{args.epoch}.pth")

    print(f"[INFO] Loading: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    model = build_model(args.model, num_classes).cuda()
    model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
    model.eval()

    save_root = join("test_results", args.dataset_name, args.model, aug_folder)
    os.makedirs(save_root, exist_ok=True)

    sample_folder = join(save_root, "samples")
    os.makedirs(sample_folder, exist_ok=True)

    dices = []
    ious = []

    vis_counter = 0

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(test_dl):

            imgs = imgs.cuda()
            masks = masks.cuda()

            logits = model(imgs)

            dices.append(dice_metric(logits, masks))
            ious.append(iou_metric(logits, masks))

            pred_masks = logits.argmax(dim=1)

            for b in range(imgs.size(0)):

                if vis_counter >= args.max_vis:
                    break

                save_path = join(sample_folder, f"sample_{vis_counter:03d}.png")

                save_overlay(
                    img=imgs[b].cpu(),
                    mask=masks[b][0].cpu(),
                    pred=pred_masks[b].cpu(),
                    save_path=save_path
                )

                vis_counter += 1

            if vis_counter >= args.max_vis:
                break

    mean_dice = float(sum(dices)/len(dices))
    mean_iou = float(sum(ious)/len(ious))

    print("====================================")
    print("Test Dice:", mean_dice)
    print("Test IoU:", mean_iou)
    print("Saved visual samples:", vis_counter)
    print("====================================")

    metrics = {
        "dataset": args.dataset_name,
        "model": args.model,
        "augmentation": args.aug,
        "epoch": args.epoch,
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "samples_visualized": vis_counter
    }

    with open(join(save_root, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Metrics saved to {join(save_root, 'metrics.json')}")


if __name__ == "__main__":
    main()