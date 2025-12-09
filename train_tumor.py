import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from os.path import join
from argparse import ArgumentParser

from dataloader.tumor_coco_dataloader import get_tumor_loaders
from model import build_model
from loss_func import dice_iou_loss


def plot_losses(train_hist, val_hist, save_path=None):
    plt.figure(figsize=(6,4))
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.xlabel("epoch")
    plt.ylabel("dice + iou loss")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def compute_loss_on_batch(model, imgs, masks):
    imgs = imgs.cuda()
    masks = masks.cuda()
    logits = model(imgs)
    loss = dice_iou_loss(logits, masks)
    return loss


def main():

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="small",
                        choices=["small", "resnet"])
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)

    # -------------------------------
    # Tumor dataset dataloader
    # -------------------------------
    train_dl, val_dl, test_dl = get_tumor_loaders(
        root="data/tumor",
        batch=args.batch_size,
        img_size=args.img_size,
        aug=args.aug
    )

    # -------------------------------
    # Build model
    # Tumor segmentation = 2 classes
    # -------------------------------
    num_classes = 2
    model = build_model(args.model, num_classes).cuda()

    # -------------------------------
    # Optimizer
    # -------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # -------------------------------
    # Save directory
    # -------------------------------
    save_dir = join(
        "saved_model",
        "tumor",
        args.model,
        "aug" if args.aug else "noaug"
    )
    os.makedirs(save_dir, exist_ok=True)

    train_history = []
    val_history = []

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(args.epochs):

        model.train()
        train_loss = 0.0

        for imgs, masks in train_dl:
            loss = compute_loss_on_batch(model, imgs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)
        train_history.append(train_loss)

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, masks in val_dl:
                loss = compute_loss_on_batch(model, imgs, masks)
                val_loss += loss.item()

        val_loss /= len(val_dl)
        val_history.append(val_loss)

        print(f"epoch {epoch+1}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

        # save checkpoint
        ckpt_path = join(save_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

    # -------------------------------
    # Plot loss curves
    # -------------------------------
    plot_losses(
        train_history,
        val_history,
        save_path=join(save_dir, "loss_curve.png")
    )


if __name__ == "__main__":
    main()
