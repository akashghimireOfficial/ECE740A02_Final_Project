import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from os.path import join
from argparse import ArgumentParser

from dataloader.oxford_dataloader import get_oxford_loaders
from model import build_model          ## updated import
from loss_func import dice_iou_loss


## simple plot function
def plot_losses(train_hist, val_hist, save_path=None):
    plt.figure(figsize=(6,4))
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.xlabel("epoch")
    plt.ylabel("dice+iou loss")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


## compute loss for a batch
def compute_loss_on_batch(model, imgs, masks):
    imgs = imgs.cuda()
    masks = masks.cuda()
    logits = model(imgs)
    loss = dice_iou_loss(logits, masks)
    return loss


def main():

    ## args
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="small", choices=["small", "resnet"])
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    ## gpu select
    torch.cuda.set_device(args.gpu_id)

    ## dataloader
    train_dl, val_dl, test_dl = get_oxford_loaders(
        root="data",
        batch_size=args.batch_size,
        aug=args.aug
    )

    ## build selected model
    num_classes = 3
    model = build_model(args.model, num_classes).cuda()

    ## optimizer (better than vanilla Adam)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ## prepare save folder
    save_dir = join(
        "saved_model",
        "oxford",
        args.model,                ## save by model name
        "aug" if args.aug else "noaug"
    )
    os.makedirs(save_dir, exist_ok=True)

    ## history for plotting
    train_history = []
    val_history = []

    ## train loop
    for epoch in range(args.epochs):

        ## train step
        model.train()
        train_loss = 0

        for imgs, masks in train_dl:
            loss = compute_loss_on_batch(model, imgs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dl)
        train_history.append(train_loss)

        ## val step
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, masks in val_dl:
                loss = compute_loss_on_batch(model, imgs, masks)
                val_loss += loss.item()

        val_loss /= len(val_dl)
        val_history.append(val_loss)

        print(f"epoch {epoch+1}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

        ## save ckpt
        ckpt_path = join(save_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

    ## plot losses
    plot_losses(
        train_history,
        val_history,
        save_path=join(save_dir, "loss_curve.png")
    )


if __name__ == "__main__":
    main()
