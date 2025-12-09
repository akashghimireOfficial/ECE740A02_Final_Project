import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
import numpy as np
import torch.nn.functional as F


# ------------------------------------------------------
# Image transforms
# ------------------------------------------------------
def get_image_transform(aug=False):
    if aug:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
    else:
        return transforms.Compose([])


# ------------------------------------------------------
# Oxford Loader
# ------------------------------------------------------
def get_oxford_loaders(
    root="data",
    batch_size=8,
    aug=False,
    train_ratio=0.8,
    val_ratio=0.1
):

    base_tf = transforms.Resize((256, 256))   # ONLY for images (PIL)
    to_tensor = transforms.ToTensor()

    ds = OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="segmentation",
        download=True,
        transform=base_tf,          # image → PIL resized
        target_transform=base_tf,   # mask → PIL resized
    )

    class FixedDataset(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, mask = self.base[idx]     # img=PIL resized, mask=PIL resized

            # convert to tensors
            img = to_tensor(img)           # (3,H,W)
            mask = torch.tensor(np.array(mask)).long() - 1   # (H,W)

            mask = mask.unsqueeze(0)       # (1,H,W)

            return img, mask

    ds = FixedDataset(ds)

    # Split dataset
    N = len(ds)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N - n_train - n_val

    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])

    # Apply augmentation only to train, but mask remains untouched
    train_img_tf = get_image_transform(aug=True if aug else False)
    clean_img_tf = get_image_transform(aug=False)

    class AugWrapper(torch.utils.data.Dataset):
        def __init__(self, subset, img_tf):
            self.subset = subset
            self.img_tf = img_tf

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, mask = self.subset[idx]
            img = self.img_tf(img)   # Apply augmentation only to images
            return img, mask

    train_ds = AugWrapper(train_ds, train_img_tf)
    val_ds   = AugWrapper(val_ds, clean_img_tf)
    test_ds  = AugWrapper(test_ds, clean_img_tf)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )