import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms.functional as TF
import random


class TumorDataset(Dataset):
    def __init__(self, root, img_size=256, aug=False):
        self.root = root
        self.img_size = img_size
        self.aug = aug

        ann_path = os.path.join(root, "_annotations.coco.json")
        self.coco = COCO(ann_path)
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    # ---------------------------------------------------------
    # augmentation (used ONLY for training)
    # ---------------------------------------------------------
    def _apply_augmentations(self, img, mask):

        # horizontal flip
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # vertical flip
        if random.random() < 0.2:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # rotation
        angle = random.uniform(-15, 15)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)

        # brightness & contrast jitter (image only)
        if random.random() < 0.7:
            img = TF.adjust_brightness(img, 0.8 + random.random() * 0.4)
            img = TF.adjust_contrast(img, 0.8 + random.random() * 0.4)

        return img, mask

    # ---------------------------------------------------------
    # main loader
    # ---------------------------------------------------------
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.root, info["file_name"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # mask creation
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)

        for ann in anns:
            m = self.coco.annToMask(ann)
            mask = np.maximum(mask, m)

        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # convert to tensors
        img = TF.to_tensor(img)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

        # apply augmentation ONLY IF aug=True
        if self.aug:
            img, mask = self._apply_augmentations(img, mask)

        return img, mask


# ---------------------------------------------------------
# Dataloader wrapper (train uses aug, val/test are clean)
# ---------------------------------------------------------
def get_tumor_loaders(root="tumor_dataset", batch=8, img_size=256, aug=False):

    train = TumorDataset(os.path.join(root, "train"), img_size, aug=aug)
    val   = TumorDataset(os.path.join(root, "valid"), img_size, aug=False)
    test  = TumorDataset(os.path.join(root, "test"),  img_size, aug=False)

    train_dl = DataLoader(train, batch_size=batch, shuffle=True, num_workers=4)
    val_dl   = DataLoader(val,   batch_size=batch, shuffle=False, num_workers=4)
    test_dl  = DataLoader(test,  batch_size=batch, shuffle=False, num_workers=4)

    return train_dl, val_dl, test_dl
