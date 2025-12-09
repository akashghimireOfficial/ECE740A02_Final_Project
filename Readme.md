# UNet-Based Image Segmentation with Explainable AI  

This project trains UNet-based segmentation models on two datasets (Tumor & Oxford-IIIT Pet) and provides explainability using Grad-CAM and multi-layer activation heatmaps.

The goal is simple:  
- train models  
- test segmentation  
- visualize what the model is "looking at"  

Everything is kept minimal so it fits directly into a reproducible workflow.

---

# Dataset Locations
Datasets must be placed inside:

[`./data/`](./data/)

Example structure:
```
data/
 ├── tumor/
 │    ├── train/
 │    ├── valid/
 │    ├── test/
 │    └── _annotations.coco.json
 │
 └── oxford-iiit-pet/
      (images + masks)
```

Direct links: [`./data/tumor/`](./data/tumor/) | [`./data/oxford-iiit-pet/`](./data/oxford-iiit-pet/)

No extra setup required beyond downloading datasets.

---

# Training Commands

## Tumor Dataset
```bash
python train_tumor.py --model resnet --aug --epochs 40 --batch_size 32 --gpu_id 0
```

## Oxford-IIIT Pet Dataset
```bash
python train_oxford.py --model resnet --aug --epochs 40 --batch_size 32 --gpu_id 0
```

**Args:**
- `--model {small,resnet}` : custom UNet or ResNet-UNet  
- `--aug` : enable augmentation  
- `--epochs N` : training epochs  
- `--batch_size N` : batch size  
- `--gpu_id N` : selects GPU, falls back to CPU  

---

# Testing Segmentation Performance

Run segmentation evaluation with Dice + IoU:
```bash
python test_seg.py \
    --dataset_name tumor \
    --model resnet \
    --aug \
    --epoch 40 \
    --save_vis
```

Output is saved inside: [`test_results/`](./test_results/)

Each sample contains:
- `orig.png`
- `gt.png`
- `pred.png`
- `overlay.png`
- `metrics.json`

---

# Grad-CAM Visualization

To see where the model focuses:
```bash
python run_gradcam.py \
    --dataset_name tumor \
    --model resnet \
    --aug \
    --epoch 40 \
    --device cpu \
    --sample_ids 0 3 7
```

Outputs go to: [`results/gradcam/`](./results/gradcam/)

---

# Multi-Layer Heatmaps (Encoder & Decoder Activations)
```bash
python run_heatmap.py \
    --dataset_name tumor \
    --model resnet \
    --aug \
    --epoch 40 \
    --device cpu \
    --sample_ids 0 1 2 3 4 5 6 7 8 9
```

Each sample folder contains:
- raw image
- heatmaps for multiple layers
- overlay visualizations

Saved under: [`results/heatmap/`](./results/heatmap/)

---

# Summary of Quantitative Results

### Oxford-IIIT Pet Dataset
| Model | Aug | Dice | IoU |
|-------|------|-------|--------|
| small | noaug | **0.8916** | **0.8163** |
| small | aug   | 0.7591 | 0.6408 |
| resnet | noaug | **0.9309** | **0.8765** |
| resnet | aug   | 0.7801 | 0.6648 |

### Tumor Dataset
| Model | Aug | Dice | IoU |
|-------|------|-------|--------|
| resnet | aug   | **0.9067** | **0.8443** |
| resnet | noaug | 0.8563 | 0.7813 |
| small | noaug | 0.8391 | 0.7581 |
| small | aug   | 0.8380 | 0.7599 |

---

# Observations
- Transfer learning (ResNet encoder) strongly boosts accuracy, especially on Oxford pets.  
- Augmentation helps more when dataset is small, e.g., brain tumor dataset.  
- Oxford dataset is already large & diverse, so augmentation sometimes reduces performance.  
- ResNet-UNet + no-augmentation gave the strongest performance overall.

---

# Final Notes
This repository contains:
- training scripts  
- testing scripts  
- Grad-CAM & heatmap explainability tools  
- organized output folders  

Everything is modular so you can replace datasets or models easily.