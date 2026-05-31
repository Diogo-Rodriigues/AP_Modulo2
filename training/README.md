# Training / Experiments

This folder contains the notebooks and artifacts for training and evaluating ERCP image classification models using the split defined in the repository.

## Background & Dataset Balancing

This work is built upon a base project developed by a colleague. As part of our improvements, we applied data augmentation (`data_aug.ipynb`) to balance the dataset, ensuring a fair evaluation by leaving the test splits with exactly 600 images each.

---

## Notebooks

- `data_aug.ipynb` - Data augmentation to balance the dataset.
- `auto_invert.ipynb` - Find the images that are inverted.
- `remove_black_bars.ipynb` - Script to preprocess images by removing black bars.
- `split_dataset.ipynb` - script/notebook used to create and validate the split (`train` / `val` / `test`) - note: fixed splits are included under `training/dataset/`.
- `RESNET.ipynb` - training and evaluation for a ResNet-based model.
- `DENSENET.ipynb` - DenseNet-based training and evaluation.
- `EFICIENTNET.ipynb` - EfficientNet-based experiments.
- `EFICIENTNETB0.ipynb` - EfficientNetB0-specific model training and evaluation.
- `CONVNEXTV2.ipynb` - ConvNeXt V2 experiments.
- `SWINTRANSFORMER.ipynb` - Swin Transformer experiments.
- `DEITIII.ipynb` - DeiT experiments.
- `MOBILENET.ipynb` - MobileNet-based experiments.
- `MIQR_compare_models_gradcam.ipynb` - Comparison of model performance and interpretability analysis using Grad-CAM.
- `MIQR_occlusion.ipynb` - Occlusion sensitivity experiments for model explainability.
- `models/` - Trained model checkpoints used for experiments (and summary visuals).

## Scripts

- `invert.py` - Script to invert colors of images in-place.
- `test.py` - Simple utility script to count unique patient IDs based on filenames.

Each notebook contains data loading, augmentations, training loop, validation and test evaluation, and plotting (confusion matrix, learning curves).

## Requirements

Install the training environment with:

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

(`training/requirements.txt` contains the pinned packages used for experiments.)

## Models & Results

Trained checkpoints and summary visuals are under `training/models/` (checkpoints have the `.pth` extension; summary images include confusion matrices `.png`). See `training/models/README.md` for details on how to load them.

Additionally, overall comparison metrics and visual outputs are stored in `training/results/`:
- `model_comparison_test_metrics.csv` - Table containing final test metrics for all models.
- `f1_macro_comparison.png` - Visual comparison of the F1-macro scores.
- `cm_*.png` - Generated confusion matrices for individual models.

## Reproducibility notes

- Fixed splits are included in `training/dataset/` (no need to re-run `split_dataset.ipynb` unless you want a different split).
- Check each notebook for hyperparameters and random-seed usage.
- The dataset used for the models is on `training/dataset_cleaned/` (no need to re-run `auto_invert.ipynb`, `data_aug.ipynb` and `remove_black_bars.ipynb` unless you want different preprocessed images). If needed, to run `auto_invert.ipynb`, you need to download the full dataset and call it data_images
