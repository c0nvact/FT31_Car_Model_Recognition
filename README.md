# Car Model Recognition Final Project

Notebook-based training and evaluation of car model classifiers on Stanford Cars, plus a Streamlit inference app.

## Project Files

- `Train_ConvNeXt_S.ipynb`: training for ConvNeXt-S (224 and 320)
- `Train_ResNet101.ipynb`: training for ResNet101 (224 and 320)
- `Train_EfficientNetV2_M.ipynb`: training for EfficientNetV2-M (224 and 320)
- `Eval_Models_Local.ipynb`: local evaluation, confusion analysis, and CAM visualizations
- `streamlit_app.py`: inference UI for uploaded/URL images
- `cars_dataset.py`: dataset class and crop utilities
- `data_utils.py`: annotation loading and preprocessing helpers
- `train_utils.py`: train/eval loops, checkpoint save, and plotting
- `checkpoints/`: Saved model checkpoints (`*_best.pth`) (ignored by git)
- `stanford_cars/`: Stanford Cars dataset files (ignored by git)

## Dataset

Source:
- `https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder`

Expected local path:
- `./stanford_cars`

Expected files used by notebooks/app:
- `names.csv`, `anno_train.csv`, `anno_test.csv`
- images in `stanford_cars/car_data/car_data/train` and `.../test`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training (Notebook Only)

Run one of:
- `Train_ConvNeXt_S.ipynb`
- `Train_ResNet101.ipynb`
- `Train_EfficientNetV2_M.ipynb`

In each notebook:
1. Verify `CFG.data_root` and `CFG.save_dir`.
2. Run cells top-to-bottom.

Training details (concise):
- Uses square crop from full image (`square_margin=0.10`)
- Trains at resolutions `224` and `320`
- Optimizer: `AdamW`
- Loss: cross entropy with label smoothing
- Tracks train/test loss, top-1, and top-5 each epoch

## Checkpoints

Saved in:
- `./checkpoints`

For each run (example: `resnet101_320`):
- `<run_name>_last.pth`: saved every epoch
- `<run_name>_best.pth`: saved when test top-1 improves

Checkpoint includes:
- model weights, optimizer state, epoch, best metric, cfg, history

## Evaluation (Notebook Only)

Run:
- `Eval_Models_Local.ipynb`

Steps:
1. Ensure trained `*_best.pth` checkpoints exist in `./checkpoints`.
2. Verify `CFG.data_root` and `CFG.save_dir`.
3. Run cells top-to-bottom.

Outputs:
- Test loss, top-1, top-5, inference time
- History plots (if present in checkpoint)
- Top-10 confused class pairs
- Top-3 confused-pair image samples
- CAM visualization (Score-CAM by default; supports GradCAM/GradCAM++)

## Streamlit App (Inference)

Run:

```bash
streamlit run streamlit_app.py
```

In sidebar:
- `Data root`: default `./stanford_cars`
- `Checkpoints dir`: default `./checkpoints`
- optional GPU toggle

## Notes

- Large data/checkpoints are excluded by `.gitignore`.
