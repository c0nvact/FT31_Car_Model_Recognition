# Car Model Recognition Final Project

This project trains and evaluates car model classifiers and provides a Streamlit app for inference with saved checkpoints.

## Project Files

- `streamlit_app.py`: Web app for image upload/URL inference across multiple model checkpoints
- `cars_dataset.py`: Dataset and crop utilities
- `data_utils.py`: Dataset loading and preprocessing helpers
- `train_utils.py`: Training/evaluation helpers
- `checkpoints/`: Saved model checkpoints (`*_best.pth`) (ignored by git)
- `stanford_cars/`: Stanford Cars dataset files (ignored by git)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run streamlit_app.py
```

In the sidebar, set:

- `Data root` to your dataset folder (default: `./stanford_cars`)
- `Checkpoints dir` to your model checkpoints folder (default: `./checkpoints`)

## Notes

- Large datasets and checkpoint files are intentionally excluded from git via `.gitignore`.
- If you want to version model weights, use Git LFS.
