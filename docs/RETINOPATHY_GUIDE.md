# Indian Diabetic Retinopathy CNN: Download, Train, Validate

This guide shows how to download the Kaggle dataset, set up your environment, train a simple CNN, and validate accuracy.

Dataset: https://www.kaggle.com/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset

IMPORTANT: You must be logged in to Kaggle and click “Accept” on the dataset page before the API can download it.

## 1) Environment

Install dependencies (macOS zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow scikit-learn pandas numpy kaggle matplotlib seaborn
```

If you have an Apple Silicon Mac, you can install Apple’s TensorFlow builds for better performance:
- https://developer.apple.com/metal/tensorflow-plugin/

## 2) Kaggle API setup

Create an API token on Kaggle: Account settings → Create New API Token. This downloads `kaggle.json`.

Place it at `~/.kaggle/kaggle.json` and secure it:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Test the Kaggle CLI:

```bash
kaggle --help | head -n 3
```

If you see help output, you’re good.

## 3) Download the dataset

From your project root (`/Applications/CODES/AiHC` in this workspace):

```bash
mkdir -p data
cd data
kaggle datasets download -d aaryapatel98/indian-diabetic-retinopathy-image-dataset -p .
unzip -q indian-diabetic-retinopathy-image-dataset.zip -d retinopathy
cd ..
```

After unzip, your structure should look like one of the following:

- Flat (classes directly under `retinopathy/`)
- Split subfolders (e.g., `retinopathy/train`, `retinopathy/val` or `validation`, optional `retinopathy/test`)

The training script will detect either layout.

## 4) Train a simple CNN

Run (adjust parameters as needed):

```bash
source .venv/bin/activate
python3 train_retino_cnn.py --data-dir data/retinopathy --epochs 5 --img-size 224 --batch-size 32 --limit-per-class 500
```

Notes:
- `--limit-per-class` lets you do a quick smoke test by sampling up to N images per class.
- The script auto-splits validation if your dataset is in a flat layout; if already split, it will use your split.

Outputs will be saved to `outputs_retino/`:
- `model.keras`: trained model
- `metrics.json`: validation/test accuracy, classification report, confusion matrix

## 5) Validate results

Open `outputs_retino/metrics.json`. You’ll see fields like:

```json
{
  "val": {
    "accuracy": 0.82,
    "report": "... classification report ...",
    "confusion_matrix": [[...]]
  }
}
```

If your dataset has a test split, you’ll also see a `test` section.

## Troubleshooting
- Kaggle 403/404: Make sure you clicked “Accept” on the dataset page and your `kaggle.json` is correct.
- ImportError (TensorFlow): ensure your venv is active and `pip install tensorflow` completed successfully.
- OOM on CPU/GPU: reduce `--batch-size`, lower `--img-size` (e.g., 160), or increase `--limit-per-class` to a smaller number.
