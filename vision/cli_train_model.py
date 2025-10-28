#!/usr/bin/env python3
"""
Simple CNN training and validation on the Indian Diabetic Retinopathy Image Dataset from Kaggle.

Usage (after downloading and extracting the dataset into a folder):

  python3 train_retino_cnn.py \
    --data-dir data/retinopathy \
    --epochs 5 \
    --img-size 224 \
    --batch-size 32 \
    --limit-per-class 500

If you haven't downloaded the dataset yet, see README_RETINO.md for instructions.
"""
import argparse
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np

# Optional imports gated to provide clearer error messages
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    print("TensorFlow is required. Install with: pip install tensorflow", file=sys.stderr)
    raise

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
except Exception:
    print("scikit-learn is required. Install with: pip install scikit-learn", file=sys.stderr)
    raise


def detect_layout(data_dir: Path) -> Dict[str, Optional[Path]]:
    """Detect common dataset layouts and return standardized paths.
    Returns dict with keys: mode ('flat' or 'split'), 'train_dir', 'val_dir', 'test_dir'.
    - flat: expects class subfolders directly under data_dir; we'll do a split.
    - split: expects subfolders like train/, val/ (or validation/), and optional test/.
    """
    result = {"mode": "flat", "train_dir": None, "val_dir": None, "test_dir": None}
    if not data_dir.exists():
        return result

    # Look for typical split folders
    candidates = {
        "train": None,
        "val": None,
        "validation": None,
        "test": None,
    }
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if name in candidates:
            candidates[name] = child

    train_dir = candidates.get("train")
    val_dir = candidates.get("val") or candidates.get("validation")
    test_dir = candidates.get("test")
    if train_dir is not None and (val_dir is not None or test_dir is not None):
        result.update({
            "mode": "split",
            "train_dir": train_dir,
            "val_dir": val_dir,
            "test_dir": test_dir,
        })
        return result

    # Fallback: treat as flat directory containing class subfolders
    result.update({
        "mode": "flat",
        "train_dir": data_dir,
        "val_dir": None,
        "test_dir": None,
    })
    return result


def build_datasets(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    val_split: float = 0.2,
    seed: int = 42,
    limit_per_class: Optional[int] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset], int]:
    """Create tf.data.Dataset objects from a directory.

    Supports two layouts:
    - flat (class subfolders directly under data_dir) -> we split into train/val
    - split (train/val/test subfolders)

    Returns: (train_ds, val_ds, test_ds_or_None, num_classes)
    """
    layout = detect_layout(data_dir)
    image_size = (img_size, img_size)

    def limit_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
        if limit_per_class is None:
            return ds
        # Balance by taking up to N per class
        def class_filter(x, y):
            return tf.cast(y, tf.int64)
        # Group by label is not trivial in tf.data; simpler approach: use take on a rebalanced dataset
        # We'll materialize a small subset by iterating once (works for small limits)
        if limit_per_class <= 0:
            return ds
        by_class = {}
        limited_elems = []
        for x, y in ds.unbatch().as_numpy_iterator():
            cls = int(y)
            cnt = by_class.get(cls, 0)
            if cnt < limit_per_class:
                limited_elems.append((x, y))
                by_class[cls] = cnt + 1
        # Recreate dataset
        xs = np.stack([e[0] for e in limited_elems], axis=0)
        ys = np.stack([e[1] for e in limited_elems], axis=0)
        new_ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        return new_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if layout["mode"] == "split":
        train_ds = tf.keras.utils.image_dataset_from_directory(
            layout["train_dir"],
            labels="inferred",
            label_mode="int",
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
        )
        val_ds = None
        if layout["val_dir"] is not None:
            val_ds = tf.keras.utils.image_dataset_from_directory(
                layout["val_dir"],
                labels="inferred",
                label_mode="int",
                image_size=image_size,
                batch_size=batch_size,
                seed=seed,
            )
        test_ds = None
        if layout["test_dir"] is not None:
            test_ds = tf.keras.utils.image_dataset_from_directory(
                layout["test_dir"],
                labels="inferred",
                label_mode="int",
                image_size=image_size,
                batch_size=batch_size,
                seed=seed,
            )
        class_names = train_ds.class_names
        if limit_per_class is not None:
            train_ds = limit_dataset(train_ds)
            if val_ds is not None:
                val_ds = limit_dataset(val_ds)
        return train_ds, val_ds, test_ds, len(class_names)

    # flat layout: split into train/val
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    class_names = train_ds.class_names
    if limit_per_class is not None:
        train_ds = limit_dataset(train_ds)
        val_ds = limit_dataset(val_ds)
    return train_ds, val_ds, None, len(class_names)


def build_model(img_size: int, num_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def evaluate_model(model: tf.keras.Model, ds: tf.data.Dataset, class_names) -> dict:
    y_true = []
    y_pred = []
    for batch_x, batch_y in ds:
        preds = model.predict(batch_x, verbose=0)
        y_true.extend(batch_y.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=[str(c) for c in class_names], digits=4)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": float(acc), "report": report, "confusion_matrix": cm}


def auto_download_dataset(dataset_slug: str, target_dir: Path) -> None:
    """Attempt to auto-download a Kaggle dataset zip and extract to target_dir.

    Requirements:
    - Kaggle CLI installed (`pip install kaggle`)
    - Kaggle credentials set up (~/.kaggle/kaggle.json with correct permissions)
    - User has accepted the dataset on Kaggle

    This function will:
    - create parent folders as needed
    - download the dataset zip into target_dir.parent (e.g., data/)
    - extract into target_dir (e.g., data/retinopathy)
    """
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # Check kaggle cli availability
    try:
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as ex:
        print("Kaggle CLI not found. Install with: pip install kaggle", file=sys.stderr)
        raise

    zip_name = dataset_slug.split("/")[-1] + ".zip"
    zip_path = target_dir.parent / zip_name

    if not zip_path.exists():
        print(f"Downloading Kaggle dataset '{dataset_slug}' to {zip_path} ...")
        try:
            subprocess.run([
                "kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(target_dir.parent), "-o"
            ], check=True)
        except subprocess.CalledProcessError as ex:
            print("Failed to download dataset via Kaggle CLI.", file=sys.stderr)
            print("Ensure you've accepted the dataset and Kaggle credentials are configured.", file=sys.stderr)
            raise

    # Extract
    print(f"Extracting {zip_path} to {target_dir} ...")
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
    except Exception as ex:
        print(f"Failed to extract {zip_path}: {ex}", file=sys.stderr)
        raise

    # Optional: try to flatten single nested directory if present
    try:
        entries = list(target_dir.iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            nested = entries[0]
            for p in nested.iterdir():
                shutil.move(str(p), str(target_dir))
            nested.rmdir()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset root directory. Either flat class-subfolder layout or split with train/val(/test).')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--limit-per-class', type=int, default=None,
                        help='If set, limits samples per class for quicker training (use for smoke tests).')
    parser.add_argument('--out-dir', type=str, default='outputs_retino',
                        help='Directory to save model and metrics.')
    parser.add_argument('--auto-download', action='store_true',
                        help='If set and data-dir does not exist, attempt Kaggle auto-download.')
    parser.add_argument('--kaggle-dataset', type=str,
                        default='aaryapatel98/indian-diabetic-retinopathy-image-dataset',
                        help='Kaggle dataset slug for auto-download.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        if args.auto_download:
            try:
                auto_download_dataset(args.kaggle_dataset, data_dir)
            except Exception:
                print(f"Auto-download failed. Please download manually. See README_RETINO.md.", file=sys.stderr)
                sys.exit(2)
        else:
            print(f"Dataset directory '{data_dir}' not found. Please download and extract the Kaggle dataset.\n"
                  f"Or rerun with --auto-download. See README_RETINO.md for step-by-step instructions.", file=sys.stderr)
            sys.exit(2)

    print("Detecting dataset layout and building datasets...")
    train_ds, val_ds, test_ds, num_classes = build_datasets(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=0.2,
        seed=42,
        limit_per_class=args.limit_per_class,
    )

    class_names = getattr(train_ds, 'class_names', None)
    if class_names is None:
        # image_dataset_from_directory always has class_names; fallback safeguard
        class_names = [str(i) for i in range(num_classes)]

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    if val_ds is not None:
        val_ds = val_ds.prefetch(AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.prefetch(AUTOTUNE)

    print(f"Classes: {class_names} (num={num_classes})")

    print("Building model...")
    model = build_model(args.img_size, num_classes)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    ]

    print("Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    metrics = {}
    print("Evaluating on validation set...")
    if val_ds is not None:
        metrics['val'] = evaluate_model(model, val_ds, class_names)
        print("Validation accuracy:", metrics['val']['accuracy'])
        print(metrics['val']['report'])

    if test_ds is not None:
        print("Evaluating on test set...")
        metrics['test'] = evaluate_model(model, test_ds, class_names)
        print("Test accuracy:", metrics['test']['accuracy'])
        print(metrics['test']['report'])

    # Save artifacts
    model_path = out_dir / 'model.keras'
    print(f"Saving model to {model_path}")
    model.save(model_path)

    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Save class names for explainability notebooks
    try:
        with open(out_dir / 'class_names.json', 'w') as f:
            json.dump({"class_names": list(class_names)}, f, indent=2)
        print(f"Saved class names to {out_dir / 'class_names.json'}")
    except Exception as ex:
        print(f"Warning: failed to save class names: {ex}")

    print("Done.")


if __name__ == "__main__":
    main()
