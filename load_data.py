"""
MalGAN -- Dataset Loader
=========================
Reads the MaleVis malware image dataset (pre-split into train/val folders)
and returns NumPy arrays of images and integer labels.
"""

import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import sys

from config import MALEVIS_DATA_DIR, CNN_IMG_SIZE


def load_malevis_presplit(base_path=None, img_size=None):
    """
    Load the MaleVis dataset from a pre-split directory structure.

    Expected layout::

        base_path/
            train/
                Androm/   (*.png, *.jpg)
                Elex/
                ...
            val/
                Androm/
                ...

    Parameters
    ----------
    base_path : str or Path, optional
        Root directory containing ``train/`` and ``val/`` subfolders.
        Defaults to ``MALEVIS_DATA_DIR`` from config.
    img_size : tuple, optional
        Target ``(height, width)`` for loaded images.
        Defaults to ``CNN_IMG_SIZE`` from config.

    Returns
    -------
    X_train : np.ndarray  (N, H, W, 3)
    y_train : np.ndarray  (N,)
    X_val   : np.ndarray
    y_val   : np.ndarray
    class_names : list[str]
    """
    base_path = Path(base_path or MALEVIS_DATA_DIR)
    img_size = img_size or CNN_IMG_SIZE

    train_dir = base_path / "train"
    val_dir = base_path / "val"

    if not train_dir.exists():
        possible = list(base_path.rglob("train"))
        if possible:
            train_dir = possible[0]
            val_dir = train_dir.parent / "val"
        else:
            raise FileNotFoundError(
                f"Train directory not found under {base_path}. "
                "Set MALEVIS_DATA_DIR env variable to the correct path."
            )

    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    def _load_split(split_dir):
        X_list, y_list = [], []
        split_class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for idx, class_dir in enumerate(split_class_dirs):
            image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            for img_path in tqdm(image_files, desc=f"  {class_dir.name}", leave=False):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    X_list.append(img)
                    y_list.append(idx)
                except Exception:
                    continue
        return np.array(X_list), np.array(y_list)

    print(f"Loading training data from {train_dir} ...")
    X_train, y_train = _load_split(train_dir)
    print(f"Loading validation data from {val_dir} ...")
    X_val, y_val = _load_split(val_dir)

    return X_train, y_train, X_val, y_val, class_names


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("MALGAN -- DATASET LOADER TEST")
    print("=" * 60)

    X_train_full, y_train_full, X_val_full, y_val_full, class_names = \
        load_malevis_presplit()

    print(f"\nTraining set:   {X_train_full.shape}")
    print(f"Validation set: {X_val_full.shape}")
    print(f"Classes:        {len(class_names)}")
