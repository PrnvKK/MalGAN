"""
MalGAN -- Balanced Subset Creator
==================================
Selects a balanced subset of the MaleVis dataset containing only the target
malware families required for this experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
import gc

from load_data import load_malevis_presplit
from config import SELECTED_FAMILIES, OUTPUT_DIR, ensure_dir


def create_balanced_subset(selected_families=None):
    """
    Load the full MaleVis dataset then filter to the requested families.

    Parameters
    ----------
    selected_families : list[str], optional
        Subset of class names to keep.  Defaults to ``SELECTED_FAMILIES``
        from config.

    Returns
    -------
    X_train : np.ndarray  (N, 224, 224, 3)
    y_train : np.ndarray  (N,)  -- remapped 0..k-1
    X_val   : np.ndarray
    y_val   : np.ndarray
    subset_class_names : list[str]
    """
    selected_families = selected_families or SELECTED_FAMILIES

    print("Requesting full dataset ...")
    X_train_full, y_train_full, X_val_full, y_val_full, class_names = \
        load_malevis_presplit()

    print("=" * 60)
    print("CREATING BALANCED SUBSET")
    print("=" * 60)

    selected_train_idx = []
    selected_val_idx = []
    class_mapping = {}
    subset_names = []

    new_idx = 0
    for family in selected_families:
        if family in class_names:
            old_idx = class_names.index(family)
            class_mapping[old_idx] = new_idx

            selected_train_idx.extend(np.where(y_train_full == old_idx)[0])
            selected_val_idx.extend(np.where(y_val_full == old_idx)[0])
            subset_names.append(family)
            new_idx += 1
        else:
            print(f"  WARNING: '{family}' not found in dataset -- skipping")

    X_train = X_train_full[np.array(selected_train_idx)]
    y_train_old = y_train_full[np.array(selected_train_idx)]
    X_val = X_val_full[np.array(selected_val_idx)]
    y_val_old = y_val_full[np.array(selected_val_idx)]

    y_train = np.array([class_mapping[old] for old in y_train_old])
    y_val = np.array([class_mapping[old] for old in y_val_old])

    print(f"Subset training set:   {X_train.shape}")
    print(f"Subset validation set: {X_val.shape}")
    print(f"Classes:               {subset_names}")

    # -- Visualise one sample per class ------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(len(subset_names)):
        class_idx = np.where(y_train == i)[0]
        if len(class_idx) > 0:
            axes[i].imshow(X_train[class_idx[0]])
            axes[i].set_title(subset_names[i], fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    out_dir = ensure_dir(OUTPUT_DIR)
    plot_path = out_dir / "subset_samples.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Sample grid saved to {plot_path}")

    # Free memory
    del X_train_full, y_train_full, X_val_full, y_val_full
    gc.collect()

    return X_train, y_train, X_val, y_val, subset_names


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, subset_class_names = create_balanced_subset()
    print("\nSubset ready for training.")
