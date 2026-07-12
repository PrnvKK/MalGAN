import os
import numpy as np
import matplotlib.pyplot as plt
import gc
import MalGAN.load_data as load_data  # Import the first script

def create_balanced_subset():
    # 1. Fetch data from the load_data module
    print("Requesting full dataset from load_data...")
    X_train_full, y_train_full, X_val_full, y_val_full, class_names = load_data.get_full_dataset()
    
    PROJECT_DIR = '/content/drive/MyDrive/GAN_Malware_Detection'
    os.makedirs(PROJECT_DIR, exist_ok=True)
    
    print("="*60)
    print("CREATING SUBSET FROM FULL DATASET")
    print("="*60 + "\n")

    SELECTED_FAMILIES = ['Androm', 'Elex', 'Expiro', 'HackKMS', 'Hlux', 'Sality']

    selected_train_indices = []
    selected_val_indices = []
    selected_class_mapping = {}
    subset_class_names = []

    new_class_idx = 0
    for family_name in SELECTED_FAMILIES:
        if family_name in class_names:
            old_idx = class_names.index(family_name)
            selected_class_mapping[old_idx] = new_class_idx

            train_mask = (y_train_full == old_idx)
            selected_train_indices.extend(np.where(train_mask)[0])

            val_mask = (y_val_full == old_idx)
            selected_val_indices.extend(np.where(val_mask)[0])

            subset_class_names.append(family_name)
            new_class_idx += 1
        else:
            print(f"  WARNING: {family_name} not found!")

    # Create subset arrays
    X_train = X_train_full[np.array(selected_train_indices)]
    y_train_old = y_train_full[np.array(selected_train_indices)]
    X_val = X_val_full[np.array(selected_val_indices)]
    y_val_old = y_val_full[np.array(selected_val_indices)]

    # Remap labels
    y_train = np.array([selected_class_mapping[old_label] for old_label in y_train_old])
    y_val = np.array([selected_class_mapping[old_label] for old_label in y_val_old])

    print(f"Subset Training set: {X_train.shape}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(len(subset_class_names)):
        class_indices = np.where(y_train == i)[0]
        if len(class_indices) > 0:
            sample_idx = class_indices[0]
            axes[i].imshow(X_train[sample_idx])
            axes[i].set_title(f"{subset_class_names[i]}", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{PROJECT_DIR}/subset_samples.png', dpi=150)
    plt.show()

    # Clean up memory
    del X_train_full, y_train_full, X_val_full, y_val_full
    gc.collect()
    
    return X_train, y_train, X_val, y_val, subset_class_names

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, subset_class_names = create_balanced_subset()
    print("\n✓ Subset ready for training!")