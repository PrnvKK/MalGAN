# ============================================
# CELL X: Augmentation Ratios Experiment
# Trains CNN with 0, 25, 50, 100, 200, 500 synth images/class
# Produces comparison table + publication-ready plot
# ============================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
import json
import cv2
from collections import OrderedDict

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import MalGAN.load_balanced_data as load_balanced_data

# ============================================
# CONFIGURATION
# ============================================
PROJECT_DIR = '/content/drive/MyDrive/GAN_Malware_Detection'
GENERATOR_PATH = f'{PROJECT_DIR}/gan_training/checkpoints/generator_final.h5'
LATENT_DIM = 100
CLASS_NAMES = ['Androm', 'Elex', 'Expiro', 'HackKMS', 'Hlux', 'Sality']
NUM_CLASSES = len(CLASS_NAMES)
AUGMENTATION_RATIOS = [0, 25, 50, 100, 200]
MAX_SYNTH = max(AUGMENTATION_RATIOS)
EPOCHS = 50
BATCH_SIZE = 32
RESULTS_OUTPUT_DIR = f'{PROJECT_DIR}/augmentation_ratios'


def build_model(num_classes):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers[:-20]:
        layer.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_generator():
    print(f"Loading generator from {GENERATOR_PATH}...")
    if not os.path.exists(GENERATOR_PATH):
        raise FileNotFoundError(f"Generator not found at {GENERATOR_PATH}")
    return tf.keras.models.load_model(GENERATOR_PATH, compile=False)


def generate_all_synth(generator, num_per_class):
    all_synth = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        images = []
        remaining = num_per_class
        while remaining > 0:
            batch = min(32, remaining)
            noise = tf.random.normal(shape=(batch, LATENT_DIM))
            labels = tf.convert_to_tensor(np.full((batch, 1), class_idx), dtype=tf.int32)
            gen_imgs = generator([noise, labels], training=False).numpy()
            for i in range(batch):
                img_64 = gen_imgs[i]
                img_uint8 = ((img_64 + 1.0) * 127.5).astype(np.uint8)
                img_224 = cv2.resize(img_uint8, (224, 224), interpolation=cv2.INTER_CUBIC)
                images.append(img_224)
            remaining -= batch
        all_synth[class_idx] = np.array(images)
    return all_synth


def subsample_synth(all_synth, num_per_class):
    if num_per_class == 0:
        return np.empty((0, 224, 224, 3), dtype=np.uint8), np.empty((0,), dtype=np.int32)
    X_aug, y_aug = [], []
    for class_idx in range(NUM_CLASSES):
        pool = all_synth[class_idx]
        indices = np.random.choice(len(pool), size=num_per_class, replace=False)
        X_aug.append(pool[indices])
        y_aug.append(np.full(num_per_class, class_idx))
    return np.concatenate(X_aug), np.concatenate(y_aug)


class CompactMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_loss = logs.get('val_loss')
            self.best_epoch = epoch + 1
        print(f"  E{epoch+1:02d}: acc={logs['accuracy']:.3f} v_acc={val_acc:.4f} "
              f"v_loss={logs['val_loss']:.4f} | best={self.best_val_acc:.4f}@{self.best_epoch}")


def train_one_run(X_train_norm, y_train, X_val_norm, y_val, ratio_label, run_dir):
    model = build_model(NUM_CLASSES)
    monitor = CompactMonitor()
    checkpoint_path = os.path.join(run_dir, 'best_model.h5')
    history_path = os.path.join(run_dir, 'history.json')

    callbacks = [
        monitor,
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0),
        EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0)
    ]

    history = model.fit(
        X_train_norm, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_norm, y_val),
        callbacks=callbacks,
        verbose=0
    )

    model.save(checkpoint_path)

    history_dict = {
        'ratio': ratio_label,
        'synthetic_per_class': ratio_label,
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'best_val_accuracy': float(monitor.best_val_acc),
        'best_val_loss': float(monitor.best_val_loss),
        'best_epoch': int(monitor.best_epoch),
        'epochs_completed': len(history.history['loss']),
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    return history, monitor


def plot_results(all_results):
    ratios = [r['ratio'] for r in all_results]
    accs = [r['best_val_acc'] * 100 for r in all_results]
    losses = [r['best_val_loss'] for r in all_results]
    epochs = [r['best_epoch'] for r in all_results]

    # --- Summary table ---
    print("\n" + "=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Synth/class':<15}{'Best Val Acc':<18}{'Best Val Loss':<16}{'Best Epoch':<12}")
    print("-" * 75)
    for r in all_results:
        marker = " <-- baseline" if r['ratio'] == 0 else ""
        print(f"{r['ratio']:<15}{r['best_val_acc']*100:>7.2f}%{'':10}{r['best_val_loss']:<16.4f}{r['best_epoch']:<12}{marker}")

    gain = (all_results[-1]['best_val_acc'] - all_results[0]['best_val_acc']) * 100
    print(f"\nMax gain over baseline: +{gain:.2f}%")

    # --- Publication-ready plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    colors = ['#2c7bb6'] + ['#abd9e9'] * (len(ratios) - 2) + ['#d7191c']
    for i in range(len(ratios)):
        if ratios[i] == 0:
            colors[i] = '#404040'

    # 1. Bar chart: Accuracy vs synthetic images
    bars = axes[0].bar(range(len(ratios)), accs, color=colors, edgecolor='black', linewidth=0.8)
    axes[0].set_xticks(range(len(ratios)))
    axes[0].set_xticklabels([str(r) for r in ratios])
    axes[0].set_xlabel('Synthetic Images per Class', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Accuracy vs Augmentation Ratio', fontsize=13, fontweight='bold')
    axes[0].set_ylim(min(accs) - 1, max(accs) + 0.5)
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                     f'{acc:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # 2. Learning curves overlay
    for i, r in enumerate(all_results):
        label = f'{r["ratio"]} synth/class'
        ls = '-' if i == 0 else ('--' if i == len(all_results) - 1 else '-.')
        axes[1].plot(r['val_accuracy'], linewidth=1.6, linestyle=ls,
                     color=plt.cm.viridis(i / len(all_results)), label=label)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')
    axes[1].set_title('Learning Curves by Augmentation Ratio', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=8, loc='lower right')
    axes[1].grid(alpha=0.3)

    # 3. Improvement over baseline
    baseline_acc = accs[0]
    improvements = [a - baseline_acc for a in accs]
    imp_colors = ['#404040'] + ['#2c7bb6' if x >= 0 else '#d7191c' for x in improvements[1:]]
    bars2 = axes[2].bar(range(len(ratios)), improvements, color=imp_colors, edgecolor='black', linewidth=0.8)
    axes[2].set_xticks(range(len(ratios)))
    axes[2].set_xticklabels([str(r) for r in ratios])
    axes[2].set_xlabel('Synthetic Images per Class', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Improvement over Baseline (%)', fontsize=11, fontweight='bold')
    axes[2].set_title('Gain from GAN Augmentation', fontsize=13, fontweight='bold')
    axes[2].axhline(y=0, color='black', linewidth=0.8)
    for bar, imp in zip(bars2, improvements):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.06 if imp >= 0 else -0.15),
                     f'{imp:+.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_OUTPUT_DIR, 'augmentation_ratios_summary.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved to: {plot_path}")

    # --- Save CSV ---
    csv_path = os.path.join(RESULTS_OUTPUT_DIR, 'augmentation_ratios_results.csv')
    with open(csv_path, 'w') as f:
        f.write("synth_per_class,best_val_accuracy_pct,best_val_loss,best_epoch\n")
        for r in all_results:
            f.write(f"{r['ratio']},{r['best_val_acc']*100:.2f},{r['best_val_loss']:.4f},{r['best_epoch']}\n")
    print(f"CSV saved to: {csv_path}")


def main():
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("AUGMENTATION RATIOS EXPERIMENT")
    print(f"Ratios: {AUGMENTATION_RATIOS} images/class")
    print("=" * 70)

    # Step 1: Load original data
    print("\n[1/4] Loading original balanced dataset...")
    X_train_orig, y_train_orig, X_val, y_val, subset_class_names = \
        load_balanced_data.create_balanced_subset()
    print(f"  Train: {X_train_orig.shape}, Val: {X_val.shape}")
    X_val_norm = X_val.astype('float32') / 255.0

    # Step 2: Load GAN generator
    print("\n[2/4] Loading trained GAN generator...")
    generator = load_generator()

    # Step 3: Generate max synth images (500 per class)
    print(f"\n[3/4] Generating {MAX_SYNTH} synthetic images per class...")
    all_synth = generate_all_synth(generator, MAX_SYNTH)
    for class_idx, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {all_synth[class_idx].shape[0]} images")

    # Step 4: Train for each ratio
    print(f"\n[4/4] Running {len(AUGMENTATION_RATIOS)} training experiments...")

    all_results = []
    run = 0

    for ratio in AUGMENTATION_RATIOS:
        run += 1
        run_dir = os.path.join(RESULTS_OUTPUT_DIR, f'ratio_{ratio:03d}')
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'#' * 70}")
        print(f"RUN {run}/{len(AUGMENTATION_RATIOS)}: {ratio} synth images/class")
        print(f"{'#' * 70}")

        if ratio == 0:
            X_train = X_train_orig.copy()
            y_train = y_train_orig.copy()
        else:
            X_aug, y_aug = subsample_synth(all_synth, ratio)
            X_train = np.concatenate([X_train_orig, X_aug], axis=0)
            y_train = np.concatenate([y_train_orig, y_aug], axis=0)
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]

        print(f"  Combined training set: {X_train.shape[0]} samples")

        X_train_norm = X_train.astype('float32') / 255.0

        t0 = time.time()
        history, monitor = train_one_run(X_train_norm, y_train, X_val_norm, y_val, ratio, run_dir)
        elapsed = time.time() - t0

        result_entry = {
            'ratio': ratio,
            'best_val_acc': monitor.best_val_acc,
            'best_val_loss': monitor.best_val_loss,
            'best_epoch': monitor.best_epoch,
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'train_time_min': elapsed / 60
        }
        all_results.append(result_entry)

        print(f"  Best: acc={monitor.best_val_acc:.4f} loss={monitor.best_val_loss:.4f} "
              f"@ epoch {monitor.best_epoch} | time={elapsed/60:.1f}m")

    # Save aggregated results
    agg_path = os.path.join(RESULTS_OUTPUT_DIR, 'all_results.json')
    with open(agg_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plot and table
    plot_results(all_results)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
