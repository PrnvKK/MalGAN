# ============================================
# CELL 8: OPTIMIZED GAN Training Loop - All Fixes Applied
# ============================================

# ============================================
# IMPORTS & GLOBAL SCOPE SETUP
# ============================================
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import cv2


LATENT_DIM = 100
NUM_CLASSES = 6
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
PROJECT_DIR = '/content/drive/MyDrive/GAN_Malware_Detection'


# --- IMPORT PROJECT LOGIC ---
import GAN_Files.load_balanced_data as load_balanced_data
import GAN_Files.cnn_arch as cnn_arch

# --- ADD THESE IMPORTS ---
import GAN_Files.gan_arch as gan_arch # Import your architecture script
from GAN_Files.load_balanced_data import create_balanced_subset # Import your data loader

from GAN_Files.gan_arch import build_generator, build_discriminator

# --- INITIALIZE VARIABLES ---
# This pulls X_train and subset_class_names into this script's memory
X_train, y_train, X_val, y_val, subset_class_names = load_balanced_data.create_balanced_subset()

# Re-establish constants
PROJECT_DIR = '/content/drive/MyDrive/GAN_Malware_Detection'
LATENT_DIM = 100
NUM_CLASSES = len(subset_class_names)
IMG_SHAPE = (64, 64, 3)


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from collections import deque
from IPython.display import clear_output
import os

print("="*80)
print("OPTIMIZED GAN TRAINING LOOP - PRODUCTION READY")
print("="*80)

# ============================================
# TRAINING CONFIGURATION
# ============================================
GAN_CONFIG = {
    'epochs': 100,
    'batch_size': 64,
    'save_interval': 5,
    'checkpoint_interval': 10,
    'label_smoothing': True,
    'smooth_real': 0.9,
    'smooth_fake': 0.1,
    'd_target_acc_min': 0.65,
    'd_target_acc_max': 0.80,
    'running_avg_window': 10,  # For batch-level monitoring
    'initial_lr': 0.0002,
    'lr_decay_epochs': 200,  # Start decay after this epoch
    'min_lr': 0.00002,
}

print(f"\nTraining Configuration:")
for key, value in GAN_CONFIG.items():
    print(f"  {key}: {value}")

# ============================================
# REBUILD MODELS WITH GRADIENT CLIPPING
# ============================================
print("\n" + "="*80)
print("REBUILDING MODELS WITH GRADIENT CLIPPING")
print("="*80)

# Rebuild generator and discriminator
generator = build_generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)
print("✓ Generator rebuilt")

discriminator = build_discriminator(img_shape=IMG_SHAPE, num_classes=NUM_CLASSES)
print("✓ Discriminator rebuilt")

discriminator.compile(
    optimizer=Adam(learning_rate=GAN_CONFIG['initial_lr'], beta_1=0.5, clipnorm=5.0),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("✓ Discriminator compiled with gradient clipping (clipnorm=5.0)")

discriminator.trainable = False
noise_input = layers.Input(shape=(LATENT_DIM,), name='gan_noise_input')
label_input = layers.Input(shape=(1,), dtype='int32', name='gan_label_input')
generated_image = generator([noise_input, label_input])
validity = discriminator([generated_image, label_input])

gan = models.Model([noise_input, label_input], validity, name='GAN')
gan.compile(
    optimizer=Adam(learning_rate=GAN_CONFIG['initial_lr'], beta_1=0.5, clipnorm=5.0),
    loss='binary_crossentropy'
)
discriminator.trainable = True
print("✓ GAN compiled with gradient clipping (clipnorm=5.0)")

# ============================================
# HELPER FUNCTIONS - OPTIMIZED
# ============================================
def normalize_images_fn(images):
    """Normalize images to [-1, 1] - for tf.data pipeline"""
    images = tf.cast(images, tf.float32)
    # FIX: Convert [0, 255] → [-1, 1]
    return (images / 127.5) - 1.0

def denormalize_images(images):
    """Convert from [-1, 1] to [0, 1]"""
    return (images + 1.0) / 2.0

def get_real_labels(batch_size, smooth=True):
    """Generate labels for real images"""
    if smooth:
        return np.ones((batch_size, 1), dtype=np.float32) * GAN_CONFIG['smooth_real']
    return np.ones((batch_size, 1), dtype=np.float32)

def get_fake_labels(batch_size, smooth=True):
    """Generate labels for fake images"""
    if smooth:
        return np.ones((batch_size, 1), dtype=np.float32) * GAN_CONFIG['smooth_fake']
    return np.zeros((batch_size, 1), dtype=np.float32)

def generate_noise(batch_size, latent_dim=100, return_tensor=True):
    """Generate random noise vectors"""
    noise = np.random.normal(0, 1, size=(batch_size, latent_dim)).astype(np.float32)
    if return_tensor:
        return tf.convert_to_tensor(noise, dtype=tf.float32)
    return noise

def generate_class_labels(batch_size, num_classes=6, return_tensor=True):
    """Generate random class labels"""
    labels = np.random.randint(0, num_classes, size=(batch_size, 1))
    if return_tensor:
        return tf.convert_to_tensor(labels, dtype=tf.int32)
    return labels

def check_for_nan(*losses):
    """Check if any loss is NaN"""
    for loss in losses:
        if isinstance(loss, (list, tuple)):
            if any(np.isnan(x) for x in loss):
                return True
        elif np.isnan(loss):
            return True
    return False

def save_generated_images(generator, epoch, latent_dim, num_classes,
                          class_names, save_dir):
    """Generate and save sample images - OPTIMIZED"""
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    # Generate all samples at once (more efficient)
    noise = generate_noise(num_classes, latent_dim, return_tensor=True)

    # FIX: Also convert labels to tensor
    labels = tf.convert_to_tensor(np.arange(num_classes).reshape(-1, 1), dtype=tf.int32)

    # Direct call instead of predict
    gen_imgs = generator([noise, labels], training=False)

    for i in range(num_classes):
        gen_img = gen_imgs[i].numpy()
        gen_img = denormalize_images(gen_img)
        gen_img = np.clip(gen_img, 0, 1)

        axes[i].imshow(gen_img)
        axes[i].set_title(class_names[i], fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f'Generated Images - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = f'{save_dir}/generated_epoch_{epoch:04d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def update_learning_rate(epoch, optimizer, config):
    """Update learning rate with decay"""
    if epoch >= config['lr_decay_epochs']:
        # Linear decay
        decay_epochs = config['epochs'] - config['lr_decay_epochs']
        decay_per_epoch = (config['initial_lr'] - config['min_lr']) / decay_epochs
        new_lr = max(config['min_lr'], config['initial_lr'] - decay_per_epoch * (epoch - config['lr_decay_epochs']))
        optimizer.learning_rate.assign(new_lr)
        return new_lr
    return config['initial_lr']

# ============================================
# CREATE TF.DATA PIPELINE - MEMORY EFFICIENT
# ============================================
print("\n" + "="*80)
print("CREATING OPTIMIZED DATA PIPELINE")
print("="*80)

# Create dataset from numpy arrays
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# Normalize on-the-fly and prepare batches
train_dataset = train_dataset.map(
    lambda x, y: (normalize_images_fn(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(GAN_CONFIG['batch_size'])
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

print(f"✓ Data pipeline created with on-the-fly normalization")
print(f"✓ Memory-efficient: No full dataset copy created")
print(f"✓ Batch size: {GAN_CONFIG['batch_size']}")

# ============================================
# CREATE SAVE DIRECTORIES
# ============================================
GAN_SAVE_DIR = f"{PROJECT_DIR}/gan_training"
GAN_SAMPLES_DIR = f"{GAN_SAVE_DIR}/samples"
GAN_CHECKPOINTS_DIR = f"{GAN_SAVE_DIR}/checkpoints"

os.makedirs(GAN_SAMPLES_DIR, exist_ok=True)
os.makedirs(GAN_CHECKPOINTS_DIR, exist_ok=True)

# ============================================
# TRAINING LOOP - OPTIMIZED
# ============================================
print("\n" + "="*80)
print("STARTING OPTIMIZED GAN TRAINING")
print("="*80)

# Training history
history = {
    'd_loss': [],
    'd_acc': [],
    'g_loss': [],
    'epoch_times': [],
    'learning_rates': [],
}

# Training variables
batch_size = GAN_CONFIG['batch_size']
epochs = GAN_CONFIG['epochs']
total_start_time = time.time()

# Adaptive training variables
d_train_ratio = 1
g_train_ratio = 1
d_acc_running_avg = deque(maxlen=GAN_CONFIG['running_avg_window'])

# Training success flag
training_failed = False


# ============================================
# FIX: RESIZE DATA & REBUILD PIPELINE
# ============================================
import cv2
import tensorflow as tf
import numpy as np

print("="*80)
print("RESIZING DATASET TO 64x64")
print("="*80)

# 1. Resize X_train to 64x64
# Check if X_train exists and is 224x224
if X_train.shape[1] == 224:
    print(f"Original X_train shape: {X_train.shape} (224x224 detected)")

    X_train_64 = []
    for img in X_train:
        # Resize to 64x64
        # cv2.resize automatically handles channel dimension
        resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        X_train_64.append(resized)

    X_train = np.array(X_train_64)
    print(f"✓ Resized X_train shape: {X_train.shape}")
else:
    print(f"X_train is already {X_train.shape} (Skipping resize)")

# Update global constant just in case
IMG_SHAPE = (64, 64, 3)

# 2. RECREATE THE PIPELINE (Crucial Step!)
# If you don't run this, the dataset will still point to the old 224x224 memory!
print("\nRebuilding tf.data Pipeline...")

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(
    lambda x, y: (normalize_images_fn(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(64) # Ensure batch size is 64 here too
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

print("✓ Pipeline ready for 64x64 training")

print(f"\nTraining Configuration:")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Initial learning rate: {GAN_CONFIG['initial_lr']}")
print(f"  Gradient clipping: clipnorm=1.0")
print(f"  Label smoothing: {GAN_CONFIG['smooth_real']}/{GAN_CONFIG['smooth_fake']}")

print("\n" + "="*80)
print("EPOCH-BY-EPOCH TRAINING")
print("="*80)

for epoch in range(epochs):

    print("="*80)
    print("DATA RANGE VERIFICATION")
    print("="*80)
    print(f"X_train dtype: {X_train.dtype}")
    print(f"X_train min value: {X_train.min()}")
    print(f"X_train max value: {X_train.max()}")
    print(f"X_train shape: {X_train.shape}")

    sample_batch = next(iter(train_dataset.take(1)))[0]
    print(f"Real images (after pipeline): [{sample_batch.numpy().min():.2f}, {sample_batch.numpy().max():.2f}]")

    # FIX: Convert labels to tensor too
    test_noise = generate_noise(4, LATENT_DIM)  # Already returns tensor
    test_labels = tf.constant([[0],[1],[2],[3]], dtype=tf.int32)  # Make it a tensor
    test_fake = generator([test_noise, test_labels], training=False)
    print(f"Fake images (generator): [{test_fake.numpy().min():.2f}, {test_fake.numpy().max():.2f}]")
    print("✓ Both should be in [-1, 1]!")

    print("="*80)

    if training_failed:
        print("\n⛔ Training stopped due to failure")
        break

    epoch_start_time = time.time()

    # Update learning rates with decay
    current_lr_d = update_learning_rate(epoch, discriminator.optimizer, GAN_CONFIG)
    current_lr_g = update_learning_rate(epoch, gan.optimizer, GAN_CONFIG)

    # Batch training
    d_losses = []
    d_accs = []
    g_losses = []

    batch_idx = 0
    for real_images, real_labels in train_dataset:
        batch_size_actual = real_images.shape[0]
        real_labels = tf.reshape(real_labels, (-1, 1))

        # ============================================
        # TRAIN DISCRIMINATOR - COMBINED UPDATE
        # ============================================
        for _ in range(d_train_ratio):
          # Generate fake images
          noise = generate_noise(batch_size_actual, LATENT_DIM)
          fake_labels_class = generate_class_labels(batch_size_actual, NUM_CLASSES)

          # CRITICAL FIX: Ensure both are int32 (or both int64)
          real_labels = tf.cast(real_labels, tf.int32)  # Cast real_labels to match fake_labels
          fake_labels_class = tf.cast(fake_labels_class, tf.int32)

          # Generate fake images
          fake_images = generator([noise, fake_labels_class], training=True)

          # Combine real and fake - now both are tensors with matching dtypes
          combined_images = tf.concat([real_images, fake_images], axis=0)
          combined_labels_class = tf.concat([real_labels, fake_labels_class], axis=0)

          # Create real/fake target labels
          batch_size_combined = batch_size_actual * 2
          real_y = get_real_labels(batch_size_actual, smooth=GAN_CONFIG['label_smoothing'])
          fake_y = get_fake_labels(batch_size_actual, smooth=GAN_CONFIG['label_smoothing'])
          combined_y_realfake = np.vstack([real_y, fake_y])

          # Train discriminator
          d_metrics = discriminator.train_on_batch(
              [combined_images, combined_labels_class],
              combined_y_realfake
          )

          d_loss = d_metrics[0]

          # FIXED ACCURACY CALCULATION
          # The accuracy metric is misleading with label smoothing
          # Calculate actual binary accuracy manually
          d_predictions = discriminator([combined_images, combined_labels_class], training=False)

          # Count correct predictions: first half should be >0.5 (real), second half <0.5 (fake)
          real_correct = tf.reduce_sum(tf.cast(d_predictions[:batch_size_actual] > 0.5, tf.float32))
          fake_correct = tf.reduce_sum(tf.cast(d_predictions[batch_size_actual:] < 0.5, tf.float32))
          d_acc = float((real_correct + fake_correct) / batch_size_combined)

          # Check for NaN
          if check_for_nan(d_loss, d_acc):
              print(f"\n❌ NaN detected in Discriminator at Epoch {epoch+1}, Batch {batch_idx}")
              print(f"   D Loss: {d_loss}, D Acc: {d_acc}")
              training_failed = True
              break

        # ============================================
        # TRAIN GENERATOR
        # ============================================
        for _ in range(g_train_ratio):
          noise = generate_noise(batch_size_actual, LATENT_DIM)
          gen_labels = generate_class_labels(batch_size_actual, NUM_CLASSES)

          # CRITICAL FIX: Convert to tensor
          gen_labels = tf.convert_to_tensor(gen_labels, dtype=tf.int32)

          # Generator wants discriminator to output "real"
          valid_y = get_real_labels(batch_size_actual, smooth=True)

          # Train generator
          g_loss = gan.train_on_batch([noise, gen_labels], valid_y)

          # Check for NaN
          if check_for_nan(g_loss):
              print(f"\n❌ NaN detected in Generator at Epoch {epoch+1}, Batch {batch_idx}")
              print(f"   G Loss: {g_loss}")
              training_failed = True
              break

        # Store metrics
        d_losses.append(d_loss)
        d_accs.append(d_acc)
        g_losses.append(g_loss)
        d_acc_running_avg.append(d_acc)

        # ============================================
        # BATCH-LEVEL ADAPTIVE STRATEGY
        # ============================================
        if len(d_acc_running_avg) >= GAN_CONFIG['running_avg_window']:
            avg_d_acc = np.mean(d_acc_running_avg)

            if avg_d_acc < GAN_CONFIG['d_target_acc_min']:
                d_train_ratio = 2
                g_train_ratio = 1
            elif avg_d_acc > GAN_CONFIG['d_target_acc_max']:
                d_train_ratio = 1
                g_train_ratio = 2
            else:
                d_train_ratio = 1
                g_train_ratio = 1

        batch_idx += 1

    if training_failed:
        break

    # ============================================
    # EPOCH SUMMARY
    # ============================================
    epoch_time = time.time() - epoch_start_time
    avg_d_loss = np.mean(d_losses)
    avg_d_acc = np.mean(d_accs)
    avg_g_loss = np.mean(g_losses)

    # Store in history
    history['d_loss'].append(float(avg_d_loss))
    history['d_acc'].append(float(avg_d_acc))
    history['g_loss'].append(float(avg_g_loss))
    history['epoch_times'].append(float(epoch_time))
    history['learning_rates'].append(float(current_lr_d))

    # Determine balance status
    if avg_d_acc < GAN_CONFIG['d_target_acc_min']:
        balance_status = "⚠️ D weak → Training D more"
    elif avg_d_acc > GAN_CONFIG['d_target_acc_max']:
        balance_status = "⚠️ D strong → Training G more"
    else:
        balance_status = "✓ Balanced"

    # ============================================
    # PROGRESS DISPLAY
    # ============================================
    elapsed_time = time.time() - total_start_time
    progress = (epoch + 1) / epochs
    bar_length = 50
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)

    print(f"\n{'─'*80}")
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"{'─'*80}")
    print(f"Progress: [{bar}] {progress*100:.1f}%")
    print(f"")
    print(f"Discriminator → Loss: {avg_d_loss:.4f} | Accuracy: {avg_d_acc:.4f}")
    print(f"Generator     → Loss: {avg_g_loss:.4f}")
    print(f"")
    print(f"Learning Rate: {current_lr_d:.6f}")
    print(f"Epoch Time: {epoch_time:.1f}s | Total Time: {elapsed_time/60:.1f}m")
    print(f"{balance_status}")

    # ============================================
    # EARLY WARNING CHECKS
    # ============================================
    if epoch > 10:
        # Check for mode collapse
        recent_g_losses = history['g_loss'][-5:]
        if np.mean(recent_g_losses) < 0.01:
            print("⚠️ WARNING: Possible mode collapse (G loss → 0)")

        # Check for discriminator collapse
        recent_d_accs = history['d_acc'][-5:]
        if 0.48 < np.mean(recent_d_accs) < 0.52:
            print("⚠️ WARNING: Discriminator at random guessing (50%)")

        # Check for discriminator domination
        if avg_d_acc > 0.95:
            print("⚠️ WARNING: Discriminator too strong (>95% accuracy)")

    # ============================================
    # SAVE SAMPLES
    # ============================================
    if (epoch + 1) % GAN_CONFIG['save_interval'] == 0 or epoch == 0:
        print(f"📸 Samples saved")
        save_generated_images(
            generator,
            epoch + 1,
            LATENT_DIM,
            NUM_CLASSES,
            subset_class_names,
            GAN_SAMPLES_DIR
        )

    # ============================================
    # SAVE CHECKPOINTS
    # ============================================
    if (epoch + 1) % GAN_CONFIG['checkpoint_interval'] == 0:
        generator.save(f"{GAN_CHECKPOINTS_DIR}/generator_epoch_{epoch+1:04d}.h5")
        discriminator.save(f"{GAN_CHECKPOINTS_DIR}/discriminator_epoch_{epoch+1:04d}.h5")
        print(f"💾 Checkpoint saved")

# ============================================
# TRAINING COMPLETE
# ============================================
total_time = time.time() - total_start_time

print("\n" + "="*80)
if training_failed:
    print("⚠️ TRAINING STOPPED DUE TO INSTABILITY")
else:
    print("✅ GAN TRAINING COMPLETE!")
print("="*80)
print(f"Total training time: {total_time/3600:.2f} hours")
print(f"Average time per epoch: {np.mean(history['epoch_times']):.1f}s")

if len(history['d_loss']) > 0:
    print(f"Final D loss: {history['d_loss'][-1]:.4f}")
    print(f"Final D accuracy: {history['d_acc'][-1]:.4f}")
    print(f"Final G loss: {history['g_loss'][-1]:.4f}")

# ============================================
# SAVE FINAL MODELS
# ============================================
print("\n" + "="*80)
print("SAVING FINAL MODELS")
print("="*80)

generator.save(f"{GAN_CHECKPOINTS_DIR}/generator_final.h5")
discriminator.save(f"{GAN_CHECKPOINTS_DIR}/discriminator_final.h5")
gan.save(f"{GAN_CHECKPOINTS_DIR}/gan_final.h5")

print(f"✓ Models saved to {GAN_CHECKPOINTS_DIR}")

# ============================================
# SAVE TRAINING HISTORY
# ============================================
history_path = f"{GAN_SAVE_DIR}/training_history.json"
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

print(f"✓ Training history saved: {history_path}")

# ============================================
# PLOT TRAINING CURVES
# ============================================
print("\n" + "="*80)
print("GENERATING TRAINING VISUALIZATIONS")
print("="*80)

if len(history['d_loss']) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    epochs_range = range(1, len(history['d_loss']) + 1)

    # Plot 1: Losses
    axes[0, 0].plot(epochs_range, history['d_loss'], 'o-', label='D Loss', linewidth=2, markersize=3)
    axes[0, 0].plot(epochs_range, history['g_loss'], 's-', label='G Loss', linewidth=2, markersize=3)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('GAN Losses Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='upper right', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Discriminator Accuracy
    axes[0, 1].plot(epochs_range, history['d_acc'], 'o-', color='green', linewidth=2, markersize=3)
    axes[0, 1].axhline(y=GAN_CONFIG['d_target_acc_min'], color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axhline(y=GAN_CONFIG['d_target_acc_max'], color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axhspan(GAN_CONFIG['d_target_acc_min'], GAN_CONFIG['d_target_acc_max'],
                       alpha=0.2, color='green')
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Discriminator Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # Plot 3: Learning Rate
    axes[1, 0].plot(epochs_range, history['learning_rates'], 'o-', color='purple', linewidth=2, markersize=3)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Learning Rate Decay', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Plot 4: Training Time
    axes[1, 1].plot(epochs_range, history['epoch_times'], 'o-', color='orange', linewidth=2, markersize=3)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{GAN_SAVE_DIR}/training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✓ Training curves saved: {plot_path}")

# ============================================
# GENERATE FINAL SAMPLES
# ============================================
print("\n" + "="*80)
print("GENERATING FINAL SAMPLE GRID")
print("="*80)

fig, axes = plt.subplots(NUM_CLASSES, 5, figsize=(15, NUM_CLASSES * 3))

for class_idx in range(NUM_CLASSES):
    # Generate 5 samples per class
    noise = generate_noise(5, LATENT_DIM)
    labels = np.full((5, 1), class_idx)

    # Direct call
    gen_imgs = generator([noise, labels], training=False)

    for sample_idx in range(5):
        gen_img = gen_imgs[sample_idx].numpy()
        gen_img = denormalize_images(gen_img)
        gen_img = np.clip(gen_img, 0, 1)

        axes[class_idx, sample_idx].imshow(gen_img)
        axes[class_idx, sample_idx].axis('off')

        if sample_idx == 0:
            axes[class_idx, sample_idx].set_ylabel(
                subset_class_names[class_idx],
                fontsize=12,
                fontweight='bold',
                rotation=0,
                ha='right',
                va='center'
            )

plt.suptitle('Final Generated Samples (5 per class)', fontsize=16, fontweight='bold')
plt.tight_layout()
final_samples_path = f"{GAN_SAVE_DIR}/final_samples_grid.png"
plt.savefig(final_samples_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Final samples saved: {final_samples_path}")

# ============================================
# TRAINING SUMMARY
# ============================================
print("\n" + "="*80)
print("🎉 TRAINING PIPELINE COMPLETE!")
print("="*80)
print(f"\n📊 Performance Metrics:")
print(f"   Total time: {total_time/3600:.2f} hours")
print(f"   Avg epoch time: {np.mean(history['epoch_times']):.1f}s")
if len(history['d_acc']) > 0:
    print(f"   Final D accuracy: {history['d_acc'][-1]:.2%}")
    print(f"   Best D accuracy: {max(history['d_acc']):.2%}")
    print(f"   Worst D accuracy: {min(history['d_acc']):.2%}")

print(f"\n📁 Results saved in: {GAN_SAVE_DIR}")
print(f"   ✓ Training history: training_history.json")
print(f"   ✓ Training curves: training_curves.png")
print(f"   ✓ Generated samples: samples/")
print(f"   ✓ Model checkpoints: checkpoints/")

print("\n✅ Ready for evaluation and CNN augmentation!")
