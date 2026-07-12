
# ============================================
# CELL 6: V2 - Train CNN with GAN Augmentation
# ============================================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time
import json
import os
import cv2
import tensorflow as tf
from IPython.display import clear_output

# --- IMPORT MODULES ---
import MalGAN.cnn_arch as cnn_arch
import MalGAN.load_balanced_data as load_balanced_data

# ============================================
# 1. LOAD ORIGINAL DATASET
# ============================================
print("="*60)
print("LOADING ORIGINAL BALANCED DATASET")
print("="*60)

# Uses your existing loader logic perfectly
X_train_orig, y_train_orig, X_val, y_val, subset_class_names = load_balanced_data.create_balanced_subset()

print(f"\nOriginal X_train: {X_train_orig.shape}")
print(f"Original y_train: {y_train_orig.shape}")

# ============================================
# 2. LOAD & COMBINE AUGMENTED DATA
# ============================================
print("\n" + "="*60)
print("LOADING GAN AUGMENTED DATA")
print("="*60)

AUG_DATA_DIR = "/content/augmented_data"
X_train_aug = []
y_train_aug = []

if not os.path.exists(AUG_DATA_DIR):
    print(f"❌ Error: Augmented data directory not found at {AUG_DATA_DIR}")
    print("Please run 'gan_augment.py' first!")
    exit()

print(f"Loading augmented images from: {AUG_DATA_DIR}")

# Mapping class names to indices
class_to_idx = {name: idx for idx, name in enumerate(subset_class_names)}

total_aug = 0

for class_name in subset_class_names:
    class_dir = os.path.join(AUG_DATA_DIR, class_name)
    class_idx = class_to_idx.get(class_name)
    
    if class_idx is None:
        print(f"Skipping unknown class folder: {class_name}")
        continue
        
    if not os.path.exists(class_dir):
        print(f"⚠️ Warning: No augmented data found for class '{class_name}'")
        continue
        
    # Read images
    files = os.listdir(class_dir)
    count = 0
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(class_dir, f)
            
            # Read image (OpenCV reads as BGR, convert to RGB)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Ensure it is 224x224 (it should be from gan_augment, but safety first)
            if img.shape[:2] != (224, 224):
                img = cv2.resize(img, (224, 224))
                
            X_train_aug.append(img)
            y_train_aug.append(class_idx)
            count += 1
            
    print(f"  Loaded {count} augmented images for '{class_name}'")
    total_aug += count

if total_aug == 0:
    print("❌ No augmented images loaded! Check paths.")
    exit()

X_train_aug = np.array(X_train_aug)
y_train_aug = np.array(y_train_aug)

print(f"\nTotal Augmented X_train: {X_train_aug.shape}")

# COMBINE DATASETS
print("\nCombining datasets...")
X_train_combined = np.concatenate([X_train_orig, X_train_aug], axis=0)
y_train_combined = np.concatenate([y_train_orig, y_train_aug], axis=0)

# Shuffle the combined dataset
indices = np.arange(len(X_train_combined))
np.random.shuffle(indices)
X_train = X_train_combined[indices]
y_train = y_train_combined[indices]

print(f"✓ Final Combined X_train: {X_train.shape}")
print(f"✓ Final Combined y_train: {y_train.shape}")


# ============================================
# 3. INITIALIZE MODEL (SAME ARCHITECTURE)
# ============================================
print("\n" + "="*60)
print("INITIALIZING CNN MODEL (RESNET)")
print("="*60)

# Re-use your exact architecture function
baseline_model = cnn_arch.get_compiled_model('resnet')

# Define PROJECT_DIR 
PROJECT_DIR = '/content/drive/MyDrive/GAN_Malware_Detection'


# ============================================
# 4. TRAINING SETUP (EXACT CLONE OF BASELINE)
# ============================================

# Normalize pixel values to [0, 1]
print("\nNormalizing data...")
X_train_norm = X_train.astype('float32') / 255.0
X_val_norm = X_val.astype('float32') / 255.0

print(f"Train data range: [{X_train_norm.min():.2f}, {X_train_norm.max():.2f}]")
print(f"Val data range: [{X_val_norm.min():.2f}, {X_val_norm.max():.2f}]")

# Training parameters (Same as baseline for fair comparison)
EPOCHS = 50
BATCH_SIZE = 32
VERBOSE = 0 

print("\nTRAINING CONFIGURATION")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Steps per epoch: {len(X_train_norm) // BATCH_SIZE}")
print(f"Optimizer: Adam (lr=0.0001)")

# Custom callback for clean progress display (Re-used)
class TrainingMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print("\nTRAINING STARTED (AUGMENTED)")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        elapsed_time = time.time() - self.start_time

        train_loss = logs.get('loss')
        train_acc = logs.get('accuracy')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')
        current_lr = float(self.model.optimizer.learning_rate.numpy())

        # Check if this is the best model
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
            self.epochs_no_improve = 0
            best_marker = " ⭐ BEST!"
        else:
            self.epochs_no_improve += 1
            best_marker = ""

        # Clear and print epoch summary
        print("\n" + "─"*80)
        print(f"Epoch {epoch + 1}/{EPOCHS} {best_marker}")
        print("─"*80)

        # Create progress bar
        progress = (epoch + 1) / EPOCHS
        bar_length = 50
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"Progress: [{bar}] {progress*100:.1f}%")

        # Training metrics
        print(f"\nTraining   → Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"Validation → Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

        # Additional info
        print(f"\nLearning Rate: {current_lr:.2e} | Epoch Time: {epoch_time:.1f}s | Total Time: {elapsed_time/60:.1f}m")
        print(f"Best Val Acc: {self.best_val_acc:.4f} (Epoch {self.best_epoch}) | No Improve: {self.epochs_no_improve}/10")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        print(f"Total Training Time: {total_time/60:.2f} minutes")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")

# Setup callbacks
checkpoint_path = f'{PROJECT_DIR}/augmented_cnn_best.h5' # Distinct name
history_path = f'{PROJECT_DIR}/augmented_training_history.json'

training_monitor = TrainingMonitor()

callbacks = [
    training_monitor,
    ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=0
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=0
    )
]

# Train the model
start_time = time.time()

history = baseline_model.fit(
    X_train_norm,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_norm, y_val),
    callbacks=callbacks,
    verbose=VERBOSE
)

training_time = time.time() - start_time

# Save training history
history_dict = {
    'loss': [float(x) for x in history.history['loss']],
    'accuracy': [float(x) for x in history.history['accuracy']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    'training_time_minutes': training_time/60,
    'epochs_completed': len(history.history['loss']),
    'best_val_accuracy': float(training_monitor.best_val_acc),
    'best_val_loss': float(training_monitor.best_val_loss),
    'best_epoch': int(training_monitor.best_epoch)
}

with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)

print(f"\n✓ Training history saved to: {history_path}")

baseline_model.save(checkpoint_path)
print(f"✓ Best model saved to: {checkpoint_path}")

# Final evaluation on validation set
print("\n" + "="*80)
print("FINAL EVALUATION ON VALIDATION SET")
print("="*80)

val_loss, val_accuracy = baseline_model.evaluate(
    X_val_norm,
    y_val,
    batch_size=BATCH_SIZE,
    verbose=0
)

print(f"Final Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"Final Validation Loss:     {val_loss:.4f}")

# Generate and save predictions
y_val_pred_probs = baseline_model.predict(X_val_norm, batch_size=BATCH_SIZE, verbose=0)
y_val_pred = np.argmax(y_val_pred_probs, axis=1)

np.save(f'{PROJECT_DIR}/augmented_val_predictions.npy', y_val_pred)

print("\n" + "="*80)
print("✅ AUGMENTED CNN TRAINING COMPLETE!")
print("="*80)
