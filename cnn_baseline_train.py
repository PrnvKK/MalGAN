# ============================================
# CELL 6: Train Baseline CNN with Enhanced Progress Tracking
# ============================================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # Added missing imports
import time
import json
import os
from IPython.display import clear_output

# --- NEW: IMPORT YOUR MODULES ---
import MalGAN.cnn_arch as cnn_arch
import MalGAN.load_balanced_data as load_balanced_data

# --- NEW: INITIALIZE DATA AND MODEL ---
# This pulls the variables into the current script's scope
X_train, y_train, X_val, y_val, subset_class_names = load_balanced_data.create_balanced_subset()
baseline_model = cnn_arch.get_compiled_model('resnet')

# Define PROJECT_DIR if not already inherited
PROJECT_DIR = '/content/drive/MyDrive/GAN_Malware_Detection'
# --------------------------------

print("="*60)
print("TRAINING BASELINE CNN MODEL")
print("="*60)

# Normalize pixel values to [0, 1]
print("\nNormalizing data...")
X_train_norm = X_train.astype('float32') / 255.0
X_val_norm = X_val.astype('float32') / 255.0

print(f"Train data range: [{X_train_norm.min():.2f}, {X_train_norm.max():.2f}]")
print(f"Val data range: [{X_val_norm.min():.2f}, {X_val_norm.max():.2f}]")

# Training parameters
EPOCHS = 50
BATCH_SIZE = 32
VERBOSE = 0  # Disable default logs (we'll use custom callback)

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Steps per epoch: {len(X_train_norm) // BATCH_SIZE}")
print(f"Validation steps: {len(X_val_norm) // BATCH_SIZE}")
print(f"Optimizer: Adam (lr=0.0001)")
print(f"Early stopping patience: 10 epochs")

# Custom callback for clean progress display
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
        print("\n" + "="*80)
        print("TRAINING STARTED")
        print("="*80)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
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
        is_best = False
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
            self.epochs_no_improve = 0
            is_best = True
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

        # Warning if overfitting detected
        if train_acc - val_acc > 0.15:
            print("⚠️  Warning: Possible overfitting detected (train-val gap > 15%)")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        print(f"Total Training Time: {total_time/60:.2f} minutes")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print("="*80)

# Setup callbacks
checkpoint_path = f'{PROJECT_DIR}/baseline_cnn_best.h5'
history_path = f'{PROJECT_DIR}/baseline_training_history.json'

training_monitor = TrainingMonitor()

callbacks = [
    training_monitor,
    ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0  # Silence default messages
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

# Plot training history
print("\n" + "="*80)
print("GENERATING TRAINING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

epochs_range = range(1, len(history.history['accuracy']) + 1)
best_epoch = training_monitor.best_epoch

# Accuracy plot
axes[0].plot(epochs_range, history.history['accuracy'], 'o-', label='Training Accuracy', linewidth=2, markersize=4)
axes[0].plot(epochs_range, history.history['val_accuracy'], 's-', label='Validation Accuracy', linewidth=2, markersize=4)
axes[0].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Best Epoch ({best_epoch})')
axes[0].scatter([best_epoch], [training_monitor.best_val_acc], color='red', s=200, zorder=5, marker='*', edgecolors='black', linewidths=1.5)
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# Loss plot
axes[1].plot(epochs_range, history.history['loss'], 'o-', label='Training Loss', linewidth=2, markersize=4)
axes[1].plot(epochs_range, history.history['val_loss'], 's-', label='Validation Loss', linewidth=2, markersize=4)
axes[1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Best Epoch ({best_epoch})')
axes[1].scatter([best_epoch], [training_monitor.best_val_loss], color='red', s=200, zorder=5, marker='*', edgecolors='black', linewidths=1.5)
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
axes[1].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = f'{PROJECT_DIR}/baseline_training_curves.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Training plots saved to: {plot_path}")

# Generate and save predictions
print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)

y_val_pred_probs = baseline_model.predict(X_val_norm, batch_size=BATCH_SIZE, verbose=0)
y_val_pred = np.argmax(y_val_pred_probs, axis=1)

np.save(f'{PROJECT_DIR}/baseline_val_predictions.npy', y_val_pred)
np.save(f'{PROJECT_DIR}/baseline_val_true_labels.npy', y_val)

print(f"✓ Predictions saved to: {PROJECT_DIR}/baseline_val_predictions.npy")

print("\n" + "="*80)
print("✅ BASELINE CNN TRAINING COMPLETE!")
print("="*80)
print("✓ Best model saved with validation-based checkpointing")
print("✓ Early stopping applied to prevent overfitting")
print("✓ Learning rate reduced on plateau for better convergence")
print("✓ All results and plots saved to Google Drive")
print("\n🎯 Ready to evaluate baseline performance and build GAN!")
