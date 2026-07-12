import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Import the previous module
import MalGAN.load_balanced_data as load_balanced_data

# ============================================
# CONFIGURATION & DATA LOADING
# ============================================

# We fetch the data and variables needed from the previous module
# This ensures X_train, X_val, etc., are defined in this script's scope
X_train, y_train, X_val, y_val, subset_class_names = load_balanced_data.create_balanced_subset()

# Constants
NUM_CLASSES = len(subset_class_names)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
PROJECT_DIR = '/content/drive/MyDrive/GAN_Malware_Detection'

print("="*60)
print("DEFINING CNN MODEL ARCHITECTURE")
print("="*60)

print(f"\nDataset Info:")
print(f"  Training samples: {len(X_train)}")
print(f"  Validation samples: {len(X_val)}")
print(f"  Number of classes: {NUM_CLASSES}")
print(f"  Input shape: {INPUT_SHAPE}")

# ============================================
# MODEL DEFINITIONS
# ============================================

def create_resnet_model(input_shape, num_classes):
    """ResNet50 with transfer learning."""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze early layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_custom_cnn(input_shape, num_classes):
    """Custom lighter CNN."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ============================================
# INITIALIZATION (Main Execution)
# ============================================

def get_compiled_model(model_type='resnet'):
    if model_type == 'resnet':
        print("\nCreating ResNet50 model...")
        model = create_resnet_model(INPUT_SHAPE, NUM_CLASSES)
    else:
        print("\nCreating Custom CNN model...")
        model = create_custom_cnn(INPUT_SHAPE, NUM_CLASSES)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Define and display model if run directly
if __name__ == "__main__":
    baseline_model = get_compiled_model('resnet')
    baseline_model.summary()

    # Count parameters
    trainable_params = sum([tf.size(w).numpy() for w in baseline_model.trainable_weights])
    total_params = sum([tf.size(w).numpy() for w in baseline_model.weights])

    print("\n" + "="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*60)

    # Setup callbacks
    checkpoint_path = f'{PROJECT_DIR}/baseline_cnn_best.h5'
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]

    print("\n✓ Model architecture defined!")
    print(f"✓ Checkpoints will be saved to: {checkpoint_path}")