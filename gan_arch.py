# ============================================
# CELL 7: Define GAN Architecture (Generator + Discriminator)
# ============================================
import os
import sys

# Ensure the module can find your other scripts if they are in a specific folder
# sys.path.append('/content/GAN_Files') 

import GAN_Files.load_balanced_data as load_balanced_data

# Retrieve variables from previous steps
# This re-runs the subset logic to ensure names and paths are consistent
_, _, _, _, subset_class_names = load_balanced_data.create_balanced_subset()

# Manually define PROJECT_DIR as it was used in previous script
PROJECT_DIR = '/content/drive/MyDrive/GAN_Malware_Detection'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("DEFINING GAN ARCHITECTURE FOR MALWARE IMAGE GENERATION")
print("="*80)

"""
# GAN Configuration
LATENT_DIM = 100  # Size of random noise vector
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
NUM_CLASSES = 6  # For conditional GAN (class-specific generation)

print(f"\nGAN Configuration:")
print(f"  Latent dimension: {LATENT_DIM}")
print(f"  Image shape: {IMG_SHAPE}")
print(f"  Number of classes: {NUM_CLASSES}")
print(f"  GAN type: Conditional DCGAN (Deep Convolutional GAN)")
"""

# GAN Configuration
LATENT_DIM = 100  # Size of random noise vector
IMG_HEIGHT = 64   # <--- CHANGED FROM 224
IMG_WIDTH = 64    # <--- CHANGED FROM 224
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
NUM_CLASSES = 6  # For conditional GAN (class-specific generation)

print(f"\nGAN Configuration (PHASE 1 - GRAYSCALE/LOW RES):")
print(f"  Latent dimension: {LATENT_DIM}")
print(f"  Image shape: {IMG_SHAPE}")
print(f"  Number of classes: {NUM_CLASSES}")
print(f"  GAN type: Conditional DCGAN (64x64 Architecture)")

# ============================================
# GENERATOR NETWORK
# ============================================
# ============================================
# REPLACEMENT FOR GENERATOR NETWORK
# ============================================
def build_generator(latent_dim=100, num_classes=6):
    """
    Generator: Takes random noise + class label -> Generates 64x64x3 malware image
    Architecture: Dense -> Reshape (8x8) -> 3 ConvTranspose layers -> 64x64
    """

    # Input: Random noise vector
    noise_input = layers.Input(shape=(latent_dim,), name='noise_input')

    # Input: Class label (for conditional generation)
    label_input = layers.Input(shape=(1,), dtype='int32', name='class_label')
    label_embedding = layers.Embedding(num_classes, 50)(label_input)
    label_embedding = layers.Flatten()(label_embedding)

    # Concatenate noise with class embedding
    combined_input = layers.Concatenate()([noise_input, label_embedding])

    # 1. Start at 8x8 resolution (Dense Layer)
    # We want 256 filters at 8x8
    n_nodes = 8 * 8 * 256
    x = layers.Dense(n_nodes, use_bias=False)(combined_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Reshape((8, 8, 256))(x)

    # 2. Upsample 8x8 -> 16x16
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    # 3. Upsample 16x16 -> 32x32
    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    # 4. Upsample 32x32 -> 64x64 (Output)
    # Note: No Batch Norm on output layer! Tanh activation for [-1, 1] range
    output = layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh')(x)

    model = models.Model([noise_input, label_input], output, name='Generator_64x64')

    return model

# ============================================
# DISCRIMINATOR NETWORK
# ============================================
# ============================================
# REPLACEMENT FOR DISCRIMINATOR NETWORK
# ============================================
def build_discriminator(img_shape=(64, 64, 3), num_classes=6):
    """
    Discriminator: 64x64 Input -> 3 Downsampling Layers -> Binary Classification
    """
    # Input: Image
    img_input = layers.Input(shape=img_shape, name='image_input')

    # Input: Class label
    label_input = layers.Input(shape=(1,), dtype='int32', name='class_label')

    # Embed label to match image size (64x64)
    label_embedding = layers.Embedding(num_classes, 50)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Dense(64 * 64)(label_embedding)
    label_embedding = layers.Reshape((64, 64, 1))(label_embedding)

    # Concatenate image and label channel
    combined_input = layers.Concatenate()([img_input, label_embedding])

    # 1. Downsample 64 -> 32
    x = layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same')(combined_input)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Dropout(0.3)(x)

    # 2. Downsample 32 -> 16
    x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # 3. Downsample 16 -> 8
    x = layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Output Layer
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model([img_input, label_input], x, name='Discriminator_64x64')
    return model

# ============================================
# ADD LABEL SMOOTHING FUNCTIONS
# ============================================
def get_real_labels(batch_size, smooth=True):
    """Generate labels for real images with smoothing"""
    if smooth:
        # Label smoothing: use 0.9 instead of 1.0
        return np.ones((batch_size, 1)) * 0.9
    return np.ones((batch_size, 1))


def get_fake_labels(batch_size, smooth=True):
    """Generate labels for fake images with smoothing"""
    if smooth:
        # Label smoothing: use 0.1 instead of 0.0
        return np.ones((batch_size, 1)) * 0.1
    return np.zeros((batch_size, 1))


# ============================================
# CRITICAL: DATA NORMALIZATION
# ============================================
def normalize_images(images):
    """
    Normalize images from [0, 255] or [0, 1] to [-1, 1]
    This matches the generator's tanh output range
    """
    if images.max() > 1.0:
        # Images are in [0, 255]
        images = images / 127.5 - 1.0
    else:
        # Images are in [0, 1]
        images = images * 2.0 - 1.0
    return images


def denormalize_images(images):
    """Convert from [-1, 1] back to [0, 1] for visualization"""
    return (images + 1.0) / 2.0


# ============================================
# IMPROVED TRAINING STEP WITH LABEL SMOOTHING
# ============================================
def train_gan_step(real_images, real_labels, batch_size):
    """
    Single training step with label smoothing
    """
    # ============= TRAIN DISCRIMINATOR =============
    # Get smoothed labels
    real_y = get_real_labels(batch_size, smooth=True)
    fake_y = get_fake_labels(batch_size, smooth=True)

    # Generate fake images
    noise = generate_noise(batch_size, LATENT_DIM)
    fake_labels = generate_class_labels(batch_size, NUM_CLASSES)
    fake_images = generator.predict([noise, fake_labels], verbose=0)

    # Train on real images
    d_loss_real = discriminator.train_on_batch([real_images, real_labels], real_y)

    # Train on fake images
    d_loss_fake = discriminator.train_on_batch([fake_images, fake_labels], fake_y)

    # Average discriminator loss
    d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
    d_acc = 0.5 * (d_loss_real[1] + d_loss_fake[1])

    # ============= TRAIN GENERATOR =============
    noise = generate_noise(batch_size, LATENT_DIM)
    gen_labels = generate_class_labels(batch_size, NUM_CLASSES)

    # Generator wants discriminator to output 1.0 (real)
    # With label smoothing, target is 0.9
    valid_y = get_real_labels(batch_size, smooth=True)

    g_loss = gan.train_on_batch([noise, gen_labels], valid_y)

    return d_loss, d_acc, g_loss
# ============================================
# BUILD GAN MODELS
# ============================================
print("\n" + "="*80)
print("BUILDING GENERATOR")
print("="*80)

generator = build_generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES)
generator.summary()

generator_params = sum([tf.size(w).numpy() for w in generator.weights])
print(f"\nGenerator parameters: {generator_params:,}")

print("\n" + "="*80)
print("BUILDING DISCRIMINATOR")
print("="*80)

discriminator = build_discriminator(img_shape=IMG_SHAPE, num_classes=NUM_CLASSES)
discriminator.summary()

discriminator_params = sum([tf.size(w).numpy() for w in discriminator.weights])
print(f"\nDiscriminator parameters: {discriminator_params:,}")

# ============================================
# COMPILE DISCRIMINATOR
# ============================================
print("\n" + "="*80)
print("COMPILING DISCRIMINATOR")
print("="*80)

discriminator.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✓ Discriminator compiled with Adam optimizer (lr=0.0002)")

# ============================================
# BUILD COMBINED GAN MODEL
# ============================================
print("\n" + "="*80)
print("BUILDING COMBINED GAN MODEL")
print("="*80)

# Freeze discriminator when training generator
discriminator.trainable = False

# GAN inputs
noise_input = layers.Input(shape=(LATENT_DIM,), name='gan_noise_input')
label_input = layers.Input(shape=(1,), dtype='int32', name='gan_label_input')

# Generate fake image
generated_image = generator([noise_input, label_input])

# Discriminator evaluates fake image
validity = discriminator([generated_image, label_input])

# Combined model: Generator tries to fool discriminator
gan = models.Model([noise_input, label_input], validity, name='GAN')

gan.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='binary_crossentropy'
)

print("✓ Combined GAN model built and compiled")
gan.summary()

gan_params = sum([tf.size(w).numpy() for w in gan.trainable_weights])
print(f"\nGAN trainable parameters: {gan_params:,}")

# ============================================
# UTILITY FUNCTIONS
# ============================================
def generate_noise(batch_size, latent_dim):
    """Generate random noise vectors"""
    return np.random.normal(0, 1, size=(batch_size, latent_dim))

def generate_class_labels(batch_size, num_classes):
    """Generate random class labels"""
    return np.random.randint(0, num_classes, size=(batch_size, 1))

def generate_fake_images(generator, batch_size, latent_dim, num_classes):
    """Generate fake images with random class labels"""
    noise = generate_noise(batch_size, latent_dim)
    labels = generate_class_labels(batch_size, num_classes)
    fake_images = generator.predict([noise, labels], verbose=0)
    return fake_images, labels

# Test generation
print("\n" + "="*80)
print("TESTING IMAGE GENERATION")
print("="*80)

test_noise = generate_noise(4, LATENT_DIM)
test_labels = np.array([[0], [1], [2], [3]])  # One sample per class
test_images = generator.predict([test_noise, test_labels], verbose=0)

print(f"Generated test images shape: {test_images.shape}")
print(f"Generated images value range: [{test_images.min():.2f}, {test_images.max():.2f}]")
print("Note: Generator outputs tanh activation → range [-1, 1]")

# Visualize initial (untrained) generated images
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    # Convert from [-1, 1] to [0, 1] for display
    img_display = (test_images[i] + 1) / 2.0
    img_display = np.clip(img_display, 0, 1)
    axes[i].imshow(img_display)
    axes[i].set_title(f"Class {test_labels[i][0]}: {subset_class_names[test_labels[i][0]]}")
    axes[i].axis('off')

plt.suptitle("Untrained Generator Output (Random Noise)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PROJECT_DIR}/gan_untrained_output.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Untrained generator output saved to: {PROJECT_DIR}/gan_untrained_output.png")

print("\n" + "="*80)
print("✅ GAN ARCHITECTURE COMPLETE!")
print("="*80)
print(f"Generator: {generator_params:,} parameters")
print(f"Discriminator: {discriminator_params:,} parameters")
print(f"Total GAN parameters: {generator_params + discriminator_params:,}")
print("\n🎯 Ready for GAN training!")
