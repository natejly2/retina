# %%
images_path = "DRIVE/image"
masks_path = "DRIVE/mask"

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation,
    LeakyReLU, MaxPool2D, UpSampling2D, Concatenate
)
from tensorflow.keras.optimizers import Adam

# %%
# Load images and masks
images = []
masks = []
for i in range(len(os.listdir(images_path))):
    image_name = f"{i+1}.tif"
    mask_name = f"{i+1}.png"
    image = plt.imread(os.path.join(images_path, image_name))
    mask = plt.imread(os.path.join(masks_path, mask_name))
    mask = np.where(mask > 0, 1, 0)  # binarize the mask
    images.append(image)
    masks.append(mask)

# Display the first image and its corresponding mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(images[0])
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(masks[0], cmap="gray")
plt.title("Mask")
plt.axis("off")
plt.show()

# %%
# Preprocessing functions
def rgb2gray(rgb):
    """Convert RGB to grayscale using weighted average"""
    if len(rgb.shape) == 3:  # Single image
        return rgb[:, :, 1] * 0.75 + rgb[:, :, 2] * 0.25
    elif len(rgb.shape) == 4:  # Batch of images
        return rgb[:, :, :, 1] * 0.75 + rgb[:, :, :, 2] * 0.25

def preprocess_image(image):
    """Preprocess a single image"""
    # Convert to grayscale if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = rgb2gray(image)
    else:
        gray = image
    
    # Normalize to 0-255 range
    gray = ((gray - np.min(gray)) / (np.max(gray) - np.min(gray)) * 255).astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Gamma correction
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    corrected = cv2.LUT(enhanced, table)
    
    return corrected

# Preprocess all images
processed_images = []
for image in images:
    processed_img = preprocess_image(image)
    processed_images.append(processed_img)

# %%
# Create training patches
img_train_path = "train_images"
masks_train_path = "train_masks"

if not os.path.exists(img_train_path):
    os.makedirs(img_train_path)
if not os.path.exists(masks_train_path):
    os.makedirs(masks_train_path)

np.random.seed(42)  # For reproducibility

def make_patches(images, masks, patch_size=48, num_samples=100):
    """Extract random patches from images and masks"""
    patch_images, patch_masks = [], []
    
    for img, msk in zip(images, masks):
        H, W = img.shape[:2]
        for _ in range(num_samples):
            x = np.random.randint(0, W - patch_size)
            y = np.random.randint(0, H - patch_size)
            
            patch_img = img[y:y+patch_size, x:x+patch_size]
            patch_mask = msk[y:y+patch_size, x:x+patch_size]
            
            patch_images.append(patch_img)
            patch_masks.append(patch_mask)
    
    return patch_images, patch_masks

patch_images, patch_masks = make_patches(processed_images, masks)

# Convert to numpy arrays and add channel dimension
X = np.array(patch_images, dtype='float32')
Y = np.array(patch_masks, dtype='float32')

# Add channel dimension for grayscale images
X = X.reshape(-1, 48, 48, 1)
Y = Y.reshape(-1, 48, 48, 1)

# Normalize X to [0,1] range
X = X / 255.0

print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# Save patches to disk
for i, (img, mask) in enumerate(zip(patch_images, patch_masks)):
    cv2.imwrite(f"train_images/{i+1}.png", img.astype(np.uint8))
    cv2.imwrite(f"train_masks/{i+1}.png", (mask * 255).astype(np.uint8))

# Show first 5 patches
plt.figure(figsize=(15, 10))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(patch_images[i], cmap='gray')
    plt.title(f"Image {i+1}")
    plt.axis("off")

    plt.subplot(2, 5, i + 6)
    plt.imshow(patch_masks[i], cmap='gray')
    plt.title(f"Mask {i+1}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# %%
# Define U-Net model components
class LinearTransform(tf.keras.layers.Layer):
    def __init__(self, name="LinearTransform"):
        super(LinearTransform, self).__init__(name=name)
        self.conv = Conv2D(1, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn = BatchNormalization()
        self.activation = Activation('sigmoid')

    def call(self, x, training=True):
        c = self.conv(x)
        out = x * (1 + self.activation(c))
        return self.bn(out, training=training)

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, out_ch, residual_path=False, stride=1):
        super(ResBlock, self).__init__()
        self.residual_path = residual_path

        self.conv1 = Conv2D(out_ch, 3, stride, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.act1 = LeakyReLU()

        self.conv2 = Conv2D(out_ch, 3, 1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()

        if residual_path:
            self.conv_sc = Conv2D(out_ch, 1, stride, padding='same', use_bias=False)
            self.bn_sc = BatchNormalization()

        self.act2 = LeakyReLU()

    def call(self, x, training=True):
        y = self.act1(self.bn1(self.conv1(x), training=training))
        y = self.bn2(self.conv2(y), training=training)

        if self.residual_path:
            x = self.bn_sc(self.conv_sc(x), training=training)

        y = x + y
        return self.act2(y)

def create_unet_model(input_shape=(48, 48, 1)):
    """Create a simplified U-Net model"""
    inputs = Input(shape=input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    
    # Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Concatenate()([up5, conv3])
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Concatenate()([up6, conv2])
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Concatenate()([up7, conv1])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    # Output
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
    
    return Model(inputs, outputs)

# %%
# Training setup
EPOCHS = 200
VAL_TIME = 2
LR = 3e-4
BATCH_SIZE = 64

# Split data
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create and compile model
model = create_unet_model()
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# %%
# Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_freq=VAL_TIME,
    shuffle=True
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Test the model on a few validation samples
predictions = model.predict(X_val[:5])

plt.figure(figsize=(15, 10))
for i in range(5):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_val[i].squeeze(), cmap='gray')
    plt.title(f"Input {i+1}")
    plt.axis("off")
    
    plt.subplot(3, 5, i + 6)
    plt.imshow(Y_val[i].squeeze(), cmap='gray')
    plt.title(f"True Mask {i+1}")
    plt.axis("off")
    
    plt.subplot(3, 5, i + 11)
    plt.imshow(predictions[i].squeeze(), cmap='gray')
    plt.title(f"Predicted {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()