import os
import numpy as np
import cv2

images_path = "DRIVE/image"
masks_path = "DRIVE/mask"

image_output_path = "train_images"
mask_output_path = "train_masks"

num_images = os.listdir(images_path)

def tiff_to_grayscale(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale using the luminosity method
    gray = 0.299 * image_rgb[:, :, 0] + 0.587 * image_rgb[:, :, 1] + 0.114 * image_rgb[:, :, 2]
    return gray.astype(np.uint8)

def shuffle_image_blocks(img, mask, num_blocks=8, seed=None):
    # resize images to 512x512
    img = cv2.resize(img, (512, 512))
    mask = cv2.resize(mask, (512, 512))
    if seed is not None:
        np.random.seed(seed)

    height, width = img.shape
    block_height = height // num_blocks
    block_width = width // num_blocks

    # Step 1: Extract blocks and their original positions
    img_blocks = []
    mask_blocks = []
    original_positions = []

    for i in range(num_blocks):
        for j in range(num_blocks):
            img_block = img[i*block_height:(i+1)*block_height,
                            j*block_width:(j+1)*block_width]
            mask_block = mask[i*block_height:(i+1)*block_height,
                              j*block_width:(j+1)*block_width]

            img_blocks.append(img_block)
            mask_blocks.append(mask_block)
            original_positions.append((i, j))

    # Step 2: Shuffle positions (same for both image and mask)
    shuffled_positions = original_positions.copy()
    np.random.shuffle(shuffled_positions)

    # Step 3: Reconstruct both image and mask
    reconstructed_img = np.zeros_like(img)
    reconstructed_mask = np.zeros_like(mask)

    for img_block, mask_block, (i_new, j_new) in zip(img_blocks, mask_blocks, shuffled_positions):
        reconstructed_img[i_new*block_height:(i_new+1)*block_height,
                          j_new*block_width:(j_new+1)*block_width] = img_block

        reconstructed_mask[i_new*block_height:(i_new+1)*block_height,
                           j_new*block_width:(j_new+1)*block_width] = mask_block

    return reconstructed_img, reconstructed_mask

for i in range(1,len(num_images)+1):
    image_file = f"{images_path}/{i}.tif"
    mask_file = f"{masks_path}/{i}.png"

    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    gray = tiff_to_grayscale(image)

    shuffled_image, shuffled_mask = shuffle_image_blocks(gray, mask, num_blocks=8, seed=42)
    cv2.imwrite(f"{image_output_path}/{i}.png", shuffled_image)
    cv2.imwrite(f"{mask_output_path}/{i}.png", shuffled_mask)
    
    

    
