import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = "tc1.tif"  # Change path if needed
img = cv2.imread(image_path)

# Extract green channel for better vessel contrast
# convert to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply thresholding to create a binary image
# _, binary_image = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY)
# use adaptive thresholding for better results
binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
show_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

# run 5x5 grid search and if percent of white pixels is below certain threshold set that grid to white
def grid_search_threshold(image, grid_size=5, threshold=0.1):
    h, w = image.shape
    for i in range(0, h, grid_size):
        for j in range(0, w, grid_size):
            grid = image[i:i + grid_size, j:j + grid_size]
            white_pixels = np.sum(grid == 255)
            total_pixels = grid.size
            if white_pixels / total_pixels < threshold:
                image[i:i + grid_size, j:j + grid_size] = 255
    return image

# Apply grid search thresholding
binary_image = grid_search_threshold(binary_image)


