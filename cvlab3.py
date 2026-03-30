# ==================== IMPORTS ====================

import cv2
# OpenCV library → provides optimized image processing functions (filters, transforms, etc.)

import numpy as np
# NumPy → used because images are stored as matrices (arrays of pixel values)

import matplotlib.pyplot as plt
# Matplotlib → used to display images in notebooks (cv2.imshow doesn’t work well here)

from skimage.util import random_noise
# Function to artificially add different types of noise (Gaussian, salt, pepper, etc.)


# ==================== DISPLAY FUNCTION ====================

def show(img, title=""):
    plt.figure(figsize=(4,4))
    # Create a new figure window with fixed size (so images look consistent)

    plt.imshow(img, cmap='gray')
    # Display image → cmap='gray' ensures grayscale images are shown correctly

    plt.title(title)
    # Add title for clarity

    plt.axis('off')
    # Remove x/y axes (cleaner visualization)


# ==================== ASSIGNMENT 1 ====================

img = cv2.imread('img.png', 0) / 255.0
# Read image in grayscale (0 → single channel)
# Divide by 255 → normalize pixel values from [0,255] → [0,1]
# WHY: random_noise expects float images in [0,1]

noisy = random_noise(img, mode='gaussian', mean=0, var=0.05)
# Add Gaussian noise:
# mean=0 → noise centered around 0
# var=0.05 → controls noise intensity

noisy = (noisy * 255).astype(np.uint8)
# Convert back to standard image format:
# Multiply by 255 → scale back to [0,255]
# astype(uint8) → convert to integer pixel format

filtered = cv2.blur(noisy, (5,5))
# Apply arithmetic mean filter:
# Each pixel = average of its 5x5 neighborhood
# WHY: reduces Gaussian noise by smoothing

show(img, "Original")
show(noisy, "Gaussian Noise")
show(filtered, "Arithmetic Mean Filter")


# ==================== ASSIGNMENT 2 ====================

def geometric_mean_filter(img, k=3):
    img = img.astype(np.float32) + 1
    # Convert to float for precision
    # Add 1 → prevents log(0) which is undefined

    kernel = np.ones((k,k))
    # Create k×k window (all ones)

    log_img = np.log(img)
    # Convert multiplication → addition (log domain trick)

    filtered = cv2.filter2D(log_img, -1, kernel)
    # Apply convolution in log space

    return np.exp(filtered / (k*k)).astype(np.uint8)
    # Divide by number of elements → average in log domain
    # exp() → convert back from log domain
    # This gives geometric mean


filtered = geometric_mean_filter(noisy, 3)

show(noisy, "Gaussian Noise")
show(filtered, "Geometric Mean Filter")


# ==================== ASSIGNMENT 3 ====================

def harmonic_mean_filter(img, k=3):
    img = img.astype(np.float32)
    # Convert to float for division operations

    kernel = np.ones((k,k))

    denom = cv2.filter2D(1.0 / (img + 1e-6), -1, kernel)
    # Compute sum of reciprocals:
    # 1e-6 prevents division by zero

    return ((k*k) / denom).astype(np.uint8)
    # Harmonic mean formula:
    # n / (sum of reciprocals)


salt = random_noise(img, mode='salt', amount=0.1)
# Add salt noise → random white pixels

salt = (salt * 255).astype(np.uint8)

filtered = harmonic_mean_filter(salt, 3)

show(salt, "Salt Noise")
show(filtered, "Harmonic Mean Filter")


# ==================== ASSIGNMENT 4 ====================

def contraharmonic_mean(img, k=3, Q=1.5):
    img = img.astype(np.float32)

    kernel = np.ones((k,k))

    num = cv2.filter2D(img**(Q+1), -1, kernel)
    # Numerator → sum(x^(Q+1))

    den = cv2.filter2D(img**Q + 1e-6, -1, kernel)
    # Denominator → sum(x^Q)
    # 1e-6 prevents division by zero

    return (num / den).astype(np.uint8)
    # Final contraharmonic mean result


pepper = random_noise(img, mode='pepper', amount=0.1)
# Add pepper noise → random black pixels

pepper = (pepper * 255).astype(np.uint8)

filtered = contraharmonic_mean(pepper, 3, Q=1.5)

show(pepper, "Pepper Noise")
show(filtered, "Contraharmonic Mean Filter")


# ==================== ASSIGNMENT 5 ====================

sp = random_noise(img, mode='s&p', amount=0.2)
# Add both salt AND pepper noise

sp = (sp * 255).astype(np.uint8)

filtered = cv2.medianBlur(sp, 5)
# Median filter:
# Replace pixel with median of neighborhood
# WHY: removes extreme values (salt & pepper)

show(sp, "Salt & Pepper Noise")
show(filtered, "Median Filter")


# ==================== ASSIGNMENT 6 ====================

sp_noise = random_noise(img, mode='s&p', amount=0.2)
sp_noise = (sp_noise * 255).astype(np.uint8)

min_filtered = cv2.erode(sp_noise, np.ones((3,3), np.uint8), iterations=1)
# Erosion = min filter
# Replaces pixel with minimum value in neighborhood → removes white noise (salt)

max_filtered = cv2.dilate(sp_noise, np.ones((3,3), np.uint8), iterations=1)
# Dilation = max filter
# Replaces pixel with maximum value → removes black noise (pepper)

show(sp_noise, "Salt & Pepper Noise")
show(min_filtered, "Min Filter")
show(max_filtered, "Max Filter")


# ==================== ASSIGNMENT 7 ====================

gaussian_noise = random_noise(img, mode='gaussian', mean=0, var=0.05)
gaussian_noise = (gaussian_noise * 255).astype(np.uint8)

mixed_noisy_image = random_noise(gaussian_noise / 255.0, mode='s&p', amount=0.1)
# First Gaussian noise, then salt & pepper → mixed noise

mixed_noisy_image = (mixed_noisy_image * 255).astype(np.uint8)

show(img, "Original Image")
show(mixed_noisy_image, "Mixed Noise")


def alpha_trimmed_mean_filter(img, k=3, d=2):
    img = img.astype(np.float32)
    rows, cols = img.shape

    filtered_img = np.zeros_like(img)

    pad = k // 2

    for i in range(pad, rows - pad):
        for j in range(pad, cols - pad):
            window = img[i-pad:i+pad+1, j-pad:j+pad+1].flatten()
            # Extract neighborhood and flatten to 1D array

            sorted_window = np.sort(window)
            # Sort values

            trim_count = int(d / 2)
            # Number of elements to remove from each side

            trimmed_window = sorted_window[trim_count : len(sorted_window)-trim_count]
            # Remove extreme values (noise)

            filtered_img[i, j] = np.mean(trimmed_window)
            # Average remaining values

    return filtered_img.astype(np.uint8)


alpha_trimmed_filtered = alpha_trimmed_mean_filter(mixed_noisy_image, k=5, d=4)

arithmetic_mean_filtered = cv2.blur(mixed_noisy_image, (5,5))
median_filtered = cv2.medianBlur(mixed_noisy_image, 5)

show(alpha_trimmed_filtered, "Alpha-Trimmed")
show(arithmetic_mean_filtered, "Mean Filter")
show(median_filtered, "Median Filter")


# ==================== ASSIGNMENT 8 ====================

img_float = img.astype(np.float64)

rows, cols = img.shape

variance_map = np.zeros_like(img_float)

for j in range(cols):
    variance_map[:, j] = 0.01 + (0.09 * (j / (cols - 1)))
    # Create varying noise → left low noise, right high noise


spatially_noisy_image = random_noise(img_float, mode='gaussian', var=variance_map)
spatially_noisy_image = (spatially_noisy_image * 255).astype(np.uint8)

show(img, "Original")
show(spatially_noisy_image, "Spatial Noise")


def adaptive_local_noise_reduction_filter(img, global_noise_variance, ksize=7):
    img_float = img.astype(np.float32)

    filtered_img = np.zeros_like(img_float)

    pad = ksize // 2

    for i in range(pad, img.shape[0] - pad):
        for j in range(pad, img.shape[1] - pad):

            window = img_float[i-pad:i+pad+1, j-pad:j+pad+1]

            local_mean = np.mean(window)
            local_variance = np.var(window)

            # Compare local vs global noise
            if local_variance < global_noise_variance:
                weight = local_variance / (global_noise_variance + 1e-6)
            else:
                weight = global_noise_variance / (local_variance + 1e-6)

            filtered_img[i, j] = img_float[i, j] - weight * (img_float[i, j] - local_mean)

    return np.clip(filtered_img, 0, 255).astype(np.uint8)


global_noise_variance_estimate = np.mean(variance_map) * (255**2)

adaptive_filtered = adaptive_local_noise_reduction_filter(
    spatially_noisy_image,
    global_noise_variance_estimate,
    ksize=7
)

show(adaptive_filtered, "Adaptive Filter")


# ==================== ASSIGNMENT 9 ====================

sp_noisy = random_noise(img, mode='s&p', amount=0.1)
sp_noisy = (sp_noisy * 255).astype(np.uint8)

filtered_3 = cv2.medianBlur(sp_noisy, 3)
filtered_5 = cv2.medianBlur(sp_noisy, 5)
filtered_7 = cv2.medianBlur(sp_noisy, 7)


def calculate_psnr(original, processed):
    mse = np.mean((original - processed)**2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))


original_uint8 = (img * 255).astype(np.uint8)

print("PSNR:")
print("Noisy:", calculate_psnr(original_uint8, sp_noisy))
print("3x3:", calculate_psnr(original_uint8, filtered_3))
print("5x5:", calculate_psnr(original_uint8, filtered_5))
print("7x7:", calculate_psnr(original_uint8, filtered_7))


# ==================== ASSIGNMENT 10 ====================

def detect_noise_type(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Compute histogram → counts frequency of pixel intensities

    total = image.size

    zero_ratio = hist[0][0] / total
    # Percentage of black pixels

    max_ratio = hist[255][0] / total
    # Percentage of white pixels

    if zero_ratio > 0.02 and max_ratio > 0.02:
        return 'salt_pepper'
    elif zero_ratio > 0.02:
        return 'pepper'
    elif max_ratio > 0.02:
        return 'salt'
    else:
        return 'gaussian'


def apply_automatic_filter(image):
    noise_type = detect_noise_type(image)

    if noise_type == 'salt_pepper':
        return cv2.medianBlur(image, 5), "Median Filter"
    elif noise_type == 'salt':
        return contraharmonic_mean(image, 5, Q=1.5), "Contraharmonic Salt"
    elif noise_type == 'pepper':
        return contraharmonic_mean(image, 5, Q=-1.5), "Contraharmonic Pepper"
    else:
        return cv2.blur(image, (5,5)), "Mean Filter"


gaussian_noise_img = random_noise(img, mode='gaussian', var=0.03)

mixed = random_noise(gaussian_noise_img, mode='s&p', amount=0.05)
mixed = (mixed * 255).astype(np.uint8)

filtered, name = apply_automatic_filter(mixed)

show(img, "Original")
show(mixed, "Mixed Noise")
show(filtered, name)
