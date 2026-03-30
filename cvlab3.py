# ==================== IMPORTS ====================

import numpy as np              # Used for matrix operations (images are matrices)
import cv2                     # OpenCV for image processing
import matplotlib.pyplot as plt  # For visualization

# ==================== UTILITY FUNCTION ====================

# Function to display images nicely
def show(img, title):
    plt.imshow(img, cmap='gray')   # Use grayscale colormap
    plt.title(title)
    plt.axis('off')                # Remove axis for cleaner display
    plt.show()


# ==================== ASSIGNMENT 1 ====================
# Custom convolution implementation

def convolution2d(image, kernel):
    h, w = image.shape             # Image height & width
    kh, kw = kernel.shape          # Kernel size

    # Padding ensures output size = input size
    pad_h, pad_w = kh // 2, kw // 2

    # Add zero padding around image borders
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Initialize output image (float to avoid overflow)
    output = np.zeros_like(image, dtype=np.float32)

    # Slide kernel across image
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]  # Extract local patch
            output[i, j] = np.sum(region * kernel)  # Convolution operation

    return output


# Load grayscale image
img = cv2.imread("landscape.jpg", cv2.IMREAD_GRAYSCALE)

# Averaging kernel (blur)
kernel = np.ones((3, 3), dtype=np.float32) / 9

# Apply custom convolution
out_custom = convolution2d(img, kernel)

# Apply OpenCV convolution (optimized)
out_library = cv2.filter2D(img, -1, kernel)

# Display results
show(img, "Original Image")
show(out_custom, "Custom Convolution Output")
show(out_library, "cv2.filter2D Output")

# Compare outputs
diff = np.abs(out_custom - out_library)
print(f"Mean Absolute Difference: {np.mean(diff):.4f}")
show(diff, "Difference (Custom - Library)")


# ==================== ASSIGNMENT 2 ====================

# Add Gaussian noise (smooth random noise)
def add_gaussian_noise(img, mean=0, std=20):
    noise = np.random.normal(mean, std, img.shape)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)  # Keep valid pixel range
    return noisy


# Add salt & pepper noise (random black/white pixels)
def add_salt_pepper_noise(img, prob=0.02):
    noisy = img.copy()
    rnd = np.random.rand(*img.shape)

    noisy[rnd < prob/2] = 0        # Pepper (black)
    noisy[rnd > 1 - prob/2] = 255  # Salt (white)

    return noisy


# Generate noisy images
gaussian_noisy = add_gaussian_noise(img)
sp_noisy = add_salt_pepper_noise(img)


# Mean filters (simple blur)
mean_3 = cv2.blur(gaussian_noisy, (3, 3))
mean_5 = cv2.blur(gaussian_noisy, (5, 5))

# Gaussian filters (weighted blur)
gauss_05 = cv2.GaussianBlur(gaussian_noisy, (5, 5), 0.5)
gauss_1  = cv2.GaussianBlur(gaussian_noisy, (5, 5), 1)
gauss_2  = cv2.GaussianBlur(gaussian_noisy, (5, 5), 2)


# PSNR calculation (quality metric)
def psnr(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    return 10 * np.log10((255 ** 2) / mse)

print("PSNR Values:")
print("Mean 3x3:", psnr(img, mean_3))
print("Mean 5x5:", psnr(img, mean_5))
print("Gaussian σ=0.5:", psnr(img, gauss_05))
print("Gaussian σ=1:", psnr(img, gauss_1))
print("Gaussian σ=2:", psnr(img, gauss_2))


# Display results
def show_multiple(images, titles):
    plt.figure(figsize=(12, 6))
    for i, (im, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i+1)
        plt.imshow(im, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()


show_multiple(
    [img, gaussian_noisy, mean_3, mean_5, gauss_1, gauss_2],
    ["Original", "Gaussian Noise", "Mean 3x3", "Mean 5x5", "Gaussian σ=1", "Gaussian σ=2"]
)


# ==================== ASSIGNMENT 3 ====================

# Blur image to remove noise
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)

# Laplacian kernel (edge detector)
laplacian_kernel = np.array([
    [0, -1, 0],
    [-1,  4, -1],
    [0, -1, 0]
])

# Extract edges
edges = cv2.filter2D(blurred, -1, laplacian_kernel)

# Laplacian sharpening (add edges back)
k_lap = 1.0
laplacian_sharp = np.clip(blurred + k_lap * edges, 0, 255).astype(np.uint8)

# Unsharp masking (classic sharpening)
k1, k2, k3 = 0.5, 1.0, 1.5

unsharp_05 = np.clip(img + k1 * (img - blurred), 0, 255).astype(np.uint8)
unsharp_10 = np.clip(img + k2 * (img - blurred), 0, 255).astype(np.uint8)
unsharp_15 = np.clip(img + k3 * (img - blurred), 0, 255).astype(np.uint8)

# Display sharpening results
titles = ["Original", "Blurred", "Laplacian Sharpened", "Unsharp k=0.5", "Unsharp k=1.0", "Unsharp k=1.5"]
images = [img, blurred, laplacian_sharp, unsharp_05, unsharp_10, unsharp_15]

plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()


# ==================== ASSIGNMENT 4 ====================

import time

# Direct 2D Gaussian filtering
start = time.time()
direct = cv2.GaussianBlur(img, (5, 5), 1)
time_direct = time.time() - start

# Separable Gaussian (1D + 1D → faster)
g1d = cv2.getGaussianKernel(5, 1)

start = time.time()
separable = cv2.sepFilter2D(img, -1, g1d, g1d)
time_separable = time.time() - start

# Compare results
difference = cv2.absdiff(direct, separable)
max_diff = np.max(difference)

print("Execution Time Comparison:")
print(f"Direct 2D Gaussian   : {time_direct:.6f} seconds")
print(f"Separable Gaussian   : {time_separable:.6f} seconds")
print(f"Maximum pixel difference: {max_diff}")

# Display comparison
titles = ["Original", "Direct Gaussian", "Separable Gaussian", "Difference"]
images = [img, direct, separable, difference]

plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()


# ==================== ASSIGNMENT 5 ====================

# Add salt & pepper noise manually
sp_noise = img.copy()

coords = np.random.randint(0, img.size, 500)
sp_noise.flat[coords] = 255  # Salt

coords = np.random.randint(0, img.size, 500)
sp_noise.flat[coords] = 0    # Pepper

# Mean filter (not ideal for this noise)
mean = cv2.blur(sp_noise, (5, 5))

# Median filter (best for salt & pepper noise)
median = cv2.medianBlur(sp_noise, 5)

# Display results
images = [sp_noise, mean, median]
titles = ["Salt & Pepper Noise", "Mean Filter", "Median Filter"]

plt.figure(figsize=(9, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
