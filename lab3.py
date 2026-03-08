import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()

# Ques 1
def convolution2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh//2, kw//2

    padded = np.pad(image, ((pad_h,pad_h),(pad_w,pad_w)), mode='constant')
    output = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i,j] = np.sum(region * kernel)

    return output

# Ques 2
# step 1: add noise
img = cv2.imread("sample.jpg", 0)

noise = np.random.normal(0, 20, img.shape)
gaussian_noisy = np.clip(img + noise, 0, 255).astype(np.uint8)

sp = img.copy()
prob = 0.02
mask = np.random.rand(*img.shape)
sp[mask < prob] = 0
sp[mask > 1-prob] = 255

#step 2
mean3 = cv2.blur(gaussian_noisy, (3,3))
mean5 = cv2.blur(gaussian_noisy, (5,5))

gauss05 = cv2.GaussianBlur(gaussian_noisy, (5,5), 0.5)
gauss1 = cv2.GaussianBlur(gaussian_noisy, (5,5), 1)
gauss2 = cv2.GaussianBlur(gaussian_noisy, (5,5), 2)

# show(gaussian_noisy, "Noisy")
# show(mean5, "Mean 5x5")
# show(gauss1, "Gaussian sigma=1")

# step 3 psnr
def psnr(original, denoised):
    mse = np.mean((original - denoised)**2)
    return 20*np.log10(255/np.sqrt(mse))

# print(psnr(img, mean5))
# print(psnr(img, gauss1))

# Ques 3
# Laplacian Filter
laplacian_kernel = np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])
edges = convolution2d(img, laplacian_kernel)
sharpened = np.clip(img + edges, 0, 255)
# show(img, "Original")
# show(sharpened.astype(np.uint8), "Sharpened")

# Unsharp Masking
blur = cv2.GaussianBlur(img, (5,5), 1)
mask = img - blur
unsharp = np.clip(img + 1.5*mask, 0, 255)

# show(unsharp.astype(np.uint8), "Unsharp")

# Ques 4
import time

img = cv2.imread("sample.jpg", 0)

start = time.time()
direct = cv2.GaussianBlur(img, (5,5), 1)
t1 = time.time() - start

g1d = cv2.getGaussianKernel(5, 1)

start = time.time()
sep = cv2.sepFilter2D(img, -1, g1d, g1d)
t2 = time.time() - start

print("Direct time:", t1)
print("Separable time:", t2)

diff = np.abs(direct.astype(np.float32) - sep.astype(np.float32))
print("Difference mean:", np.mean(diff))

# Ques 5
sp_noise = img.copy()
coords = np.random.randint(0, img.size, 500)
sp_noise.flat[coords] = 255
mean = cv2.blur(sp_noise, (5,5))
median = cv2.medianBlur(sp_noise, 5)