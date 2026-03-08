import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

def show(img, title=""):
  plt.figure(figsize=(4,4))
  plt.imshow(img, cmap='gray')
  plt.title(title)
  plt.axis('off')
  plt.show()

"""ASSIGNMENT 1"""

img = cv2.imread('sample.jpg', 0) / 255.0
noisy = random_noise(img, mode='gaussian', mean=0, var=0.05)
noisy = (noisy * 255).astype(np.uint8)
filtered = cv2.blur(noisy, (5,5))

plt.figure(figsize=(4,4))
plt.subplot(1,2,1); plt.imshow(noisy, cmap='gray'); plt.title("Noisy Image"); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(filtered, cmap='gray'); plt.title("Filtered Image"); plt.axis('off')
plt.show()