import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image (change path)
img = cv2.imread("image.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----- Sobel -----
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
sobel = np.sqrt(sobelx**2 + sobely**2)

# ----- Prewitt -----
kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

prewittx = cv2.filter2D(gray, -1, kernelx)
prewitty = cv2.filter2D(gray, -1, kernely)
prewitt = np.sqrt(prewittx**2 + prewitty**2)

# Show results
plt.imshow(sobel, cmap='gray')
plt.title("Sobel")
plt.show()

plt.imshow(prewitt, cmap='gray')
plt.title("Prewitt")
plt.show()