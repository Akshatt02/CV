# Import OpenCV for image processing
import cv2  

# Import matplotlib for displaying images (since cv2.imshow doesn't work well in notebooks)
import matplotlib.pyplot as plt  


"""==================== 1. Reading and Displaying an Image ===================="""  

# Read image from disk
# NOTE: OpenCV loads image in BGR format (not RGB)
img = cv2.imread("/content/image.jpg")

# Display image using matplotlib
# But matplotlib expects RGB, so colors may look wrong here
plt.imshow(img)
plt.title("Flower image displayed")
plt.show()


"""==================== 2. Understanding Image Properties ===================="""  

# shape = (height, width, channels)
print("Image shape: ", img.shape)

# dtype tells how pixel values are stored (usually uint8 → range 0–255)
print("Data type: ", img.dtype)


"""==================== 3. Splitting Color Channels ===================="""  

# Split into Blue, Green, Red channels
# OpenCV stores as BGR (important!)
b, g, r = cv2.split(img)

# Visualize each channel separately
plt.figure(figsize=(12,4))

# Blue channel
plt.subplot(1,3,1)
plt.imshow(b, cmap='gray')  # show as grayscale
plt.title("Blue channel")

# Green channel
plt.subplot(1,3,2)
plt.imshow(g, cmap='gray')
plt.title("Green channel")

# Red channel
plt.subplot(1,3,3)
plt.imshow(r, cmap='gray')
plt.title("Red channel")

plt.show()


"""==================== 4. Accessing Pixel Values ===================="""  

# Coordinates (x = column, y = row)
x, y = 150, 500

# Access pixel → returns [B, G, R]
pixel = img[y, x]

print("Pixel value at (150,500): ", pixel)


"""==================== 5. Modifying Image Region ===================="""  

# Copy image so original is not modified
img_modified = img.copy()

# Set a rectangular region to white
# Why? → demonstrates region-based editing
img_modified[1207:1727, 320:1520] = [255, 255, 255]

# Display modified image
plt.imshow(img_modified)
plt.show()

# Convert BGR → RGB for correct colors in matplotlib
plt.imshow(cv2.cvtColor(img_modified, cv2.COLOR_BGR2RGB))


"""==================== 6. Creating Image from Scratch ===================="""  

import numpy as np

# Create blank image (all zeros → black)
blankimg = np.zeros((1024,1024,3), dtype=np.uint8)

# Fill entire image with color (BGR)
# (255, 0, 255) → Purple
blankimg[:] = 255, 0, 255

plt.imshow(blankimg)
plt.show()


"""==================== 7. Grayscale Conversion ===================="""  

# Convert BGR → Grayscale
# Why? → simplifies processing (1 channel instead of 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.title("Gray image")
plt.show()


"""==================== ASSIGNMENT Q1 ===================="""  

# Access another pixel
x, y = 277, 390
pixel = img[y, x]

print("Pixel value at (277,390): ", pixel)


"""==================== ASSIGNMENT Q2 ===================="""  

img_modify = img.copy()

# Get dimensions
h, w, _ = img.shape

# Extract top half of image
region = img_modify[0:h//2, 0:w]

# Convert only that region to grayscale
grr = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

# Convert back to BGR (so it fits in original image structure)
gray_region_bgr = cv2.cvtColor(grr, cv2.COLOR_GRAY2BGR)

# Replace top half with grayscale version
img_modify[0:h//2, 0:w] = gray_region_bgr

plt.imshow(cv2.cvtColor(img_modify, cv2.COLOR_BGR2RGB))
plt.show()


"""==================== ASSIGNMENT Q3 ===================="""  

# Convert to HSV (Hue, Saturation, Value)
# Why? → easier to manipulate brightness (V channel)
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split channels
h, s, v = cv2.split(hsv_image)

# Increase brightness
# cv2.add prevents overflow issues (unlike v + 150)
v = cv2.add(v, 150)

# Merge channels back
final_hsv = cv2.merge((h, s, v))

# Convert back to BGR for display
final_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

plt.imshow(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))


"""==================== ASSIGNMENT Q4 ===================="""  

# Find center of image
center_x, center_y = blankimg.shape[1] // 2, blankimg.shape[0] // 2

# Define square size
square_size = 200
half_size = square_size // 2

# Calculate square boundaries
start_x = center_x - half_size
end_x = center_x + half_size
start_y = center_y - half_size
end_y = center_y + half_size

# Draw red square (BGR → [255,0,0] = Blue actually in OpenCV!)
blankimg[start_y:end_y, start_x:end_x] = [255, 0, 0]

plt.imshow(blankimg)
plt.title("Blank Image with Red Square")
plt.show()


"""==================== ASSIGNMENT Q5 ===================="""  

# Swap Red and Blue channels
b, g, r = cv2.split(img)

# Merge in swapped order
swapped_img = cv2.merge([r, g, b])

plt.imshow(cv2.cvtColor(swapped_img, cv2.COLOR_BGR2RGB))
plt.title("Image with Red and Blue Channels Swapped")
plt.show()


"""==================== SECOND PART ===================="""  

# Read another image
img = cv2.imread('landscape.jpg')

# Save image to disk
cv2.imwrite('output.png', img)


"""==================== IMAGE INFO ===================="""  

height, width = img.shape[:2]

# If color → 3 channels, else 1
channels = img.shape[2] if len(img.shape) == 3 else 1

data_type = img.dtype

print(f"Dimensions: Height={height}, Width={width}")
print(f"Channels: {channels}")
print(f"Data Type: {data_type}")


"""==================== RESIZING ===================="""  

# Resize image to fixed size
# INTER_AREA → best for shrinking images
resized_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)


"""==================== SCALING ===================="""  

# Reduce size to half
scaled_img = cv2.resize(img, (width//2, height//2), interpolation=cv2.INTER_AREA)


"""==================== GRAYSCALE ===================="""  

grayscale_img_q5 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


"""==================== GRAYSCALE → BGR ===================="""  

# Convert back to 3-channel image
reconverted_bgr_img = cv2.cvtColor(grayscale_img_q5, cv2.COLOR_GRAY2BGR)


"""==================== HISTOGRAM ===================="""  

# Flatten image and plot intensity distribution
plt.hist(grayscale_img_q7.ravel(), 256, [0, 256])


"""==================== NORMALIZATION ===================="""  

# Convert to float for precision
normalized_img = img.astype(np.float32)

# Scale values from [0,255] → [0,1]
normalized_img = normalized_img / 255.0


"""==================== NEGATIVE IMAGE ===================="""  

# Invert colors
negative_img = 255 - img


"""==================== THRESHOLDING ===================="""  

# Convert to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to binary image
# Pixels >127 → 255 (white), else 0 (black)
_, binary_img = cv2.threshold(grayscale_img, 127, 255, cv2.THRESH_BINARY)