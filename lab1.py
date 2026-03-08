import cv2
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg")

# 1) Change BGR to RGB
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# 2) Print Shape and dtype
# print(img.shape)
# print(img.dtype)

# 3) Split channels
b, g, r = cv2.split(img)
# plt.imshow(b, cmap='gray')
# plt.show()

# 4) Access any pixel
h, w, _ = img.shape
center = img[h//2, w//2]
# print(center)

# 5) BGR to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray.shape)

# 6) Metadata
h, w, c = img.shape
# print("Height:", h)
# print("Width:", w)
# print("Channels:", c)
# print("Datatype:", img.dtype)

# 7) Resize
resized = cv2.resize(img, (300,300)) # Remember cv2.resize(img, (width, height))

# 8) Scale 50%
h, w = img.shape[:2]
scaled = cv2.resize(img, (w//2, h//2))
# plt.imshow(scaled)
# plt.show()
# print(img.shape)
# print(scaled.shape)

# 9) BGR to Gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray.shape)

# 10) Histogram
# plt.hist(gray.ravel(), bins=256)
# plt.xlabel("Pixel Value")
# plt.ylabel("Frequency")
# plt.show()

# 11) Normalisation
img_float = img.astype("float32")
img_norm = img_float / 255.0

# print(img_norm.min(), img_norm.max())

# 12) Negative
negative = 255 - img

# 13) Binary
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

binary = gray.copy()
binary[binary > 127] = 255
binary[binary <= 127] = 0
# plt.imshow(binary)
# plt.show()