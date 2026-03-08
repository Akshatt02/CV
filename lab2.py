import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Ques 1
negative = 255 - gray

# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1); plt.imshow(gray, cmap='gray'); plt.title("Original")
# plt.subplot(1,2,2); plt.imshow(negative, cmap='gray'); plt.title("Negative")
# plt.show()

# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1); plt.hist(gray.ravel(), 256)
# plt.subplot(1,2,2); plt.hist(negative.ravel(), 256)
# plt.show()

# Ques 2
def log_transform(img, c):
    img_float = img.astype(np.float32)
    log_img = c * np.log(1 + img_float)
    log_img = log_img / log_img.max() * 255
    return log_img.astype(np.uint8)

log1 = log_transform(gray, 20)
log2 = log_transform(gray, 40)
log3 = log_transform(gray, 60)

# plt.figure(figsize=(10,4))
# plt.subplot(1,4,1); plt.imshow(gray, cmap='gray')
# plt.subplot(1,4,2); plt.imshow(log1, cmap='gray')
# plt.subplot(1,4,3); plt.imshow(log2, cmap='gray')
# plt.subplot(1,4,4); plt.imshow(log3, cmap='gray')
# plt.show()

# Ques 3
def gamma_transform(img, gamma):
    img_norm = img / 255.0
    gamma_img = np.power(img_norm, gamma)
    return (gamma_img * 255).astype(np.uint8)

g1 = gamma_transform(gray, 0.5)
g2 = gamma_transform(gray, 1.0)
g3 = gamma_transform(gray, 2.0)

# plt.figure(figsize=(10,4))
# plt.subplot(1,4,1); plt.imshow(gray, cmap='gray'); plt.title("Original")
# plt.subplot(1,4,2); plt.imshow(g1, cmap='gray'); plt.title("0.5")
# plt.subplot(1,4,3); plt.imshow(g2, cmap='gray'); plt.title("1.0")
# plt.subplot(1,4,4); plt.imshow(g3, cmap='gray'); plt.title("2.0")
# plt.show()

# Ques 4
neg = 255 - gray
log_img = log_transform(gray, 40)
gamma05 = gamma_transform(gray, 0.5)
gamma2 = gamma_transform(gray, 2.0)

# plt.figure(figsize=(10,6))
# images = [gray, neg, log_img, gamma05, gamma2]
# titles = ["Original", "Negative", "Log", "Gamma 0.5", "Gamma 2"]

# for i in range(5):
#     plt.subplot(5,1,i+1)
#     plt.imshow(images[i], cmap='gray')
#     plt.title(titles[i])

# plt.show()

# Ques 5
# plt.hist(gray.ravel(), 256)
# plt.title("Original")
# plt.show()

# plt.hist(gamma05.ravel(), 256)
# plt.title("Gamma 0.5")
# plt.show()

# Ques 6
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

v_new = gamma_transform(v, 2)

hsv_new = cv2.merge([h, s, v_new])
result = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
# plt.show()

# Ques 7
slice_img = gray.copy()

slice_img[(gray >= 100) & (gray <= 150)] = 255
slice_img[(gray < 100) | (gray > 150)] = 0

# plt.imshow(slice_img, cmap='gray')
# plt.show()

# Ques 8
low_light = gray

log_img = log_transform(low_light, 40)
contrast = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX) # different

gamma_neg = 255 - gamma_transform(low_light, 0.5)

neg_gamma = gamma_transform(255 - low_light, 2.0)

# plt.figure(figsize=(8,4))
# plt.subplot(1,3,1); plt.imshow(contrast, cmap='gray')
# plt.subplot(1,3,2); plt.imshow(gamma_neg, cmap='gray')
# plt.subplot(1,3,3); plt.imshow(neg_gamma, cmap='gray')
# plt.show()

# Ques 9
# gaussian noise
noise = np.random.normal(0, 20, gray.shape)
noisy = gray + noise
noisy = np.clip(noisy, 0, 255).astype(np.uint8)

# plt.imshow(noisy, cmap='gray')
# plt.show()

# salt & pepper
sp = gray.copy()
prob = 0.02
mask = np.random.rand(*gray.shape)
sp[mask < prob] = 0
sp[mask > 1 - prob] = 255


# Ques 10
img_4bit = (gray // 16) * 16
# plt.imshow(img_4bit, cmap='gray')
# plt.show()

# Ques 11
np.mean(gray) # Mean
np.std(gray) # Contrast
def entropy(img):
    hist, _ = np.histogram(img.ravel(), 256)
    p = hist / hist.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

