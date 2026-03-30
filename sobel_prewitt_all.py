import cv2                          # OpenCV for image handling
import numpy as np                 # Numerical operations

# ===================== LOAD IMAGE =====================

img = cv2.imread('image.jpg', 0)   # Load image in grayscale

# ===================== GENERIC FILTER FUNCTION =====================

def apply_filter(image, mask):
    """
    Applies a 3x3 mask manually using loops
    """
    output = np.zeros_like(image)  # Create empty output image

    # Loop over image (avoid borders)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):

            region = image[i-1:i+2, j-1:j+2]  # Extract 3x3 region
            value = np.sum(region * mask)     # Multiply & sum

            output[i, j] = abs(value)         # Store absolute value

    return output

# ===================== 1. DIAGONAL POINT DETECTION =====================

diag_mask = np.array([[1, 0, -1],
                      [0, 0, 0],
                      [-1, 0, 1]])

diag_output = apply_filter(img, diag_mask)

# ===================== 2. HORIZONTAL LINE DETECTION =====================

horiz_mask = np.array([[-1, -1, -1],
                       [ 2,  2,  2],
                       [-1, -1, -1]])

horiz_output = apply_filter(img, horiz_mask)

# ===================== 3. VERTICAL LINE DETECTION =====================

vert_mask = np.array([[-1, 2, -1],
                      [-1, 2, -1],
                      [-1, 2, -1]])

vert_output = apply_filter(img, vert_mask)

# ===================== 4. SOBEL OPERATOR =====================

sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

sobel_output = np.zeros_like(img)

for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):

        region = img[i-1:i+2, j-1:j+2]

        gx = np.sum(region * sobel_x)   # Gradient in x-direction
        gy = np.sum(region * sobel_y)   # Gradient in y-direction

        sobel_output[i, j] = np.sqrt(gx**2 + gy**2)  # Magnitude

# ===================== 5. PREWITT OPERATOR =====================

prewitt_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewitt_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

prewitt_output = np.zeros_like(img)

for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):

        region = img[i-1:i+2, j-1:j+2]

        gx = np.sum(region * prewitt_x)
        gy = np.sum(region * prewitt_y)

        prewitt_output[i, j] = np.sqrt(gx**2 + gy**2)

# ===================== 6. CANNY EDGE DETECTION =====================

canny_edges = cv2.Canny(img, 100, 200)  # Built-in multi-stage edge detector

# ===================== 7. GABOR FILTER =====================

kernel = cv2.getGaborKernel((5,5), 1.0, 0, 10, 0.5)  # Create Gabor kernel

gabor_output = np.zeros_like(img)

for i in range(2, img.shape[0]-2):
    for j in range(2, img.shape[1]-2):

        region = img[i-2:i+3, j-2:j+3]
        gabor_output[i, j] = np.sum(region * kernel)

# ===================== 8. SHARPENING FILTER =====================

sharpen_mask = np.array([[0,-1,0],
                         [-1,5,-1],
                         [0,-1,0]])

sharpen_output = apply_filter(img, sharpen_mask)

# ===================== 9. NOISE + BLUR + EDGE =====================

noise = np.random.normal(0, 25, img.shape)        # Add Gaussian noise
noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)

blurred = cv2.GaussianBlur(noisy_img, (5,5), 0)   # Blur image

edges_after_blur = cv2.Canny(blurred, 100, 200)   # Edge detection

# ===================== 10. COUNT EDGE PIXELS =====================

edge_count = np.sum(canny_edges > 0)              # Count non-zero edge pixels
print("Edge Pixel Count:", edge_count)

# ===================== 11. LAPLACIAN AFTER GAUSSIAN =====================

blur = cv2.GaussianBlur(img, (5,5), 0)

laplacian_mask = np.array([[0,1,0],
                           [1,-4,1],
                           [0,1,0]])

laplacian_output = apply_filter(blur, laplacian_mask)

# ===================== 12. DISPLAY ONLY EDGES =====================

cv2.imshow("Edges Only", canny_edges)

# ===================== 13. BOUNDING BOXES =====================

contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)  # Get bounding box
    cv2.rectangle(img_color, (x,y), (x+w,y+h), (0,255,0), 2)

cv2.imshow("Bounding Boxes", img_color)

# ===================== DISPLAY RESULTS =====================

cv2.imshow("Original", img)
cv2.imshow("Diagonal", diag_output)
cv2.imshow("Horizontal", horiz_output)
cv2.imshow("Vertical", vert_output)
cv2.imshow("Sobel", sobel_output)
cv2.imshow("Prewitt", prewitt_output)
cv2.imshow("Gabor", gabor_output)
cv2.imshow("Sharpen", sharpen_output)
cv2.imshow("Laplacian", laplacian_output)

cv2.waitKey(0)
cv2.destroyAllWindows()