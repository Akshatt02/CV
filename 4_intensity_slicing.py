# Assignment 7: Intensity Slicing (Highlighting Specific Ranges)
# Objective: Learn to highlight specific intensity ranges while suppressing others.

# ---------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------

# numpy: used for efficient numeric array operations and vectorized masking
import numpy as np
# matplotlib.pyplot: used for displaying images and plotting histograms
import matplotlib.pyplot as plt
# skimage.data: contains sample images for testing (like brain or astronaut)
from skimage import data, img_as_ubyte
# skimage.color: used for converting RGB images to grayscale
from skimage.color import rgb2gray
# skimage.exposure: used for intensity transformations and histogram utilities
from skimage import exposure

# ---------------------------------------------------------
# LOAD SAMPLE IMAGES (MEDICAL AND LANDSCAPE)
# ---------------------------------------------------------

# Load a medical image from skimage.data.
# data.brain() provides a 3D volume; we use it as a 2D slice or fallback to camera().
try:
    # Try loading the brain dataset
    medical_color = data.brain()
    print('Loaded skimage.data.brain()')
except AttributeError:
    # Fallback if brain data isn't installed or available in this version
    medical_color = data.camera()
    print('skimage.data.brain() unavailable, using camera() as medical placeholder')

# Load a landscape image. data.astronaut() is a common color sample.
landscape_color = data.astronaut()

# Pre-processing: Ensure images are in grayscale for intensity slicing.
# If the medical image is RGB (3 channels), convert it to grayscale.
if medical_color.ndim == 3 and medical_color.shape[2] == 3:
    medical_gray = rgb2gray(medical_color)
else:
    # Otherwise, just normalize the intensities to [0,1] range.
    medical_gray = medical_color / np.max(medical_color)

# Convert the landscape color image to grayscale.
landscape_gray = rgb2gray(landscape_color)

# Convert both images to 8-bit unsigned integers (range 0 to 255) for consistent processing.
medical_gray_ubyte = img_as_ubyte(medical_gray)
landscape_gray_ubyte = img_as_ubyte(landscape_gray)

# Print image metadata to verify dimensions and data types.
print('Medical image shape:', medical_gray_ubyte.shape, 'dtype:', medical_gray_ubyte.dtype)
print('Landscape image shape:', landscape_gray_ubyte.shape, 'dtype:', landscape_gray_ubyte.dtype)

# Display the original grayscale images side-by-side.
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# Show medical image
axes[0].imshow(medical_gray_ubyte, cmap='gray')
axes[0].set_title('Original Medical (Gray)')
axes[0].axis('off') # Hide axes for better visual
# Show landscape image
axes[1].imshow(landscape_gray_ubyte, cmap='gray')
axes[1].set_title('Original Landscape (Gray)')
axes[1].axis('off')
plt.show()

# ---------------------------------------------------------
# DEFINE INTENSITY SLICING FUNCTIONS
# ---------------------------------------------------------

def intensity_slice_gray(img, low, high, gray_level=128):
    """
    Intensity Slicing: Highlights a range by keeping original intensities,
    while setting all other pixels outside the range to a constant gray value.
    
    Parameters:
    - img: 2D numpy array (grayscale image)
    - low: lower bound of the intensity range to highlight
    - high: upper bound of the intensity range to highlight
    - gray_level: the intensity value used for suppression (default is 128)
    """
    # Create an output array filled with the background gray level
    out = np.full_like(img, gray_level, dtype=np.uint8)
    
    # Create a boolean mask where pixel values fall within the range [low, high]
    mask = (img >= low) & (img <= high)
    
    # Apply the mask: only keep original values where the mask is True
    out[mask] = img[mask]
    
    return out

def intensity_slice_bw(img, low, high, low_val=0, high_val=255):
    """
    Intensity Slicing (Binary Background): Highlights a range by keeping original 
    intensities, while setting others to black (0) or white (255).
    
    Parameters:
    - img: 2D numpy array
    - low, high: intensity range boundaries
    - low_val: value for pixels below the 'low' threshold
    - high_val: value for pixels above the 'high' threshold
    """
    # Initialize output array with zeros
    out = np.zeros_like(img, dtype=np.uint8)
    
    # Map values below the range to low_val (usually black)
    out[img < low] = low_val
    # Map values above the range to high_val (usually white)
    out[img > high] = high_val
    
    # Create a mask for the range to highlight
    mask = (img >= low) & (img <= high)
    # Preserve original intensities within the specified range
    out[mask] = img[mask]
    
    return out

# ---------------------------------------------------------
# TASK 1 & 2: APPLY SLICING AND DISPLAY RESULTS
# ---------------------------------------------------------

# Define the intensity range to highlight (e.g., 100 to 150)
low_range, high_range = 100, 150

# 1. Apply intensity slicing with gray background
med_gray_out = intensity_slice_gray(medical_gray_ubyte, low_range, high_range, gray_level=120)
land_gray_out = intensity_slice_gray(landscape_gray_ubyte, low_range, high_range, gray_level=120)

# 2. Apply intensity slicing with black/white background
med_bw_out = intensity_slice_bw(medical_gray_ubyte, low_range, high_range, low_val=0, high_val=255)
land_bw_out = intensity_slice_bw(landscape_gray_ubyte, low_range, high_range, low_val=0, high_val=255)

# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------

# Helper function to plot image and its histogram with range markers
def plot_with_histogram(img, title, low, high, ax_img, ax_hist):
    # Display the image
    ax_img.imshow(img, cmap='gray')
    ax_img.set_title(title)
    ax_img.axis('off')

    # Plot the histogram of pixel intensities
    ax_hist.hist(img.ravel(), bins=256, range=(0,255), color='tab:blue', alpha=0.7)
    # Add vertical lines to mark the slicing range
    ax_hist.axvline(low, color='red', linestyle='--', label=f'Low: {low}')
    ax_hist.axvline(high, color='green', linestyle='--', label=f'High: {high}')
    ax_hist.set_xlim(0,255)
    ax_hist.set_xlabel('Intensity Value')
    ax_hist.set_ylabel('Pixel Count')
    ax_hist.legend()

# Create a multi-plot figure to compare all results
fig, axs = plt.subplots(4, 2, figsize=(14, 18))

# Row 1: Original Medical Image and Histogram
plot_with_histogram(medical_gray_ubyte, 'Original Medical', low_range, high_range, axs[0,0], axs[0,1])
# Row 2: Medical Slice (Gray Background)
plot_with_histogram(med_gray_out, 'Medical Slice (Gray Background)', low_range, high_range, axs[1,0], axs[1,1])
# Row 3: Medical Slice (B/W Background)
plot_with_histogram(med_bw_out, 'Medical Slice (B/W Background)', low_range, high_range, axs[2,0], axs[2,1])
# Row 4: Original Landscape Image
plot_with_histogram(landscape_gray_ubyte, 'Original Landscape', low_range, high_range, axs[3,0], axs[3,1])

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# COMPARISON WITH SIMPLE THRESHOLDING
# ---------------------------------------------------------

# Simple thresholding collapses the entire range into a single value (binary)
med_thresh = np.zeros_like(medical_gray_ubyte)
med_thresh[(medical_gray_ubyte >= low_range) & (medical_gray_ubyte <= high_range)] = 255

# Display comparison between Slicing and Thresholding
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(med_bw_out, cmap='gray')
axs[0].set_title('Intensity Slicing (Preserves Detail in Band)')
axs[0].axis('off')
axs[1].imshow(med_thresh, cmap='gray')
axs[1].set_title('Simple Thresholding (Binary Result)')
axs[1].axis('off')
plt.show()

# ---------------------------------------------------------
# ANSWERS TO QUESTIONS
# ---------------------------------------------------------

"""
Question 1: What happens to image detail outside the highlighted range?
Answer: 
- In 'Gray Slicing', detail outside the range is completely lost as all pixels are mapped to a 
  single constant gray value, creating a flat background.
- In 'B/W Slicing', detail is also lost, but the contrast at the boundaries of the highlighted 
  regions is extremely high, making the shape of the highlighted area more distinct.

Question 2: How would you choose the intensity range to highlight for a specific application?
Answer:
- You should first inspect the image histogram to see which intensity values correspond to the 
  features of interest (e.g., bone in medical scans usually has higher intensity).
- In medical imaging, you target specific tissue densities.
- In landscape images, you might target the bright pixels of the sky or the mid-tones of vegetation.
- Interactive tools (like sliders) can help fine-tune these thresholds dynamically.

Question 3: Compare this approach with simple thresholding.
Answer:
- Simple thresholding is binary: it maps pixels inside the range to one color (usually white) 
  and outside to another (black). It destroys all texture/detail within the highlighted region.
- Intensity slicing is more sophisticated: it preserves the internal variations and textures 
  of the objects within the highlighted range, while only suppressing the background. 
  This makes it far more useful for clinical or detailed analytical work.
"""
