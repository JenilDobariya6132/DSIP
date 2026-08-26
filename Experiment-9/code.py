import cv2
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Smoothing Filter
# -------------------------------
def apply_smoothing_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))


# -------------------------------
# Sharpening Filter
# -------------------------------
def apply_sharpening_filter(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(image, -1, kernel)


# -------------------------------
# Load Image
# -------------------------------
image_path = "E:\DSIP\Experiment-9\IMG_1.jpg"

input_image = cv2.imread(image_path)

if input_image is None:
    print("Error: IMG_2.jpg not found!")
    exit()


# -------------------------------
# Apply Filters
# -------------------------------
smoothed_image = apply_smoothing_filter(
    input_image,
    kernel_size=5
)

sharpened_image = apply_sharpening_filter(
    input_image
)


# -------------------------------
# Convert BGR to RGB
# Matplotlib uses RGB
# -------------------------------
input_rgb = cv2.cvtColor(
    input_image,
    cv2.COLOR_BGR2RGB
)

smoothed_rgb = cv2.cvtColor(
    smoothed_image,
    cv2.COLOR_BGR2RGB
)

sharpened_rgb = cv2.cvtColor(
    sharpened_image,
    cv2.COLOR_BGR2RGB
)


# -------------------------------
# Display Images
# -------------------------------
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 3, 1)
plt.imshow(input_rgb)
plt.title("Original Image")
plt.axis("off")

# Smoothed
plt.subplot(1, 3, 2)
plt.imshow(smoothed_rgb)
plt.title("Smoothed Image")
plt.axis("off")

# Sharpened
plt.subplot(1, 3, 3)
plt.imshow(sharpened_rgb)
plt.title("Sharpened Image")
plt.axis("off")

# Adjust spacing
plt.tight_layout()

# Show output
plt.show()


# -------------------------------
# Save Images
# -------------------------------
cv2.imwrite(
    "smoothed_image.jpg",
    smoothed_image
)

cv2.imwrite(
    "sharpened_image.jpg",
    sharpened_image
)

print("Smoothed image saved as: smoothed_image.jpg")
print("Sharpened image saved as: sharpened_image.jpg")