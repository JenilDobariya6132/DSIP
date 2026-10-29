
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image in grayscale
image = cv2.imread(r"D:\DSIP\Experiment-11\1.jpg", 0)

if image is None:
    print("Error: Image not found. Check the image path.")
    exit()

# 2. Define the kernel
kernel = np.ones((5, 5), np.uint8)

# 3. Perform morphological operations
erosion = cv2.erode(image, kernel, iterations=1)
dilation = cv2.dilate(image, kernel, iterations=1)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# 4. Display all results in one figure
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(erosion, cmap='gray')
plt.title("Erosion")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(dilation, cmap='gray')
plt.title("Dilation")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(opening, cmap='gray')
plt.title("Opening")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(closing, cmap='gray')
plt.title("Closing")
plt.axis("off")

plt.tight_layout()
plt.show()
