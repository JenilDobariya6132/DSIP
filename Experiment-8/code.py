import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = "E:\DSIP\Experiment-8\enhanced_1.jpg"

# Read image in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check whether image was loaded successfully
if image is None:
    print("Error: Image not found. Check the image path.")
else:
    # -----------------------------------
    # 1. Calculate Original Histogram
    # -----------------------------------
    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    )

    # Plot original histogram
    plt.figure(figsize=(8, 6))
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.grid(True)
    plt.show()

    # -----------------------------------
    # 2. Histogram Equalization
    # -----------------------------------
    equalized_image = cv2.equalizeHist(image)

    # -----------------------------------
    # 3. Display Original & Equalized
    # -----------------------------------
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Equalized Image')
    plt.imshow(equalized_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # -----------------------------------
    # 4. Calculate Equalized Histogram
    # -----------------------------------
    equalized_histogram = cv2.calcHist(
        [equalized_image],
        [0],
        None,
        [256],
        [0, 256]
    )

    # Plot equalized histogram
    plt.figure(figsize=(8, 6))
    plt.title('Equalized Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(equalized_histogram)
    plt.xlim([0, 256])
    plt.grid(True)
    plt.show()