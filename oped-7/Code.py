import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Contrast Stretching

def contrast_stretching(image):

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val:
        return image

    result = ((image.astype(np.float32) - min_val) /
              (max_val - min_val)) * 255

    return result.astype(np.uint8)

# 2. CLAHE Enhancement

def clahe_enhancement(image):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(image)


# 3. Gamma Correction

def gamma_correction(image, gamma):

    normalized = image.astype(np.float32) / 255.0

    result = np.power(normalized, gamma)

    result = result * 255

    return result.astype(np.uint8)


# 4. Log Transformation

def log_transformation(image):

    image_float = image.astype(np.float32)

    maximum = np.max(image_float)

    if maximum == 0:
        return image

    c = 255 / np.log(1 + maximum)

    result = c * np.log(1 + image_float)

    return result.astype(np.uint8)

# 5. Negative Transformation

def negative_transformation(image):

    return 255 - image


image_names = [
    "IMG_1.jpg",
    "IMG_2.jpg",
    "IMG_3.jpg",
    "IMG_4.jpg",
    "IMG_5.jpg"
]


for i, name in enumerate(image_names):

    image = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("ERROR: Cannot find image:", name)
        print("Please check the image name and location.")
        print()
        continue


    if i == 0:

        enhanced = contrast_stretching(image)
        technique = "Contrast Stretching"


    elif i == 1:

        enhanced = clahe_enhancement(image)
        technique = "CLAHE"


    elif i == 2:

        enhanced = gamma_correction(image, 0.4)
        technique = "Gamma Correction"


    elif i == 3:

        enhanced = log_transformation(image)
        technique = "Log Transformation"


    else:

        enhanced = negative_transformation(image)
        technique = "Negative Transformation"

    plt.figure(figsize=(10, 5))


    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")


    # Enhanced image
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced, cmap="gray")
    plt.title(technique)
    plt.axis("off")


    plt.tight_layout()
    plt.show()


    output_name = "enhanced_" + str(i + 1) + ".jpg"

    success = cv2.imwrite(output_name, enhanced)


    if success:

        print("----------------------------------------")
        print("Image     :", name)
        print("Technique :", technique)
        print("Output    :", output_name)
        print("----------------------------------------")
        print()

    else:

        print("ERROR: Could not save", output_name)