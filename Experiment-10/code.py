import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read image using image path
img = cv2.imread("D:\\DSIP\\Experiment-10\\IMG_2090.PNG", 0)

# Check whether image is loaded successfully
if img is None:
    print("Error: Image not found. Please check the image path.")
else:
    # 2. Fourier Transform
    F = np.fft.fftshift(np.fft.fft2(img))
    M, N = img.shape

    # 3. Create frequency distance matrix
    u, v = np.meshgrid(
        np.arange(N) - N // 2,
        np.arange(M) - M // 2
    )
    D = np.sqrt(u**2 + v**2)

    # 4. Create Low-Pass and High-Pass Filters
    LPF = (D < 30)
    HPF = (D >= 30)

    # 5. Apply Low-Pass Filter
    F_LPF = F * LPF
    smooth = np.abs(
        np.fft.ifft2(np.fft.ifftshift(F_LPF))
    )

    # 6. Apply High-Pass Filter
    F_HPF = F * HPF
    high_pass = np.abs(
        np.fft.ifft2(np.fft.ifftshift(F_HPF))
    )

    # 7. Sharpen the original image
    sharp = np.clip(
        img.astype(float) + high_pass,
        0, 255
    ).astype(np.uint8)

    # 8. Display Original, Smooth and Sharpened Images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(smooth, cmap='gray')
    plt.title('LPF - Smoothening')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(sharp, cmap='gray')
    plt.title('HPF - Sharpening')
    plt.axis('off')

    plt.tight_layout()
    plt.show()