import cv2
import numpy as np

# Load the binary image
image = cv2.imread(r'D:\DSIP\Experiment-12\image.jpg', 0)

# Define the structuring element for the pattern to be detected (0 for don't care, 1 for foreground, -1 for background)
pattern = np.array([[-1, 1, -1], [1, 1, 1], [-1, 1, -1]], dtype=np.int8)

# Apply the Hit or Miss Transformation
result = cv2.morphologyEx(image, cv2.MORPH_HITMISS, pattern)

# Display the original image and the result
cv2.imshow('Original Image', image)
cv2.imshow('Hit or Miss Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()