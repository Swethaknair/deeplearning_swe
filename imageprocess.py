import cv2
import numpy as np

# Load an image
image = cv2.imread(r'C:\Users\HP\OneDrive\Pictures\WhatsApp Image 2023-10-16 at 8.26.58 AM.jpeg')
im1=image
# Convert the image to grayscale
mask = np.zeros(image.shape[:2], np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Define a rectangle around the object of interest (you may need to adjust this)
rect = (50, 50, image.shape[1]-50, image.shape[0]-50)

# Apply GrabCut algorithm
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Modify the mask to create a binary mask for the foreground
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply the binary mask to the original image
result = image * mask2[:, :, np.newaxis]

# Display the result
cv2.imshow('Segmented Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

