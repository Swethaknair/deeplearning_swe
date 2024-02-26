import cv2
import numpy as np

# Load an image
image = cv2.imread(r'C:\Users\HP\OneDrive\Pictures\WhatsApp Image 2023-10-16 at 8.26.58 AM.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply noise removal (you can use other filters as needed)
denoised = cv2.medianBlur(gray, 5)

# Apply thresholding
ret, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Sure background area
sure_bg = cv2.dilate(thresh, None, iterations=3)

# Marker labelling
_, markers = cv2.connectedComponents(sure_bg)

# Apply watershed algorithm
markers = markers + 1
markers[thresh == 255] = 0
markers = cv2.watershed(image, markers)

# Draw contours on the original image
image[markers == -1] = [0, 0, 255]  # Mark watershed boundaries

# Display the result
cv2.imshow('Segmented Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
