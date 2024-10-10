import cv2
import numpy as np

# Load the image
# image = cv2.imread('dataset/reconstructed/Cabview 16  Pardubice-Týniště n O-Hradec Králové  točna a depo  strojvedoucicom/yolov5mu/traffic light/2878.867_roi0.jpg')
image = cv2.imread("dataset/reconstructed/Cabview 16  Pardubice-Týniště n O-Hradec Králové  točna a depo  strojvedoucicom/yolov5mu/traffic light/92.950_roi0.jpg")
# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the ranges for red colors
lower_red_1 = np.array([0, 100, 100])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([160, 100, 100])
upper_red_2 = np.array([180, 255, 255])

# Create masks for red areas
mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

# Combine the masks
red_mask = mask1 | mask2

# Bitwise-AND mask and original image
result = cv2.bitwise_and(image, image, mask=red_mask)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Red Areas', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
