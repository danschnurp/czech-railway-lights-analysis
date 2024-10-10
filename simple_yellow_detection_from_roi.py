import cv2
import numpy as np

# Load the image
image = cv2.imread('dataset/reconstructed/Cabview 16  Pardubice-Týniště n O-Hradec Králové  točna a depo  strojvedoucicom/yolov5mu/traffic light/2878.867_roi0.jpg')


# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Define the range for yellow colors
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Define the range for orange colors
lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])

# Create a mask for orange areas
orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

# Bitwise-AND mask and original image
orange_result = cv2.bitwise_and(image, image, mask=orange_mask)

# Create a mask for yellow areas
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Bitwise-AND mask and original image
yellow_result = cv2.bitwise_and(image, image, mask=yellow_mask)

cv2.imshow('Orange Areas', orange_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
