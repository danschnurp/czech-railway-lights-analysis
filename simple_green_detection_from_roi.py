import cv2
import numpy as np


# Example usage
image = cv2.imread('dataset/reconstructed/Cabview 16  Pardubice-Týniště n O-Hradec Králové  točna a depo  '
                   'strojvedoucicom/yolov5mu/traffic light/2878.867_roi0.jpg')

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for green colors
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

# Create a mask for green areas
mask = cv2.inRange(hsv, lower_green, upper_green)

# Bitwise-AND mask and original image
result = cv2.bitwise_and(image, image, mask=mask)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Green Areas', result)
cv2.waitKey(0)
cv2.destroyAllWindows()



# detected_circles = detect_colored_circles(image)
# print(detected_circles)