import cv2
import numpy as np


def detect_colored_circles(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges
    color_ranges = {
        'red': ([0, 100, 100], [10, 255, 255]),
        'green': ([40, 100, 100], [80, 255, 255]),
        'white': ([0, 0, 200], [180, 30, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255])
    }

    results = {}

    for color, (lower, upper) in color_ranges.items():
        # Create a mask for the color
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Find circles
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0, maxRadius=30)

        if circles is not None:
            results[color] = len(circles[0])

    return results


# Example usage
image = cv2.imread('railway_light.jpg')
detected_circles = detect_colored_circles(image)
print(detected_circles)