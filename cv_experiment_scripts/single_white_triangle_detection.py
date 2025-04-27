import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.image_utils import calculate_nonzero_percent


def detect_white_triangles(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image with specified values
    _, thresh = cv2.threshold(gray, 90, 250, cv2.THRESH_BINARY)

    # # Apply morphological operations
    # kernel = np.ones((3, 3), np.uint8)
    # thresh = cv2.erode(thresh, kernel, iterations=1)  # Erosion to remove noise
    # thresh = cv2.dilate(thresh, kernel, iterations=2)  # Dilation to restore shapes

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy the original image to draw results
    result_image = image.copy()

    for contour in contours:
        # Approximate the contour to simplify its shape
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the approximated contour has 3 vertices (triangle)
        if len(approx) == 3:
            # Draw the triangle on the result image
            cv2.drawContours(result_image, [approx], 0, (0, 0, 255), 3)
    _ ,result_image = cv2.threshold(result_image, 254, 255, cv2.THRESH_BINARY)
    res =calculate_nonzero_percent(  result_image  )
    # Display the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Processed Threshold Image")
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Detected White Triangles {res}")
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Example usage
detect_white_triangles("Screenshot 2024-11-16 at 19.43.09.png")
detect_white_triangles("464.167_box.jpg")
detect_white_triangles("Screenshot 2024-11-16 at 20.34.56.png")
