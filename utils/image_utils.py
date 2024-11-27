import time

import numpy as np
from math import gcd

from easyocr import Reader
import cv2


class MovementDetector:

    def __init__(self, frame, refresh_time=5):
        self.reference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.t1 = time.time()
        self.refresh_time = refresh_time

    def detect_movement(self, frame):
        if time.time() - self.t1 > self.refresh_time:
            self.t1 = time.time()
            return True
        # Convert the current frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the difference between the current frame and the reference frame
        diff = cv2.absdiff(gray_frame, self.reference_frame)

        # Threshold the difference image to create a binary mask
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        movement_prob = np.count_nonzero(thresh) / thresh.shape[0] / thresh.shape[1]
        if movement_prob < 0.05:
            print(".........vlak stojí........")
            return False
        return True


def yellow(hsv):
    # Define the range for yellow colors
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for yellow areas
    return cv2.inRange(hsv, lower_yellow, upper_yellow)


def orange(hsv):
    # Define the range for orange colors
    lower_orange = np.array([10, 80, 80])
    upper_orange = np.array([40, 255, 255])
    # Create a mask for orange areas
    return cv2.inRange(hsv, lower_orange, upper_orange)


def yellow_orange(hsv):
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    # Define the range for orange colors
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    return cv2.inRange(hsv, lower_yellow, upper_yellow) | cv2.inRange(hsv, lower_orange, upper_orange)


def red(hsv):
    # Define the ranges for red colors
    lower_red_1 = np.array([0, 100, 100])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([160, 100, 100])
    upper_red_2 = np.array([180, 255, 255])
    # Create masks for red areas
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    # Combine the masks
    return mask1 | mask2


def green(hsv):
    # Define the range for green colors
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    return cv2.inRange(hsv, lower_green, upper_green)


def detect_color(
        image,
        wokrdir="/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed",
        color_filter=red):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(image, image, mask=color_filter(hsv))
    return result


def crop_top_half(img):
    # Get image dimensions
    height, width = img.shape[:2]

    # Crop the top half
    # Starting point: (0,0)
    # Ending point: (width, height//2)
    top_half = img[0:height // 2, 0:width]

    return top_half


def crop_sides_percentage(img, crop_percentage=10):
    # Get image dimensions
    height, width = img.shape[:2]
    # Calculate crop width (10% from each side)
    crop_width = int(width * (crop_percentage / 100))
    # Crop the image
    # Start from crop_width on the left
    # End at width - crop_width on the right
    return img[:, crop_width:width - crop_width]


def enlarge_bounding_box(bounding_box, bigger_top_percents=0.1, bigger_bottoms_percents=0.25):
    """
    The function `enlarge_bounding_box` adjusts the top and bottom coordinates of a bounding box based
    on specified percentage values.

    :param bounding_box: The `bounding_box` parameter represents a rectangle defined by its top-left and
    bottom-right coordinates. In this case, it seems to be represented as a tuple with four values:
    `(top, left, bottom, right)`. The function `enlarge_bounding_box` takes this bounding box and enlarg
    :param bigger_top_percents: The `bigger_top_percents` parameter in the `enlarge_bounding_box`
    function represents the percentage by which the top of the bounding box will be enlarged. It is used
    to calculate the adjustment to increase the top coordinate of the bounding box by a certain
    percentage of the original height
    :param bigger_bottoms_percents: The `bigger_bottoms_percents` parameter in the
    `enlarge_bounding_box` function represents the percentage by which the bottom of the bounding box
    will be enlarged relative to the original height of the bounding box
    :return: The function `enlarge_bounding_box` returns the new top and bottom coordinates of the
    bounding box after enlarging it based on the specified percentage adjustments for the top and
    bottom.
    """
    # enlarging ROI for digits and lines detections
    # Calculate original height
    original_height = bounding_box[3] - bounding_box[1]
    # Calculate 10% of the original height
    adjustment = bigger_top_percents * original_height
    bottom_adjustment = bigger_bottoms_percents * original_height
    # Adjust top and bottom coordinates
    new_top = bounding_box[1] - float(adjustment)
    new_bottom = bounding_box[3] + float(bottom_adjustment)
    return new_top, new_bottom


def calculate_aspect_ratio(img):
    # Get image dimensions
    height, width = img.shape[:2]
    # Calculate the greatest common divisor
    divisor = gcd(width, height)
    # Calculate the simplified ratio
    simple_width = width // divisor
    simple_height = height // divisor
    # Calculate decimal aspect ratio
    decimal_ratio = width / height
    return decimal_ratio, simple_width, simple_height


def check_content_centered(img, tolerance_percentage=20):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    height, width = gray.shape

    # Find non-zero pixels
    non_zero_positions = np.where(gray > 0)[1]  # Get x-coordinates of non-zero pixels

    if len(non_zero_positions) == 0:
        return False, "No non-zero pixels found in the image"

    # Calculate content boundaries
    leftmost = np.min(non_zero_positions)
    rightmost = np.max(non_zero_positions)
    content_center = (leftmost + rightmost) // 2
    image_center = width // 2

    # Calculate the content width
    content_width = rightmost - leftmost

    # Calculate allowed deviation (tolerance zone)
    tolerance = (width * tolerance_percentage) / 100

    # Check if content center is within tolerance of image center
    is_centered = abs(content_center - image_center) <= tolerance

    # Prepare detailed analysis
    report = {
        'image_width': width,
        'image_center': image_center,
        'content_width': content_width,
        'content_left': leftmost,
        'content_right': rightmost,
        'content_center': content_center,
        'deviation': abs(content_center - image_center),
        'tolerance': tolerance,
        'is_centered': is_centered
    }
    # print(report)

    # Create a visualization
    visualization = img.copy()

    # Draw lines for visualization
    # Image center (green)
    cv2.line(visualization, (image_center, 0), (image_center, height), (0, 255, 0), 2)

    # Content center (blue)
    cv2.line(visualization, (content_center, 0), (content_center, height), (255, 0, 0), 2)

    # Tolerance zone (red)
    left_tolerance = int(image_center - tolerance)
    right_tolerance = int(image_center + tolerance)
    cv2.line(visualization, (left_tolerance, 0), (left_tolerance, height), (0, 0, 255), 1)
    cv2.line(visualization, (right_tolerance, 0), (right_tolerance, height), (0, 0, 255), 1)

    return is_centered


def calculate_nonzero_percent(result):
    """
    The function calculates the percentage of non-zero pixels in a given image.

    :param result: It looks like you have provided a code snippet for a function that calculates the
    percentage of non-zero pixels in an image represented by the `result` array. However, you have not
    provided the actual `result` array for which you want to calculate the percentage
    :return: The function `calculate_nonzero_percent` returns the percentage of non-zero pixels in the
    input `result` array.
    """
    # Get total number of pixels
    total_pixels = result.shape[0] * result.shape[1]
    nonzero_mask = np.any(result > 0, axis=2)
    # Count non-zero pixels
    # Using np.count_nonzero() on any channel since mask affects all channels
    nonzero_pixels = np.count_nonzero(nonzero_mask)

    # Calculate percentage
    percent = (nonzero_pixels / total_pixels) * 100

    return percent


def crop_bounding_box(box, img):
    """
    The function `crop_bounding_box` takes a bounding box coordinates and an image, and returns the
    cropped region of interest (ROI) from the image based on the box coordinates.

    :param box: The `box` parameter is a tuple containing the coordinates and dimensions of a bounding
    box. The tuple should have four elements in the order (x, y, width, height), where:
    :param img: The `img` parameter is typically an image represented as a NumPy array. It could be a
    color image (3D array) or a grayscale image (2D array). The function `crop_bounding_box` takes a
    bounding box represented by the `box` parameter and crops the region of
    :return: The function `crop_bounding_box` returns the region of interest (ROI) from the input image
    `img` based on the bounding box coordinates provided in the `box` parameter.
    """
    x, y, w, h = box
    x, y, w, h = int(x), int(y), int(w), int(h)
    roi = img[y:h, x:w]
    return roi


def perform_ocr(reader: Reader, frame: np.ndarray, confidence_threshold=0.1):
    """
    The `perform_ocr` function takes an image frame, performs optical character recognition using a
    reader object, and displays the recognized text with bounding boxes if the confidence level is above
    a specified threshold.

    :param reader: The `reader` parameter in the `perform_ocr` function is expected to be an object of
    type `Reader`. This object likely contains the functionality to perform optical character
    recognition (OCR) on an image
    :type reader: Reader
    :param frame: The `frame` parameter in the `perform_ocr` function is expected to be a NumPy array
    representing an image frame. This image frame will be processed by the OCR (Optical Character
    Recognition) reader to extract text information. The function will then draw bounding boxes around
    the recognized text regions and
    :type frame: np.ndarray
    :param confidence_threshold: The `confidence_threshold` parameter in the `perform_ocr` function is
    used to filter out OCR results based on their confidence level. OCR engines provide a confidence
    score for each recognized text, indicating how confident the engine is in the accuracy of the
    recognition
    :return: The function `perform_ocr` is returning the processed frame with bounding boxes drawn
    around the detected text regions along with the text and confidence level displayed on the frame.
    """
    results = reader.readtext(frame)
    # Process and display results
    for (bbox, text, prob) in results:
        index_for_save = 0
        if prob > confidence_threshold:
            index_for_save += 1
            # Unpack the bounding box
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            # Draw the bounding box
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            # Put the text and probability
            cv2.putText(frame, f"{text}", (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            print(f"Text: {text}")
            print(f"Bounding Box: {bbox}")
            print(f"Confidence: {prob}")
            print("---")
    return frame
