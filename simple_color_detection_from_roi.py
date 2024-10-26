import os

import cv2
import numpy as np

from utils import calculate_nonzero_percent


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
