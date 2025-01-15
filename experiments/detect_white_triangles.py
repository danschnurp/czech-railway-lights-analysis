import argparse
import os
import sys
import time

import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from utils.general_utils import get_jpg_files
from utils.image_utils import calculate_nonzero_percent, crop_bounding_box, crop_sides_percentage


def detect_white_triangles(image, image2, name):
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

    _, result_image = cv2.threshold(result_image, 254, 255, cv2.THRESH_BINARY)
    res = calculate_nonzero_percent(crop_sides_percentage(result_image, crop_percentage=25))
    if 31. > res > 27. or 22. > res > 18. or 12. > res > 8.:
        # Display the results
        return True

        plt.figure(figsize=(12, 3))

        plt.subplot(1, 3, 1)
        plt.title(f"{name.split('/')[-4]}\n {name.split('/')[-1]}\n Original Image ")
        plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title(f"processed Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f"Detected White Triangles {res}")
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(f"./results/{time.time()}.png")
    return


def main(config):
    interesting_labels = {'traffic light'}
    directory = config.work_dir

    # Load a model
    model = YOLO(config.nett_name)  # load an official model
    files = get_jpg_files(directory)
    for index, i in enumerate(files):
        # Load the image
        image = cv2.imread(f"{i}")
        if image is None:
            print(f"Error: Could not read the image {image}.", file=sys.stderr)
            continue
        results = model.predict(image)
        # Iterate over the results
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            class_indices = boxes.cls  # Class indices of the detections
            class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names

            if len(interesting_labels & set(class_names)) > 0:
                for r in results:
                    # annotator = Annotator(image, line_width=2)
                    boxes = r.boxes
                    for index, box in enumerate(boxes):
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                        c = box.cls
                        if model.names[int(c)] in interesting_labels:
                            cropped_roi = crop_bounding_box(b, image)
                            detect_white_triangles(cropped_roi, image, name=i)
            # else:
            #     cv2.imshow("no detections", image)
            #     cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--nett_name', default='yolov10n.pt')
    parser.add_argument('--work_dir', default="/Volumes/zalohy/dip")

    args = parser.parse_args()
    main(args)
