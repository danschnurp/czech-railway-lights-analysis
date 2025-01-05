import os
import sys
import time

import numpy as np
from math import gcd

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


def annotate_pictures(args, save_path, interesting_label = 'traffic light',     detected_nett_name = "yolov5mu",
                      tolerance = 0.5):
    from utils.general_utils import get_times_by_video_name, get_jpg_files, normalize_list_of_texts

    nett_name = args.nett_name

    traffic_lights = get_times_by_video_name(args.sequences_jsom_path)
    del traffic_lights["names"]
    del traffic_lights["todo"]
    # Load a model
    model = YOLO(nett_name)

    for video_name in traffic_lights:
        if video_name not in normalize_list_of_texts(os.listdir(args.in_dir)):
            raise FileNotFoundError(f"Video {video_name} not found in {args.in_dir}")
        try:
            # creating folder with video name
            if video_name not in os.listdir(save_path):
                os.mkdir(f"{save_path}/{video_name}")
        except FileExistsError as e:
            print(e, "maybe different encoding")

        # creating folder with yolo type and label folders
        if nett_name[:-3] not in os.listdir(f"{save_path}/{video_name}"):
            os.mkdir(f"{save_path}/{video_name}/{nett_name[:-3]}/")
            os.mkdir(f"{save_path}/{video_name}/{nett_name[:-3]}/{interesting_label}/")

        image_index = 0
        input_dir = f"{args.in_dir}/{video_name}/{detected_nett_name}/{interesting_label}/"
        real_timestamps = [float(i.split("_")[0]) for i in os.listdir(input_dir)]
        for timestamp in traffic_lights[video_name]:
            try:
                nearest_index = real_timestamps[[1 if -tolerance < i - timestamp < tolerance else 0
                                                 for i in real_timestamps].index(1)]
            except ValueError:
                continue
            frame = cv2.imread(f"{args.in_dir}/{video_name}/{detected_nett_name}/{interesting_label}/{nearest_index:.3f}_clean.jpg")

            results = model.predict(frame)
            # Iterate over the results
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                class_indices = boxes.cls  # Class indices of the detections
                class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
                print(class_names, "timestamp:", timestamp)
                if len({interesting_label} & set(class_names)) > 0:
                    # saves the result
                    save_name = f"{save_path}/{video_name}/{nett_name[:-3]}/" \
                                f"{list({interesting_label} & set(class_names))[0]}/{timestamp}"
                    dropout_time = 0.1
                    if args.clean_pictures:
                        cv2.imwrite(
                            save_name + "_clean.jpg",
                            frame)
                    for r in results:
                        annotator = Annotator(frame, line_width=2)
                        boxes = r.boxes
                        for index, box in enumerate(boxes):
                            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                            # new_top, new_bottom = enlarge_bounding_box(b)
                            # b = [b[0], new_top, b[2], new_bottom]
                            c = box.cls
                            if model.names[int(c)] == interesting_label:
                                if args.roi_pictures:
                                    cropped_roi = crop_bounding_box(b, frame)
                                    try:
                                        cv2.imwrite(f"{save_name}_roi{index}.jpg", cropped_roi)
                                    except cv2.error as e:
                                        print(e)
                                annotator.box_label(b, model.names[int(c)])

                    img = annotator.result()
                    if args.bounding_box_pictures:
                        # saves the result with bounding box
                        cv2.imwrite(save_name + "_box.jpg", img)

                    image_index += 1


def detect_keyframes_histogram(cap, threshold=0.2, timeout=1000):
    prev_hist = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > timeout:
            break

        frame_count += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the histogram for the current frame
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            # Compute correlation between histograms
            score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)

            # Detect keyframe if similarity is below threshold
            if score < threshold:
                return frame

        prev_hist = hist  # Update previous histogram


def get_picture(cap, model, args, interesting_labels, video_name, nett_name, image_index, SAVE_PATH):


    _, frame = cap.read()

    timestamp = f"{float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.):.3f}"
    results = model.predict(frame)
    # Iterate over the results
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        class_indices = boxes.cls  # Class indices of the detections
        class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
        print(class_names, "timestamp:", timestamp)
        if len(interesting_labels & set(class_names)) > 0:
            # saves the result
            save_name = f"{SAVE_PATH}/{video_name[:-4]}/{nett_name[:-3]}/" \
                        f"{list(interesting_labels & set(class_names))[0]}/{timestamp}"
            dropout_time = 0.1
            if args.clean_pictures:
                cv2.imwrite(
                    save_name + "_clean.jpg",
                    frame)
            for r in results:
                annotator = Annotator(frame, line_width=2)
                boxes = r.boxes
                for index, box in enumerate(boxes):
                    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    # new_top, new_bottom = enlarge_bounding_box(b)
                    # b = [b[0], new_top, b[2], new_bottom]
                    c = box.cls
                    if model.names[int(c)] in interesting_labels:
                        if args.roi_pictures:
                            cropped_roi = crop_bounding_box(b, frame)
                            try:
                                cv2.imwrite(f"{save_name}_roi{index}.jpg", cropped_roi)
                            except cv2.error as e:
                                print(e)
                        annotator.box_label(b, model.names[int(c)])

            img = annotator.result()
            if args.bounding_box_pictures:
                # saves the result with bounding box
                cv2.imwrite(save_name + "_box.jpg", img)

            image_index += 1

            return image_index

    return 0


def test_roi_detections(d_video, metadata, args, SAVE_PATH):

    interesting_labels = {'traffic light'}

    nett_name = args.nett_name

    video_name = d_video

    # Load a model
    model = YOLO(nett_name)  # load an official model

    # Load video
    video_path = SAVE_PATH + '/' + video_name
    cap = cv2.VideoCapture(video_path)

    image_index = 0
    #
    start_time = metadata["timestamp in video"] - args.sequence_seconds_before
    print("from", metadata["timestamp in video"], file=sys.stderr)
    if start_time < 0.:
        print("starting from beginning")
        start_time = 0
    cap.set(cv2.CAP_PROP_POS_MSEC,
            start_time * 1000)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # end of sequence
        if (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.) > (args.sequence_seconds_after + metadata["timestamp in video"]):
            return
        else:
            # timestamp seconds from video beginning
            timestamp = f"{float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.):.3f}"
            results = model.predict(frame, conf=0.1)
            # Iterate over the results
            annotator = Annotator(frame, line_width=2)
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                class_indices = boxes.cls  # Class indices of the detections
                class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
                print(class_names, "timestamp:", timestamp)
                if len(interesting_labels & set(class_names)) > 0:

                    b = metadata["roi coordinates"].split(" ")

                    x =b[0]
                    y = b[1]
                    width = float(b[2])
                    height = float(b[3])

                    x1 = float(x) + width
                    y1 = float(y)   + height
                    x2 = float(x) - width
                    y2 = float(y) - height

                    ultralytics_coordinates = (float(x1) * frame.shape[1], float(y1)* frame.shape[0], float(x2) * frame.shape[1], y2* frame.shape[0])
                    annotator.box_label(ultralytics_coordinates, metadata['color'])
                    img = annotator.result()
                    cv2.imwrite( f"./{metadata['ID']}_{metadata['color']}.jpg", img)




def get_pictures(d_video, seek_seconds, args, SAVE_PATH):

    interesting_labels = {'traffic light'}

    nett_name = args.nett_name

    video_name = d_video
    try:
        # creating folder with video name
        if video_name[:-4] not in os.listdir(SAVE_PATH):
            os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}")
    except FileExistsError as e:
        print(e, "maybe different encoding")

    # creating folder with yolo type and label folders
    if nett_name[:-3] not in os.listdir(f"{SAVE_PATH}/{video_name[:-4]}"):
        os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}/{nett_name[:-3]}/")
        for i in interesting_labels:
            os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}/{nett_name[:-3]}/{i}/")

    # Load a model
    model = YOLO(nett_name)  # load an official model

    # Load video
    video_path = SAVE_PATH + '/' + video_name
    cap = cv2.VideoCapture(video_path)

    image_index = 0
    #
    start_time = seek_seconds - args.sequence_seconds_before
    print("from", seek_seconds, file=sys.stderr)
    if start_time < 0.:
        print("starting from beginning")
        start_time = 0
    cap.set(cv2.CAP_PROP_POS_MSEC,
            start_time * 1000)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # end of sequence
        if (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.) > (args.sequence_seconds_after + seek_seconds):
            print(f"finished {seek_seconds}")
            return
        else:
            # timestamp seconds from video beginning
            timestamp = f"{float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.):.3f}"
            results = model.predict(frame, conf=0.5)
            # Iterate over the results
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                class_indices = boxes.cls  # Class indices of the detections
                class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
                print(class_names, "timestamp:", timestamp)
                if len(interesting_labels & set(class_names)) > 0:
                    # saves the result
                    save_name = f"{SAVE_PATH}/{video_name[:-4]}/{nett_name[:-3]}/" \
                                f"{list(interesting_labels & set(class_names))[0]}/{timestamp}"
                    dropout_time = 0.1
                    if args.clean_pictures:
                        cv2.imwrite(
                            save_name + "_clean.jpg",
                            frame)
                    for r in results:
                        annotator = Annotator(frame, line_width=2)
                        boxes = r.boxes
                        for index, box in enumerate(boxes):
                            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                            # new_top, new_bottom = enlarge_bounding_box(b)
                            # b = [b[0], new_top, b[2], new_bottom]
                            c = box.cls
                            if model.names[int(c)] in interesting_labels:
                                if args.roi_pictures:
                                    cropped_roi = crop_bounding_box(b, frame)
                                    try:
                                        cv2.imwrite(f"{save_name}_roi{index}.jpg", cropped_roi)
                                    except cv2.error as e:
                                        print(e)
                                annotator.box_label(b, model.names[int(c)])

                    img = annotator.result()
                    if args.bounding_box_pictures:
                        # saves the result with bounding box
                        cv2.imwrite(save_name + "_box.jpg", img)

                    image_index += 1
    cap.release()





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
        if movement_prob < 0.15:
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


def detect_red_without_stats(image_path, color=red):
    bad_colors = {yellow, red, orange, yellow_orange, green}

    bad_colors -= {color}

    image = cv2.imread(image_path)
    aspect_ratio, w, h = calculate_aspect_ratio(image)
    # image = replace_white_with_black(image)
    result_color = crop_sides_percentage(crop_top_half(detect_color(image, color_filter=color)))
    bad_colors_result_perc = [calculate_nonzero_percent(detect_color(image, i)) for i in bad_colors]

    is_centered = check_content_centered(result_color)
    if calculate_nonzero_percent(result_color) > 0.2 \
                and 0.4 > max(bad_colors_result_perc) and is_centered:
        return True
    return False


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


def crop_top_bottom_percentage(img, crop_percentage=10):
    # Get image dimensions
    height, width = img.shape[:2]
    # Calculate crop height (percentage from top and bottom)
    crop_height = int(height * (crop_percentage / 100))
    # Crop the image
    # Start from crop_height from top
    # End at height - crop_height from bottom
    return img[crop_height:height - crop_height, :]

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


def perform_ocr(reader, frame: np.ndarray, confidence_threshold=0.1):
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
