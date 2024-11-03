import json

import os

import argparse
from math import gcd
from typing import Tuple

from easyocr import Reader
import cv2
import numpy as np
from pytube import YouTube
import subprocess


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


def get_jpg_files(path):
    jpg_files = []

    # Walk through directory and subdirectories
    for root, dirs, files in os.walk(path):
        # Find all jpg/jpeg files in current directory
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                # Create full file path and add to list
                full_path = os.path.join(root, file)
                jpg_files.append(full_path)

    return jpg_files


def calculate_nonzero_percent(result):
    # Get total number of pixels
    total_pixels = result.shape[0] * result.shape[1]
    nonzero_mask = np.any(result > 0, axis=2)
    # Count non-zero pixels
    # Using np.count_nonzero() on any channel since mask affects all channels
    nonzero_pixels = np.count_nonzero(nonzero_mask)

    # Calculate percentage
    percent = (nonzero_pixels / total_pixels) * 100

    return percent


def split_train_val_test(
        path="/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed"
             "/czech_railway_dataset/train/images/metadata.txt"):
    with open(path) as f:
        metadata = f.readlines()

    metadata = np.array(metadata)
    np.random.shuffle(metadata)
    train = metadata[:int(metadata.shape[0] * 0.9)]
    test = metadata[int(metadata.shape[0] * 0.9):]

    assert (len(train) + len(test)) == len(metadata)

    # [os.replace(f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed"
    #          f"/czech_railway_dataset/train/images/traffic light/{i[:-1]}",
    #             f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed"
    #             f"/czech_railway_dataset/val/images/traffic light/{i[:-1]}"
    #             ) for i in test]

    [os.replace(f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed"
             f"/czech_railway_dataset/train/images/traffic light/{i[:-1]}",
                f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed"
                f"/czech_railway_dataset/val/images/traffic light/{i[:-1]}"
                ) for i in test]


#
# for i in os.listdir("/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset"
#                     "/reconstructed/czech_railway_dataset/val/images/traffic light"):
#     # Full paths for old and new names
#     old_path = os.path.join(f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset"
#                             f"/reconstructed"
#              f"/czech_railway_dataset/train/labels/traffic light/", i[:-4] + ".txt")
#     new_path = os.path.join(f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset"
#                             f"/reconstructed"
#              f"/czech_railway_dataset/val/labels/traffic light/",  i[:-4] + ".txt")
#
#     # Rename the file
#     os.replace(old_path, new_path)


def perform_ocr(reader: Reader, frame: np.ndarray, confidence_threshold=0.1):
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def enlarge_bounding_box(bounding_box, bigger_top_percents=0.1, bigger_bottoms_percents=0.25):
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


def crop_bounding_box(box, img):
    x, y, w, h = box
    x, y, w, h = int(x), int(y), int(w), int(h)
    roi = img[y:h, x:w]

    # cropped_roi = cropped_roi[int(cropped_roi.shape[0] * 0.7):]
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #
    #
    # cropped_roi = cv2.filter2D(cropped_roi, -1, kernel)

    return roi


def download_video_pytube(link, SAVE_PATH):
    try:
        # object creation using YouTube
        yt = YouTube(link)
    except:
        # to handle exception
        print(" YouTube Connection Error")

        # Get all streams and filter for mp4 files
    mp4_streams = yt.streams.filter(resolution="720p")

    # get the video with the highest resolution
    d_video = mp4_streams[0]

    if d_video in os.listdir(SAVE_PATH):
        return d_video

    try:
        print('downloading the video' + d_video.default_filename)

        d_video.download(output_path=SAVE_PATH)
        print('Video downloaded successfully!')
    except Exception:
        print("Some Error!")
    return d_video


def get_youtube_video_info(youtube_url):
    command = [
        'yt-dlp',
        '-j',
        youtube_url
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    video_info = json.loads(result.stdout)
    return video_info['title']


def download_video(link, SAVE_PATH):
    video_name = get_youtube_video_info(link)
    video_name = video_name.strip().replace("⧸", "").replace("/", "").replace("#", "").replace(",", "").replace(".", "")
    video_name += ".mp4"

    if video_name in os.listdir(SAVE_PATH):
        return video_name
    try:
        command = [
            'yt-dlp',
            link,
            '-f', 'bestvideo[height=1080][fps=60][ext=mp4]/best[height=1080][fps=60][ext=mp4]',
            '-o', f'{SAVE_PATH}/' + video_name,
        ]
        subprocess.run(command, check=True)
    except:
        print("full HD quality failed... trying 720p")
        command = [
            'yt-dlp',
            link,
            '-f', 'bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]',
            '-o', f'{SAVE_PATH}/' + video_name,
        ]
        subprocess.run(command, check=True)

    return video_name
