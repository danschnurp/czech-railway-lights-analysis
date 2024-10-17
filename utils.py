import json

import os

import argparse
from easyocr import Reader
import cv2
import numpy as np
from pytube import YouTube
import subprocess


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
            '-f', 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]',
            '-o', f'{SAVE_PATH}/' + video_name,
        ]
        subprocess.run(command, check=True)

    return video_name
