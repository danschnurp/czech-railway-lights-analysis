import sys

import numpy as np
from easyocr import Reader
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os
import argparse
from utils import download_video, crop_bounding_box, perform_ocr
import time

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='yolov5mu.pt')
parser.add_argument('-l', '--link', default="https://youtu.be/8rtVqE2yclo")
parser.add_argument('--skip_seconds', type=int, default=0)

args = parser.parse_args()

# where to save
SAVE_PATH = "./videos"  # to_do

# link of the video to be downloaded
link = args.link

d_video = download_video(link, SAVE_PATH)

interesting_labels = {'traffic light'}

nett_name = args.nett_name

video_name = d_video
# creating folder with video name
if video_name[:-4] not in os.listdir("./videos/"):
    os.mkdir(f"./videos/{video_name[:-4]}")

# creating folder with yolo type and label folders
if nett_name[:-3] not in os.listdir(f"./videos/{video_name[:-4]}"):
    os.mkdir(f"./videos/{video_name[:-4]}/{nett_name[:-3]}/")
    for i in interesting_labels:
        os.mkdir(f"./videos/{video_name[:-4]}/{nett_name[:-3]}/{i}/")

# Load a models
model = YOLO(nett_name)  # load an official model
reader = Reader(['en'])  # Specify language(s) as needed
# Set confidence threshold for ocr
confidence_threshold = 0.01

# Load video
video_path = 'videos/' + video_name
cap = cv2.VideoCapture(video_path)

image_index = 0
dropout_time = 0
skip_seconds = args.skip_seconds
#
cap.set(cv2.CAP_PROP_POS_MSEC,
        skip_seconds * 1000
        )

t1 = 10
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if time.time() - t1 - dropout_time > 0.05:
        dropout_time = 0
        # timestamp seconds from video beginning
        timestamp = f"{float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.):.3f}"
        results = model.predict(frame)
        # Iterate over the results
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            class_indices = boxes.cls  # Class indices of the detections
            class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
            print(class_names)
            if len(interesting_labels & set(class_names)) > 0:
                # saves the result
                save_name = f"./videos/{video_name[:-4]}/{nett_name[:-3]}/" \
                            f"{list(interesting_labels & set(class_names))[0]}/"
                dropout_time = 0.1
                # cv2.imwrite(
                #     save_name + "_clean.jpg",
                #     frame)
                for r in results:
                    annotator = Annotator(frame, line_width=2)
                    boxes = r.boxes
                    for index, box in enumerate(boxes):
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                        # enlarging ROI for digits and lines detections
                        # Calculate original height
                        original_height = b[3] - b[1]
                        # Calculate 10% of the original height
                        adjustment = 0.1 * original_height
                        # Adjust top and bottom coordinates
                        new_top = b[1] - float(adjustment)
                        new_bottom = b[3] + float(0.25 * original_height)
                        b = [b[0], new_top, b[2], new_bottom]
                        c = box.cls
                        if model.names[int(c)] in interesting_labels:
                            cropped_roi = crop_bounding_box(b, frame)
                            cropped_roi = cropped_roi[int(cropped_roi.shape[0] * 0.7):]
                            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

                            try:
                                cropped_roi = cv2.filter2D(cropped_roi, -1, kernel)
                                perform_ocr(reader, cropped_roi, confidence_threshold)
                                annotator.box_label(b, model.names[int(c)])
                                cv2.imwrite(f"{save_name}roi_{timestamp}_{index}.jpg", cropped_roi)
                            except cv2.error:
                                print(f"err {save_name} file", file=sys.stderr)
                                continue
                            img = annotator.result()
                            # saves the result with bounding box
                            cv2.imwrite(save_name + f"box_{timestamp}.jpg", img)

                image_index += 1
        t1 = time.time()
cap.release()
