from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', '--movie', default="4K  Bezdružice - Plzeň  842 Kvatro.mp4")

parser.add_argument('--nett_name', default='yolov5mu.pt')

args = parser.parse_args()

interesting_labels = {'traffic light'}

nett_name = args.nett_name

video_name = args.movie

if video_name[:-4] not in os.listdir("./videos/"):
    os.mkdir(f"./videos/{video_name[:-4]}")
if nett_name[:-3] not in os.listdir(f"./videos/{video_name[:-4]}"):
    os.mkdir(f"./videos/{video_name[:-4]}/{nett_name[:-3]}/")
    for i in interesting_labels:
        os.mkdir(f"./videos/{video_name[:-4]}/{nett_name[:-3]}/{i}/")

# Load a model
model = YOLO(nett_name)  # load an official model

# Load video
video_path = 'videos/' + video_name
cap = cv2.VideoCapture(video_path)

import time

image_index = 0
dropout_time = 0

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
                            f"{list(interesting_labels & set(class_names))[0]}/{timestamp}"
                dropout_time = 0.1
                cv2.imwrite(
                    save_name + "_clean.jpg",
                    frame)
                for r in results:
                    annotator = Annotator(frame)
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                        c = box.cls
                        annotator.box_label(b, model.names[int(c)])

                img = annotator.result()
                # saves the result with bounding box
                cv2.imwrite(save_name + "_box.jpg", img)

                image_index += 1
        t1 = time.time()
cap.release()
