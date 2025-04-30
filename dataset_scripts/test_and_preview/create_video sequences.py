import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from classifiers.combined_model import CzechRailwayLightModel




def get_jpg_files(path):
    """
    The function `get_jpg_files` retrieves a list of all JPG and JPEG files within a specified directory
    and its subdirectories.

    :param path: The `get_jpg_files` function you provided is designed to retrieve a list of all JPG and
    JPEG files within a specified directory and its subdirectories. To use this function, you need to
    provide the `path` parameter, which should be the directory path where you want to search for JPG
    and
    :return: The function `get_jpg_files(path)` returns a list of full file paths for all the JPG/JPEG
    files found in the specified directory `path` and its subdirectories.
    """
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


# Load YOLOv5 model
model = CzechRailwayLightModel(
    detection_nett_path="../../classifiers/czech_railway_light_detection_backbone/detection_backbone/weights/best.pt",
    classification_nett_path="../../classifiers/czech_railway_lights_model.pt"
                               )
# model = YOLO("../../classifiers/czech_railway_light_detection_backbone/detection_backbone/weights/best.pt")

# Open video file
video_path = "../../videos/Cabview 19  Pardubice-Polička duben 2023  strojvedoucicom"
times_path = "../../reconstructed/all_yolov5mu_raw/Cabview 19  Pardubice-Polička duben 2023  strojvedoucicom/yolov5mu/traffic light"

to_process = sorted([float(i[i.rfind("\\")+1:].replace("_box.jpg", "")) for i in os.listdir(times_path) if "_box.jpg" in i])
cap = cv2.VideoCapture(video_path + ".mp4")
# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter(video_path + '.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    for i in tqdm(to_process[:6]):

        frame_number = int(fps * (i - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                frame_number)
        frames = []
        for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                       int(np.ceil(cap.get(cv2.CAP_PROP_POS_FRAMES) + (fps))), 20)):
            ret, frame = cap.read()
            results = model(frame, conf=0.55, iou=0.55, verbose=False)
            results, classes = model(frame, conf=0.55, iou=0.55, verbose=False)
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                class_indices = boxes.cls  # Class indices of the detections
                class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names


                annotator = Annotator(frame, line_width=2)
                boxes = result.boxes
                for index, box in enumerate(boxes):
                for cls, box in zip(classes, boxes):
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                        c = box.cls

                        annotator.box_label(b, model.names[int(c)])
            out.write(frame)
                        annotator.box_label(b, model.names[cls])
            cv2.imshow("", cv2.resize(frame, (1280,720)))
            cv2.waitKey(100)
            if not ret:
                break


    cap.release()
    out.release()

