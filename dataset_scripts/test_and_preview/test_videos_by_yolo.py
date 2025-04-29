import argparse
import json
import os
import time

import cv2
import torch
import yaml
from ultralytics.utils.plotting import Annotator

from classifiers.combined_model import CzechRailwayLightModel
from utils.general_utils import download_video
from utils.image_utils import MovementDetector, convert_normalized_roi_to_pixels



parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_path', default='../../reconstructed/100_lights_2_yolov10n.pt_0.55/weights/best.pt')
parser.add_argument('--sequences_jsom_path', default="../../railway_datasets/video_names.json")
parser.add_argument('--in-dir', default="../../videos") # test_videos
parser.add_argument('--out-dir', default="../../test_results")
parser.add_argument('--skip_seconds', type=int, default=0)

args = parser.parse_args()

SAVE_PATH = args.out_dir
LOAD_PATH = args.in_dir

with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))

with open("../../metacentrum_experiments/CRL_single_images_less_balanced.yaml") as f:
    interesting_labels = set(list(yaml.load(f, yaml.SafeLoader)["names"].values()))

# Load a model
model = CzechRailwayLightModel(
    detection_nett_path="../../classifiers/czech_railway_light_detection_backbone/detection_backbone/weights/best.pt",
    classification_nett_path="../../classifiers/czech_railway_lights_model.pt"
                               )


def annotate_video():

    video_name = d_video
    # creating folder with video name
    try:
        if video_name[:-4] not in os.listdir(f"{SAVE_PATH}/"):
            os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}")

    except FileExistsError as e:
        print(e, "maybe different encoding")

    # Load video
    video_path = LOAD_PATH + '/' + video_name
    cap = cv2.VideoCapture(video_path)

    image_index = 0
    dropout_time = 0
    skip_seconds = args.skip_seconds
    #
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * skip_seconds)
    cap.set(cv2.CAP_PROP_POS_FRAMES,
            frame_number)

    t1 = 5

    ret, frame = cap.read()

    mov_detector = MovementDetector(frame=frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if time.time() - t1 - dropout_time > 0.05:
            movement = mov_detector.detect_movement(frame)
            if not movement:
                dropout_time = 2
                continue
            dropout_time = 0.3
            # timestamp seconds from video beginning
            timestamp = f"{float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.):.3f}"
            # frame = crop_sides_percentage(frame.copy(), crop_percentage=10)
            # frame = crop_top_bottom_percentage(frame.copy(), crop_percentage=10)
            results = model(frame,conf=0.55, iou=0.45)
            # Iterate over the results
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                class_indices = boxes.cls  # Class indices of the detections
                class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
                if len(interesting_labels & set(class_names)) > 0:
                    # saves the result
                    save_name = f"{SAVE_PATH}/{video_name[:-4]}/{timestamp}"
                    dropout_time = 0.5
                    annotator = Annotator(frame, line_width=2)
                    boxes = result.boxes
                    for index, box in enumerate(boxes):
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                        c = box.cls
                        # Extract ROI
                        frame_height, frame_width = frame.shape[:2]

                        x_min, y_min, width, height = convert_normalized_roi_to_pixels(
                            " ".join([str(box.xywhn.tolist()[0][0]), str(box.xywhn.tolist()[0][1]),
                                      str(box.xywhn.tolist()[0][2]), str(box.xywhn.tolist()[0][3])])
                            , frame_width, frame_height)

                        crop = frame[y_min:height, x_min:width]
                        annotator.box_label(b, model.names[int(c)])


                    image_index += 1
                    cv2.imwrite(
                        save_name + "_box.jpg",
                        frame)
            t1 = time.time()
    cap.release()


done = {}
for i in traffic_lights["names"].values():
    d_video = download_video(i, LOAD_PATH, names_jsom_path=args.sequences_jsom_path)
    annotate_video()
    done[d_video] = i

print(done)
