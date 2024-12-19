import sys
import unicodedata

from ultralytics import YOLO
import cv2
import os
import argparse

import json

from utils.general_utils import download_video, get_jpg_files

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default="yolov5mu.pt")
parser.add_argument('--sequences_jsom_path', default="../colored_lights.json")
parser.add_argument('--in-dir', default="/Volumes/zalohy/DIP_unannontated")
parser.add_argument('--out-dir', default="/Volumes/zalohy/dip")
parser.add_argument('--label-light', type=int, default=81)
parser.add_argument('--train-test-split', type=int, default=0.75)

args = parser.parse_args()

# where to save
SAVE_PATH = args.out_dir


czech_railway_folder = "czech_railway_dataset"
img_index = 0


with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))


interesting_labels = {'stop'}
label_light = args.label_light

# creating folder with video name
if czech_railway_folder not in os.listdir(SAVE_PATH):
    os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}")

if "train" not in os.listdir(f"{SAVE_PATH}/{czech_railway_folder}"):
    for i in interesting_labels:
        if i not in f"{SAVE_PATH}/{czech_railway_folder}/train/images//":
            os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/images/{i}/")
            os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/labels/{i}")
            os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/images/{i}/")
            os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/labels/{i}")


image_counter = 0
for i in traffic_lights:
    image_counter += len(i)

last_train_sample = int(image_counter * args.train_test_split)
model = YOLO(args.nett_name)  # load an official model
image_counter = 0
files = get_jpg_files(args.in_dir)

#  todo filter the relevant files based on JSON colored_lights

for i in files:
    # Load a model

    results = model.predict(i)
    # Iterate over the results
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        class_indices = boxes.cls  # Class indices of the detections
        class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
        if len(interesting_labels & set(class_names)) > 0:
            # saves the result

            if image_counter > last_train_sample:
                save_name = f"{SAVE_PATH}/{czech_railway_folder}/val/"
            else:
                save_name = f"{SAVE_PATH}/{czech_railway_folder}/train/"
            cv2.imwrite(
                f"{save_name}images/{list(interesting_labels & set(class_names))[0]}/{img_index}.jpg",
                i)
            for r in results:
                boxes = r.boxes
                with open(
                        f"{save_name}labels/{list(interesting_labels & set(class_names))[0]}/{img_index}.txt",
                        mode="w") as label_f:
                    for index, box in enumerate(boxes):
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                        if result.names[int(box.cls)] in interesting_labels:
                            label_f.write(f"{label_light} {box.xywhn.tolist()[0][0]} {box.xywhn.tolist()[0][1]}"
                                          f" {box.xywhn.tolist()[0][2]} {box.xywhn.tolist()[0][3]}\n")
            img_index += 1
