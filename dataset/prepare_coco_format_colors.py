import sys
import unicodedata
import random

import numpy as np
from ultralytics import YOLO
import cv2
import os
import argparse

import json

from utils.general_utils import download_video, get_jpg_files

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default="yolov5mu.pt")
parser.add_argument('--sequences_jsom_path', default="../colored_lights.json")
parser.add_argument('--in-dir', default="/Volumes/zalohy/dip/all_yolov5")
parser.add_argument('--out-dir', default="../dataset")
parser.add_argument('--label-light', type=int, default=79)
parser.add_argument('--train-test-split', type=int, default=0.75)

args = parser.parse_args()

# where to save
SAVE_PATH = args.out_dir


czech_railway_folder = "czech_railway_dataset"
img_index = 0


with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))
    traffic_lights = traffic_lights["data"]

original_label = 'traffic light'
target_labels = {
    'warning_go':"warning_go",
                 # 'green':"go"
                 }
target_ids = {
    'warning_go': 2,
    # 'green': 1
}

label_light = args.label_light

# creating folder with video name
if czech_railway_folder not in os.listdir(SAVE_PATH):
    os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}")
    os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/")
    os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val")
    os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/images")
    os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/images")
    os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/labels")
    os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/labels")


for i in target_labels.values():
        if i not in os.listdir(f"{SAVE_PATH}/{czech_railway_folder}/train/images/"):
            os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/images/{i}/")
            os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/labels/{i}")
            os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/images/{i}/")
            os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/labels/{i}")

one_class = []

for i in traffic_lights:
    if  i["color"] in target_labels.keys():
        one_class.append(i)

last_train_sample = int(len(one_class) * args.train_test_split)
model = YOLO(args.nett_name)  # load an official model
image_counter = 0

random.shuffle(traffic_lights)

for metadata in traffic_lights:
    if metadata["color"] in target_labels.keys():
        dir_path = f"{args.in_dir}/{metadata['video name']}/{metadata['detection method']}/{original_label}"
        real_picture_path = f"{dir_path}/{metadata['timestamp in video']}_clean.jpg"
        difference_previous = np.inf
        if not os.path.exists(real_picture_path):
            for i in os.listdir(dir_path):
                if i.find("_clean.jpg") != -1:
                    difference = np.abs(metadata['timestamp in video'] - float(i[:i.find("_clean")]))
                    if difference <= difference_previous:
                        difference_previous = difference
            closest = float(metadata['timestamp in video'] + difference_previous)
            real_picture_path = f"{args.in_dir}/{metadata['video name']}/{metadata['detection method']}/{original_label}/{closest:0.3f}_clean.jpg"

        img = cv2.imread(real_picture_path)

        if image_counter > last_train_sample:
            save_name = f"{SAVE_PATH}/{czech_railway_folder}/val/"
        else:
            save_name = f"{SAVE_PATH}/{czech_railway_folder}/train/"
        cv2.imwrite(f"{save_name}images/{target_labels[metadata['color']]}/{img_index}.jpg",img)
        with open(
                f"{save_name}labels/{target_labels[metadata['color']]}/{img_index}.txt",
                mode="w") as label_f:

            label_f.write(f"{target_ids[metadata['color']]} {metadata['roi coordinates']}\n")
        img_index += 1
        image_counter += 1
