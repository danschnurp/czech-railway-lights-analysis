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
parser.add_argument('--sequences_jsom_path', default="../railway_datasets/video_names.json")
parser.add_argument('--in-dir', default="../reconstructed/all_yolov5mu_raw")
parser.add_argument('--out-dir', default="../dataset")
parser.add_argument('--label-light', type=int, default=79)
parser.add_argument('--train-test-split', type=int, default=0.75)

args = parser.parse_args()

# where to save
SAVE_PATH = args.out_dir


czech_railway_folder = "czech_railway_dataset"
img_index = 0


class_mapping = {"stop": 1, "go": 2, "warning_go": 3}

with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    video_names = dict(json.load(f))
    video_names = video_names["names"]

original_label = 'traffic light'


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


for i in ["multi_class"]:
        if i not in os.listdir(f"{SAVE_PATH}/{czech_railway_folder}/train/images/"):
            os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/images/{i}/")
            os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/labels/{i}")
            os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/images/{i}/")
            os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/labels/{i}")

classes_dir_path = "../railway_datasets/simple_classes"

all_classes = {}

for i in os.listdir(classes_dir_path):
    with open(classes_dir_path + "/" + i) as f:
        data = dict(json.load(f))["data"]
    for j in data:
        try:
            all_classes[j["video name"]][j["timestamp in video"]].append(f"{class_mapping[j['color']]} {j['roi coordinates']}")
        except KeyError:
            try:
                all_classes[j["video name"]][j["timestamp in video"]] = [f"{class_mapping[j['color']]} {j['roi coordinates']}"]
            except KeyError:
                all_classes[j["video name"]] = {
                    j["timestamp in video"]: [f"{class_mapping[j['color']]} {j['roi coordinates']}"]
                }


last_train_sample = int(
sum([len([j for j in all_classes[i]]) for i in all_classes])
 * args.train_test_split)
model = YOLO(args.nett_name)  # load an official model
image_counter = 0
#
# with open("today_results.json", "w", encoding="utf-8") as f:
#     json.dump(all_classes, f, indent=2, ensure_ascii=False)
#
# exit(0)


for video_name in all_classes:
    timestamps_shuffled = list(all_classes[video_name].keys())
    random.shuffle(timestamps_shuffled)
    for timestamp in timestamps_shuffled:
        dir_path = f"{args.in_dir}/{video_name}/yolov5mu/{original_label}"
        real_picture_path = f"{dir_path}/{timestamp}_clean.jpg"
        difference_previous = np.inf
        if not os.path.exists(real_picture_path):
            print("jsem v prdeli")
            continue
        # if not os.path.exists(real_picture_path):
        #     for i in os.listdir(dir_path):
        #         if i.find("_clean.jpg") != -1:
        #             difference = np.abs(timestamp - float(i[:i.find("_clean")]))
        #             if difference <= difference_previous:
        #                 difference_previous = difference
        #     closest = float(metadata['timestamp in video'] + difference_previous)
        #     real_picture_path = f"{args.in_dir}/{metadata['video name']}/{metadata['detection method']}/{original_label}/{closest:0.3f}_clean.jpg"

        img = cv2.imread(real_picture_path)

        if image_counter > last_train_sample:
            save_name = f"{SAVE_PATH}/{czech_railway_folder}/val/"
        else:
            save_name = f"{SAVE_PATH}/{czech_railway_folder}/train/"
        cv2.imwrite(f"{save_name}images/multi_class/{img_index}.jpg",img)
        with open(
                f"{save_name}labels/multi_class/{img_index}.txt",
                mode="w") as label_f:
            coordinates = ""
            for ii in all_classes[video_name][timestamp]:
                coordinates += f"{ii}\n"
            label_f.write(coordinates)
        img_index += 1
        image_counter += 1
