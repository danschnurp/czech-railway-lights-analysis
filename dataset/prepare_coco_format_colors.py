import sys
import time

import yaml
import random

import numpy as np
from ultralytics import YOLO
import cv2
import os
import argparse

import json

from utils.general_utils import download_video, get_jpg_files , normalize_text
from utils.image_utils import get_roi_coordinates

def euclidean_distance(a, b):
    return np.sqrt(np.sum((np.array(a) - np.array(b))**2))


parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default="../yolov5mu.pt")
parser.add_argument('--sequences_jsom_path', default="../railway_datasets/video_names.json")
parser.add_argument('--in-dir', default="../reconstructed/all_yolov5mu_raw")
parser.add_argument('--out-dir', default="../dataset")
parser.add_argument('--label-light', type=int, default=79)
parser.add_argument('--train-test-split', type=int, default=0.25)

args = parser.parse_args()

# where to save
SAVE_PATH = args.out_dir

czech_railway_folder = "czech_railway_dataset_4_classes"
classes_dir_path = "../railway_datasets/4_classes"
dataset_yaml = '../metacentrum/CRTL_multi_4_labeled.yaml'


img_index = 0

with open(dataset_yaml, encoding="utf-8") as f:
    class_mapping = yaml.load(f, Loader=yaml.SafeLoader)

class_mapping = class_mapping["names"]
class_mapping = dict(zip(class_mapping.values(), class_mapping.keys()))

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

all_classes = {}

for i in os.listdir(classes_dir_path):
    with open(classes_dir_path + "/" + i, encoding="utf-8") as f:
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


total_pictures_count = sum([len([j for j in all_classes[i]]) for i in all_classes])
last_train_sample = int(total_pictures_count * args.train_test_split)
model = YOLO(args.nett_name)  # load an official model
image_counter = 0
#
# with open("today_results.json", "w", encoding="utf-8") as f:
#     json.dump(all_classes, f, indent=2, ensure_ascii=False)
#
# exit(0)

lost_pictures = 0
for video_name in all_classes:
    timestamps_shuffled = list(all_classes[video_name].keys())
    random.shuffle(timestamps_shuffled)
    for timestamp in timestamps_shuffled:
        dir_path = f"{args.in_dir}/{normalize_text(video_name)}/yolov5mu/{original_label}"
        real_picture_path = f"{dir_path}/{timestamp}_clean.jpg"
        closest = 0
        difference_previous = np.inf

        if not os.path.exists(real_picture_path):
            for i in os.listdir(dir_path):
                if i.find("_clean.jpg") != -1:
                    difference = np.abs(timestamp - float(i[:i.find("_clean")]))
                    if difference <= difference_previous:
                        difference_previous = difference
                        closest = float(i[:i.find("_clean")])
            if difference_previous > 0.5:
                lost_pictures += 1
                print("difference:", difference_previous, "closest:", closest, "timestamp:", timestamp, file=sys.stderr)
                continue
            real_picture_path = f"{args.in_dir}/{video_name}/yolov5mu/{original_label}/{closest:0.3f}_clean.jpg"
        try:
            img = cv2.imread(real_picture_path)
        except FileNotFoundError:
            try:
                print("neco se posralo, počkáme", real_picture_path, file=sys.stderr)
                time.sleep(0.6)
                img = cv2.imread(real_picture_path)
            except FileNotFoundError:
                print("neco se posralo", real_picture_path, file=sys.stderr)
        most_similar_quadruples = []
        for original in all_classes[video_name][timestamp]:
            detected = get_roi_coordinates(model, frame=img)
            original_split = list(map(float, original[original.find(" "):].split()))
            # Calculate distances
            distances = [euclidean_distance([original_split[:2], original_split[2:]], quad) for quad in detected]
            if len(distances) == 0:
                print("strange", file=sys.stderr)
                lost_pictures += 1
                continue
            # Find the most similar quadruple
            most_similar_index = np.argmin(distances)
            detected[most_similar_index] = [*detected[most_similar_index][0], *detected[most_similar_index][1]]
            most_similar_quadruples.append(
                f"{original[:original.find(' ')]} {detected[most_similar_index][0]} {detected[most_similar_index][1]}"
                f" {detected[most_similar_index][2]} {detected[most_similar_index][3]}\n")
        all_classes[video_name][timestamp] = most_similar_quadruples

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

print("lost/total", lost_pictures, "/", total_pictures_count)