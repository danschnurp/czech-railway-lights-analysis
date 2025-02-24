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

from utils.phash_similarity import calculate_pictures_similarity
from utils.general_utils import download_video , normalize_text
from utils.image_utils import get_roi_coordinates, save_annotated_picture


def euclidean_distance(a, b):
    return np.sqrt(np.sum((np.array(a) - np.array(b))**2))


parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default="../yolov5mu.pt")
parser.add_argument('--sequences_jsom_path', default="../railway_datasets/video_names.json")
parser.add_argument('--in-dir', default="../videos")
parser.add_argument('--out-dir', default="../dataset")
parser.add_argument('--mili_seconds_before', type=float, default=3000)
parser.add_argument('--mili_seconds_after', type=float, default=500)
parser.add_argument('--delta_step', type=float, default=500)
parser.add_argument('--label-light', type=int, default=79)
parser.add_argument('--train-test-split', type=int, default=0.25)

args = parser.parse_args()

# where to save
SAVE_PATH = args.out_dir

czech_railway_folder = "czech_railway_dataset_4_classes"
classes_dir_path = "../railway_datasets/simple_classes"
dataset_yaml = '../metacentrum/CRTL_multi_labeled.yaml'
# todo from 'https://www.youtube.com/watch?v=Mhwh4KXrlb8' and 161. picture

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

for i in ["multi_class_annotated"]:
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
            all_classes[j["ytlink"]][j["timestamp in video"]].append(f"{class_mapping[j['color']]} {j['roi coordinates']}")
        except KeyError:
            try:
                all_classes[j["ytlink"]][j["timestamp in video"]] = [f"{class_mapping[j['color']]} {j['roi coordinates']}"]
            except KeyError:
                all_classes[j["ytlink"]] = {
                    j["timestamp in video"]: [f"{class_mapping[j['color']]} {j['roi coordinates']}"]
                }


total_pictures_count = sum([len([j for j in all_classes[i]]) for i in all_classes])

last_train_sample = int(total_pictures_count)

total_pictures_count *= (args.mili_seconds_before / args.delta_step)

print("estimated dataset size:", total_pictures_count)

# last_train_sample = int(total_pictures_count * args.train_test_split)
model = YOLO(args.nett_name)  # load an official model
image_counter = 0
#
# with open("today_results.json", "w", encoding="utf-8") as f:
#     json.dump(all_classes, f, indent=2, ensure_ascii=False)
#
# exit(0)

lost_pictures = 0
video_links = list(all_classes.keys())
np.random.shuffle(video_links)
all_classes = {key: all_classes[key] for key in video_links}



def process_frame(lost_pictures, image_counter, img_index, previous_img,
                  hamming_dist_difference=5):

    _, frame = cap.read()
    if frame is None:
        return lost_pictures, image_counter, img_index, frame
    if previous_img is not None:
        hamming_dist = calculate_pictures_similarity(frame, previous_img)
        if hamming_dist < hamming_dist_difference:
            return lost_pictures, image_counter, img_index, frame
    most_similar_quadruples = []
    # gets all detection coordinates within one frame
    for original in all_classes[video_link][timestamp]:
        detected = get_roi_coordinates(model, frame=frame)
        # takes coordinates except the class iD
        original_split = list(map(float, original[original.find(" "):].split()))
        # Calculate distances within annotated and real detection
        distances = [euclidean_distance([original_split[:2], original_split[2:]], quad) for quad in detected]
        if len(distances) == 0:
            # print("strange", file=sys.stderr)
            continue
        print(np.min(distances))
        if np.min(distances) > 0.1:
            continue
        # gets the most similar quadruple
        most_similar_index = np.argmin(distances)
        detected[most_similar_index] = [*detected[most_similar_index][0], *detected[most_similar_index][1]]
        most_similar_quadruples.append(
            f"{original[:original.find(' ')]} {detected[most_similar_index][0]} {detected[most_similar_index][1]}"
            f" {detected[most_similar_index][2]} {detected[most_similar_index][3]}\n")
    if len(most_similar_quadruples) == 0:
        return lost_pictures, image_counter, img_index, frame
    all_classes[video_link][timestamp] = most_similar_quadruples

    if image_counter > last_train_sample:
        save_name = f"{SAVE_PATH}/{czech_railway_folder}/val/"
    else:
        save_name = f"{SAVE_PATH}/{czech_railway_folder}/train/"
    cv2.imwrite(f"{save_name}images/multi_class/{img_index}.jpg", frame)

    with open(
            f"{save_name}labels/multi_class/{img_index}.txt",
            mode="w") as label_f:
        coordinates = ""
        save_annotated_picture(all_classes[video_link][timestamp], frame, f"{save_name}images/multi_class_annotated/{img_index}.jpg")
        for ii in all_classes[video_link][timestamp]:
            coordinates += f"{ii}\n"
            list(map(float, original[ii.find(" "):].split()))
        label_f.write(coordinates)
    img_index += 1
    image_counter += 1
    return lost_pictures, image_counter, img_index, frame



for video_link in all_classes:
    previous_img = None
    timestamps_shuffled = list(all_classes[video_link].keys())
    # random.shuffle(timestamps_shuffled)
    d_video = download_video(video_link, args.in_dir, use_internet=False)
    if d_video is None:
        continue
    cap = cv2.VideoCapture(args.in_dir + "/" + d_video)
    for timestamp in timestamps_shuffled:
        for seconds_before in range(args.mili_seconds_before, 0, -args.delta_step):
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(fps * (timestamp - seconds_before / 1000.))
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    frame_number)
            lost_pictures, image_counter, img_index, previous_img = process_frame(lost_pictures,
                                                                                  image_counter,
                                                                                  img_index, previous_img)

        for seconds_after in range(0, args.mili_seconds_after, args.delta_step):
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(fps * (timestamp + seconds_after / 1000.))
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    frame_number)
            lost_pictures, image_counter, img_index, previous_img = process_frame(lost_pictures,
                                                                                  image_counter,
                                                                                  img_index, previous_img)


print("lost/total", lost_pictures, "/", total_pictures_count)
