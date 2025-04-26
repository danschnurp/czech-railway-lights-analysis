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

parser.add_argument('--nett_name', default="./100_lights_2_yolov10n_0.55.pt")
parser.add_argument('--sequences_jsom_path', default="../railway_datasets/video_names_test.json")
parser.add_argument('--in-dir', default="/Volumes/zalohy/test_videos")
parser.add_argument('--out-dir', default="./")
parser.add_argument('--mili_seconds_before', type=float, default=4500)
parser.add_argument('--mili_seconds_after', type=float, default=500)
parser.add_argument('--delta_step', type=float, default=500)
parser.add_argument('--label-light', type=int, default=79)
parser.add_argument('--train-test-split', type=int, default=0.25)

args = parser.parse_args()

# where to save
SAVE_PATH = args.out_dir

czech_railway_folder = "czech_railway_dataset_test"
classes_dir_path = "../railway_datasets/test_metadata.json"
dataset_yaml = '../metacentrum/CRL_extended.yaml'

img_index = 0

with open(dataset_yaml, encoding="utf-8") as f:
    class_mapping = yaml.load(f, Loader=yaml.SafeLoader)

class_mapping = class_mapping["names"]
class_mapping = dict(zip(class_mapping.values(), class_mapping.keys()))

with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    video_names = dict(json.load(f))
    video_names = video_names["names"]
data_ordered_videos = dict(zip(list(video_names.values()), [[] for _ in range(len(video_names))]))



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

with open(classes_dir_path, encoding="utf-8") as f:
    all_classes = dict(json.load(f))["data"]



total_pictures_count = len(all_classes)

[data_ordered_videos[i["ytlink"]].append(i) for i in all_classes]

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
all_classes = data_ordered_videos



def process_frame(lost_pictures, image_counter, img_index, previous_img,
                  hamming_dist_difference=5, metadata=None):
    _, frame = cap.read()
    if frame is None:
        return lost_pictures, image_counter, img_index, frame, metadata
    if previous_img is not None:
        hamming_dist = calculate_pictures_similarity(frame, previous_img)
        if hamming_dist < hamming_dist_difference:
            return lost_pictures, image_counter, img_index, frame, metadata
    # gets all detection coordinates within one frame
    detected = get_roi_coordinates(model, frame=frame)
    if len(detected) == 0:
        return lost_pictures, image_counter, img_index, frame, metadata
    # takes coordinates except the class iD
    # Calculate distances within annotated and real detection
    # gets the most similar quadruple
    most_similar_quadruple_metrics = [euclidean_distance(np.fromstring(metadata["roi coordinates"], sep=" ").reshape(2,2), detection) for detection in detected]
    most_similar_quadruple_index = most_similar_quadruple_metrics.index(np.min(most_similar_quadruple_metrics))
    most_similar_quadruple = (f"{class_mapping[metadata['color']]}"
                                        f" {detected[most_similar_quadruple_index][0][0]} "
                                         f"{detected[most_similar_quadruple_index][0][1]}"
                                         f" {detected[most_similar_quadruple_index][1][0]}"
                                         f" {detected[most_similar_quadruple_index][1][1]}\n")
    if image_counter > last_train_sample:
        save_name = f"{SAVE_PATH}/{czech_railway_folder}/val/"
    else:
        save_name = f"{SAVE_PATH}/{czech_railway_folder}/train/"
    cv2.imwrite(f"{save_name}images/multi_class/{img_index}.jpg", frame)

    with open(
            f"{save_name}labels/multi_class/{img_index}.txt",
            mode="w+") as label_f:
        label_f.write(f"{most_similar_quadruple}\n")
    image_counter += 1
    return lost_pictures, image_counter, img_index, frame, metadata



for video_link in all_classes:
    previous_img = None
    timestamps_shuffled = all_classes[video_link]
    # random.shuffle(timestamps_shuffled)
    d_video = download_video(video_link, args.in_dir, use_internet=False, names_jsom_path=args.sequences_jsom_path)
    if d_video is None:
        continue
    cap = cv2.VideoCapture(args.in_dir + "/" + d_video)
    for metadata in timestamps_shuffled:

        timestamp = metadata["timestamp in video"]
        for seconds_before in range(args.mili_seconds_before, 0, -args.delta_step):
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(fps * (timestamp - seconds_before / 1000.))
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    frame_number)
            lost_pictures, image_counter, img_index, previous_img, metadata = process_frame(lost_pictures,
                                                                                  image_counter,
                                                                                  frame_number, previous_img, metadata=metadata)

        for seconds_after in range(0, args.mili_seconds_after, args.delta_step):
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(fps * (timestamp + seconds_after / 1000.))
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    frame_number)
            lost_pictures, image_counter, img_index, previous_img, metadata = process_frame(lost_pictures,
                                                                                  image_counter,
                                                                                  frame_number, previous_img, metadata=metadata)


print("lost/total", lost_pictures, "/", total_pictures_count)
