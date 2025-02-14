import sys

from ultralytics import YOLO
import cv2
import os
import argparse

import json

from utils.general_utils import download_video

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default="yolov5mu.pt")
parser.add_argument('--sequences_jsom_path', default="../traffic_lights.json")
parser.add_argument('--sequence_seconds_before', type=float, default=0.002)
parser.add_argument('--sequence_seconds_after', type=float, default=0.002)
parser.add_argument('--work-dir', default="/Volumes/zalohy/dip")
parser.add_argument('--label-light', type=int, default=80)
parser.add_argument('--train-test-split', type=int, default=0.75)

args = parser.parse_args()

# where to save
SAVE_PATH = args.work_dir


czech_railway_folder = "czech_railway_dataset"
img_index = 0


with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))

del traffic_lights["names"]
del traffic_lights["todo"]

interesting_labels = {'traffic light'}
label_light = args.label_light

# creating folder with video name
if czech_railway_folder not in os.listdir(SAVE_PATH):
    os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}")
    for i in interesting_labels:
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/")
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/")
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/images/")
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/labels/")
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/images/{i}/")
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/train/labels/{i}")
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/images/")
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/labels/")
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/images/{i}/")
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/val/labels/{i}")


image_counter = 0
for i in traffic_lights:
    image_counter += len(i)

last_train_sample = int(image_counter * args.train_test_split)

image_counter = 0

for i in traffic_lights:
    d_video = download_video(i, SAVE_PATH)
    # Load a model
    model = YOLO(args.nett_name)  # load an official model

    # Load video
    video_path = SAVE_PATH + '/' + d_video

    for seek_seconds in traffic_lights[i]:
        image_counter += 1
        video_name = d_video
        #
        cap = cv2.VideoCapture(video_path)
        start_time = seek_seconds - args.sequence_seconds_before
        print("starting from", seek_seconds, file=sys.stderr)
        if start_time < 0.:
            print("starting from beginning", file=sys.stderr)
            start_time = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(fps * start_time)
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                frame_number)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # end of sequence
            if (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.) > (args.sequence_seconds_after + seek_seconds):
                print(f"finished {seek_seconds}\n", file=sys.stderr)
                break
            else:
                # timestamp seconds from video beginning
                timestamp = f"{float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.):.3f}"
                results = model.predict(frame)
                # Iterate over the results
                for result in results:
                    boxes = result.boxes  # Boxes object for bbox outputs
                    class_indices = boxes.cls  # Class indices of the detections
                    class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
                    print(class_names, "timestamp:", timestamp)
                    if len(interesting_labels & set(class_names)) > 0:
                        # saves the result

                        if image_counter > last_train_sample:
                            save_name = f"{SAVE_PATH}/{czech_railway_folder}/val/"
                        else:
                            save_name = f"{SAVE_PATH}/{czech_railway_folder}/train/"
                        cv2.imwrite(
                            f"{save_name}images/{list(interesting_labels & set(class_names))[0]}/{img_index}.jpg",
                            frame)
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

        cap.release()
