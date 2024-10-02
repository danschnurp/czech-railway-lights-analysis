from ultralytics import YOLO
import cv2
import os
import argparse

import json

from utils import download_video, str2bool

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='yolov5mu.pt')
parser.add_argument('--sequences_jsom_path', default="../traffic_lights.json")
parser.add_argument('--sequence_seconds_before', type=float, default=0.01)
parser.add_argument('--sequence_seconds_after', type=float, default=0.01)

args = parser.parse_args()

if "reconstructed" not in os.listdir("./") or not os.path.isdir("./reconstructed"):
    os.mkdir("./reconstructed")

czech_railway_folder = "czech_railway_"
img_index = 0

# where to save
SAVE_PATH = "./reconstructed"

with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))

del traffic_lights["names"]
del traffic_lights["todo"]

interesting_labels = {'traffic light'}
label_light = 0

# creating folder with video name
if czech_railway_folder not in os.listdir(SAVE_PATH):
    os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}")
    for i in interesting_labels:
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/images/")
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/labels/")
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/images/{i}/")
        os.mkdir(f"{SAVE_PATH}/{czech_railway_folder}/labels/{i}")

for i in traffic_lights:
    d_video = download_video(i, SAVE_PATH)
    # Load a model
    model = YOLO(args.nett_name)  # load an official model

    # Load video
    video_path = SAVE_PATH + '/' + d_video

    for seek_seconds in traffic_lights[i]:
        video_name = d_video
        #
        cap = cv2.VideoCapture(video_path)
        start_time = seek_seconds - args.sequence_seconds_before
        print("starting from", seek_seconds)
        if start_time < 0.:
            print("starting from beginning")
            start_time = 0
        cap.set(cv2.CAP_PROP_POS_MSEC,
                start_time * 1000)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # end of sequence
            if (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.) > (args.sequence_seconds_after + seek_seconds):
                print(f"finished {seek_seconds}")
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
                        save_name = f"{SAVE_PATH}/{czech_railway_folder}/"
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
