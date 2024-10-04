from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os
import argparse

import json

from utils import download_video, crop_bounding_box, str2bool, enlarge_bounding_box

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='yolov5mu.pt')
parser.add_argument('--sequences_jsom_path', default="../traffic_lights.json")
parser.add_argument('--sequence_seconds_before', type=float, default=0.01)
parser.add_argument('--sequence_seconds_after', type=float, default=0.01)
parser.add_argument('--clean_pictures', default=False)
parser.add_argument('--bounding_box_pictures', default=False)
parser.add_argument('--roi_pictures', default=True)

args = parser.parse_args()
args.clean_pictures = str2bool(args.clean_pictures)
args.bounding_box_pictures = str2bool(args.bounding_box_pictures)
args.roi_pictures = str2bool(args.roi_pictures)

if "reconstructed" not in os.listdir("./") or not os.path.isdir("./reconstructed"):
    os.mkdir("./reconstructed")

# where to save
SAVE_PATH = "./reconstructed"

with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))

del traffic_lights["names"]
del traffic_lights["todo"]


def get_pictures(d_video, seek_seconds):

    interesting_labels = {'traffic light'}

    nett_name = args.nett_name

    video_name = d_video
    # creating folder with video name
    if video_name[:-4] not in os.listdir(SAVE_PATH):
        os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}")

    # creating folder with yolo type and label folders
    if nett_name[:-3] not in os.listdir(f"{SAVE_PATH}/{video_name[:-4]}"):
        os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}/{nett_name[:-3]}/")
        for i in interesting_labels:
            os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}/{nett_name[:-3]}/{i}/")

    # Load a model
    model = YOLO(nett_name)  # load an official model

    # Load video
    video_path = SAVE_PATH + '/' + video_name
    cap = cv2.VideoCapture(video_path)

    image_index = 0
    #
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
            return
        else:
            # timestamp seconds from video beginning
            timestamp = f"{float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.):.3f}"
            results = model.predict(frame, conf=0.5)
            # Iterate over the results
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                class_indices = boxes.cls  # Class indices of the detections
                class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
                print(class_names, "timestamp:", timestamp)
                if len(interesting_labels & set(class_names)) > 0:
                    # saves the result
                    save_name = f"{SAVE_PATH}/{video_name[:-4]}/{nett_name[:-3]}/" \
                                f"{list(interesting_labels & set(class_names))[0]}/{timestamp}"
                    dropout_time = 0.1
                    if args.clean_pictures:
                        cv2.imwrite(
                            save_name + "_clean.jpg",
                            frame)
                    for r in results:
                        annotator = Annotator(frame, line_width=2)
                        boxes = r.boxes
                        for index, box in enumerate(boxes):
                            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                            new_top, new_bottom = enlarge_bounding_box(b)
                            b = [b[0], new_top, b[2], new_bottom]
                            c = box.cls
                            if model.names[int(c)] in interesting_labels:
                                if args.roi_pictures:
                                    cropped_roi = crop_bounding_box(b, frame)
                                    cv2.imwrite(f"{save_name}_roi{index}.jpg", cropped_roi)
                                annotator.box_label(b, model.names[int(c)])

                    img = annotator.result()
                    if args.bounding_box_pictures:
                        # saves the result with bounding box
                        cv2.imwrite(save_name + "_box.jpg", img)

                    image_index += 1
    cap.release()


for i in traffic_lights:
    d_video = download_video(i, SAVE_PATH)
    for j in traffic_lights[i]:
        get_pictures(d_video, j)
