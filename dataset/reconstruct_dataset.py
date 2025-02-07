import argparse
import json
import os
import sys
import time

import cv2
from ultralytics import YOLO

from utils.general_utils import download_video, str2bool
from utils.image_utils import get_picture

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='yolov5mu.pt')
parser.add_argument('--sequences_jsom_path', default="../railway_datasets/annotated_traffic_lights.json")
parser.add_argument('--sequence_seconds_before', type=float, default=0.45)
parser.add_argument('--sequence_seconds_after', type=float, default=0.45)
parser.add_argument('--clean_pictures', default=True)
parser.add_argument('--bounding_box_pictures', default=True)
parser.add_argument('--in-dir', default="/Volumes/zalohy/dip")
parser.add_argument('--out-dir', default="/Volumes/zalohy/dip")
parser.add_argument('--roi_pictures', default=True)

args = parser.parse_args()
# where to save
SAVE_PATH = args.out_dir

args.clean_pictures = str2bool(args.clean_pictures)
args.bounding_box_pictures = str2bool(args.bounding_box_pictures)
args.roi_pictures = str2bool(args.roi_pictures)


czech_railway_folder = "czech_railway_dataset"
img_index = 0


with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))


summary = {}
detected_count = 0
for i in traffic_lights:
    d_video = download_video(i, args.in_dir)

    interesting_labels = {'traffic light'}

    nett_name = args.nett_name

    video_name = d_video
    try:
        # creating folder with video name
        if video_name[:-4] not in os.listdir(SAVE_PATH):
            os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}")
    except FileExistsError as e:
        print(e, "maybe different encoding")

    # creating folder with yolo type and label folders
    if nett_name[:-3] not in os.listdir(f"{SAVE_PATH}/{video_name[:-4]}"):
        os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}/{nett_name[:-3]}/")
        for k in interesting_labels:
            os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}/{nett_name[:-3]}/{k}/")

    # Load a model
    model = YOLO(nett_name)  # load an official model

    # Load video
    video_path = args.in_dir + '/' + video_name
    video_name = video_name.replace(".mp4", "")
    cap = cv2.VideoCapture(video_path)
    strange_pictures = []
    failed_pictures = []
    for j in traffic_lights[i]:
        if d_video is not None:
            start_time = j
            if start_time < 0.:
                print("starting from beginning")
                start_time = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(fps * start_time)
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    frame_number)
            detected = get_picture(cap, model, args, interesting_labels, video_name,
                                   nett_name, image_index=0, SAVE_PATH=SAVE_PATH)
            detected_count += detected
            if detected == 0:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_number = int(fps * (start_time - args.sequence_seconds_before))
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        frame_number)
                detected = get_picture(cap, model, args, interesting_labels, video_name,
                                       nett_name, image_index=0, SAVE_PATH=SAVE_PATH)
                print("second try:", j, file=sys.stderr)

                if detected == 0:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_number = int(fps * (start_time + args.sequence_seconds_before))
                    cap.set(cv2.CAP_PROP_POS_FRAMES,
                            frame_number)
                    detected = get_picture(cap, model, args, interesting_labels, video_name,
                                           nett_name, image_index=0, SAVE_PATH=SAVE_PATH)
                    detected_count += detected

                    if detected == 0:
                        print("failed:", j, file=sys.stderr)
                        failed_pictures.append(j)
                    else:
                        print("third try:", j, file=sys.stderr)
                        strange_pictures.append(j)
                else:
                    strange_pictures.append(j)
                    detected_count += detected
    cap.release()
    summary[d_video] = {"original": len(traffic_lights[i]),
                        "detected": detected_count,
                        "lost_pictures": failed_pictures.copy(),
                        "lost_pictures_count": int(len(traffic_lights[i]) - detected_count),
                        "strange_pictures": strange_pictures.copy()}

    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("---------------------summary--------------------------")
    print(json.dumps(summary[d_video], indent=2))
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    print("------------------------------------------------------")
    detected_count = 0
with open(args.in_dir + "/summary.json", mode="w") as f:
    json.dump(summary, f, indent=2)

