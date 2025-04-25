import argparse
import json
import os
import sys
import time

import cv2
import yaml
from ultralytics import YOLO

from image_utils import calculate_aspect_ratio, crop_bounding_box
from utils.general_utils import download_video, str2bool


parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='100_lights_2_yolov10n_0.55.pt')
parser.add_argument('--sequences_jsom_path', default="../railway_datasets/test_times.json")
parser.add_argument('--sequence_seconds_before', type=float, default=0.45)
parser.add_argument('--sequence_seconds_after', type=float, default=0.45)
parser.add_argument('--clean_pictures', default=True)
parser.add_argument('--bounding_box_pictures', default=False)
parser.add_argument('--in-dir', default="/Volumes/zalohy/test_videos")
parser.add_argument('--out-dir', default="../reconstructed/all_yolov5m_tst")
parser.add_argument('--roi_pictures', default=False)

args = parser.parse_args()
# where to save
SAVE_PATH = args.out_dir

args.clean_pictures = str2bool(args.clean_pictures)
args.bounding_box_pictures = str2bool(args.bounding_box_pictures)
args.roi_pictures = str2bool(args.roi_pictures)

img_index = 0


with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))

dataset_yaml = '../metacentrum/CRL_extended.yaml'

with open(dataset_yaml, encoding="utf-8") as f:
    class_mapping = yaml.load(f, Loader=yaml.SafeLoader)

class_mapping = class_mapping["names"]
class_mapping = set(zip(class_mapping.values()))

with open("../railway_datasets/video_names_test.json", encoding="utf-8", mode="r") as f:
    video_names = dict(json.load(f))["names"]

summary = {}
detected_count = 0
stats = []
for i in traffic_lights:
    d_video = download_video(video_names[i], args.in_dir)

    interesting_labels = class_mapping

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

        def get_picture(cap, model, args, interesting_labels, video_name, nett_name, image_index, SAVE_PATH,
                        stats=None):

            _, frame = cap.read()

            timestamp = f"{float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.):.3f}"
            results = model.predict(frame)
            # Iterate over the results
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                class_indices = boxes.cls  # Class indices of the detections
                class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
                print(class_names, "timestamp:", timestamp)
                if True:
                    # saves the result
                    dropout_time = 0.1
                    for r in results:
                        boxes = r.boxes
                        for index, box in enumerate(boxes):
                            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                            # new_top, new_bottom = enlarge_bounding_box(b)
                            # b = [b[0], new_top, b[2], new_bottom]
                            c = box.cls
                            if True:
                                this_roi = crop_bounding_box(b, frame)

                                stats.append({
                                    "ID": 1,
                                    "aspect ratio": float(f"{float(calculate_aspect_ratio(this_roi)[0]):.3f}"),
                                    "video name": video_name,
                                    "detection method": args.nett_name,
                                    "class": "traffic_light",
                                    "timestamp in video": float(
                                        f"{j:.3f}"),
                                    "color": model.names[int(c)],
                                    "roi coordinates": f"{box.xywhn.tolist()[0][0]} {box.xywhn.tolist()[0][1]} {box.xywhn.tolist()[0][2]} {box.xywhn.tolist()[0][3]}"
                                })
                    image_index += 1
                    return image_index, stats
            return image_index, stats
        if d_video is not None:
            start_time = j
            if start_time < 0.:
                print("starting from beginning")
                start_time = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(fps * start_time)
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    frame_number)

            detected, stats =  get_picture(cap, model, args, interesting_labels, video_name,
                                   nett_name, image_index=0, SAVE_PATH=SAVE_PATH, stats=stats)
            detected_count += detected
            if detected == 0:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_number = int(fps * (start_time - args.sequence_seconds_before))
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        frame_number)
                detected, stats =  get_picture(cap, model, args, interesting_labels, video_name,
                                        nett_name, image_index=0, SAVE_PATH=SAVE_PATH, stats=stats)
                print("second try:", j, file=sys.stderr)

                if detected == 0:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_number = int(fps * (start_time + args.sequence_seconds_before))
                    cap.set(cv2.CAP_PROP_POS_FRAMES,
                            frame_number)
                    detected, stats =  get_picture(cap, model, args, interesting_labels, video_name,
                                            nett_name, image_index=0, SAVE_PATH=SAVE_PATH, stats=stats)
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

print(stats)

with open(f"./metadata.json", mode="w", encoding="utf-8") as f:
    json.dump({"data": stats}, f, indent=2, ensure_ascii=False)

