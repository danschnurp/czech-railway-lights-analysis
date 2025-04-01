import json

from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os
import argparse
from utils.general_utils import download_video
import time

from utils.image_utils import MovementDetector

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='yolov5mu.pt')
parser.add_argument('--sequences_jsom_path', default="../../railway_datasets/video_names.json")
parser.add_argument('--in-dir', default="../../videos")
parser.add_argument('--out-dir', default="../reconstructed/all_yolov5m")
parser.add_argument('--skip_seconds', type=int, default=0)

args = parser.parse_args()

SAVE_PATH = args.out_dir
LOAD_PATH = args.in_dir

with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))



def annotate_video():

    interesting_labels = {'traffic light'}

    nett_name = args.nett_name

    video_name = d_video
    # creating folder with video name
    try:
        if video_name[:-4] not in os.listdir(f"{SAVE_PATH}/"):
            os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}")

    except FileExistsError as e:
        print(e, "maybe different encoding")

    # creating folder with yolo type and label folders
    if nett_name[:-3] not in os.listdir(f"{SAVE_PATH}/{video_name[:-4]}"):
        os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}/{nett_name[:-3]}/")
        for i in interesting_labels:
            os.mkdir(f"{SAVE_PATH}/{video_name[:-4]}/{nett_name[:-3]}/{i}/")

    # Load a model
    model = YOLO(nett_name)  # load an official model

    # Load video
    video_path = LOAD_PATH + '/' + video_name
    cap = cv2.VideoCapture(video_path)

    image_index = 0
    dropout_time = 0
    skip_seconds = args.skip_seconds
    #
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * skip_seconds)
    cap.set(cv2.CAP_PROP_POS_FRAMES,
            frame_number)

    t1 = 5

    ret, frame = cap.read()

    mov_detector = MovementDetector(frame=frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if time.time() - t1 - dropout_time > 0.05:
            movement = mov_detector.detect_movement(frame)
            if not movement:
                dropout_time = 5
                continue
            dropout_time = 0.5
            # timestamp seconds from video beginning
            timestamp = f"{float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.):.3f}"
            results = model.predict(frame)
            # Iterate over the results
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                class_indices = boxes.cls  # Class indices of the detections
                class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
                print(class_names)
                if len(interesting_labels & set(class_names)) > 0:
                    # saves the result
                    save_name = f"{SAVE_PATH}/{video_name[:-4]}/{nett_name[:-3]}/" \
                                f"{list(interesting_labels & set(class_names))[0]}/{timestamp}"
                    dropout_time = 0.5
                    cv2.imwrite(
                        save_name + "_clean.jpg",
                        frame)
                    # for r in results:
                    #     annotator = Annotator(frame, line_width=2)
                    #     boxes = r.boxes
                    #     for box in boxes:
                    #         b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                    #         c = box.cls
                    #         annotator.box_label(b, model.names[int(c)])
                    image_index += 1
            t1 = time.time()
    cap.release()


for i in traffic_lights:
    d_video = download_video(i, LOAD_PATH)
    annotate_video()
