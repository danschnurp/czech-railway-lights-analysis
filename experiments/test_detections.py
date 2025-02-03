import json

from ultralytics import YOLO

from general_utils import get_times_by_video_name, download_video
from image_utils import test_roi_detections
import argparse

import json

from utils.general_utils import download_video, str2bool
from utils.image_utils import get_pictures

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='../yolov5mu.pt')
parser.add_argument('--sequence_seconds_before', type=float, default=0.002)
parser.add_argument('--sequence_seconds_after', type=float, default=0.002)


args = parser.parse_args()
# where to save


with open("../railway_datasets/non_traffic_lights/crossing_light.json", encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))
model= YOLO(args.nett_name)
for i in traffic_lights["data"]:
    d_video = download_video(i["ytlink"], "../videos")
    test_roi_detections(d_video, i, args, SAVE_PATH="./",  model=model )
