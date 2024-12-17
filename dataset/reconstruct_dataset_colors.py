import argparse

import json

from utils.general_utils import download_video, str2bool
from utils.image_utils import get_pictures

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='yolov5mu.pt')
parser.add_argument('--sequences_jsom_path', default="../colored_lights.json")
parser.add_argument('--sequence_seconds_before', type=float, default=0.002)
parser.add_argument('--sequence_seconds_after', type=float, default=0.002)
parser.add_argument('--clean_pictures', default=False)
parser.add_argument('--bounding_box_pictures', default=False)
parser.add_argument('--work_dir', default="/Volumes/zalohy/dip")
parser.add_argument('--roi_pictures', default=True)

args = parser.parse_args()
# where to save
SAVE_PATH = args.work_dir

args.clean_pictures = str2bool(args.clean_pictures)
args.bounding_box_pictures = str2bool(args.bounding_box_pictures)
args.roi_pictures = str2bool(args.roi_pictures)


img_index = 0


with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))




for i in traffic_lights["data"]:
    d_video = download_video(i["ytlink"], SAVE_PATH)
    get_pictures(d_video, i["timestamp in video"], args, SAVE_PATH)
