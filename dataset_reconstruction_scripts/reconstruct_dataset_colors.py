import argparse

import json

import yaml

from utils.general_utils import download_video, str2bool
from utils.image_utils import get_pictures

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='yolov5mu.pt')
parser.add_argument('--sequences_jsom_path', default="../railway_datasets/simple_classes/stop.json")
parser.add_argument('--sequence_seconds_before', type=float, default=0.002)
parser.add_argument('--sequence_seconds_after', type=float, default=0.002)
parser.add_argument('--clean_pictures', default=True)
parser.add_argument('--bounding_box_pictures', default=True)
parser.add_argument('--in-dir', default="../videos")
parser.add_argument('--out-dir', default="../reconstructed/all_yolov5mu_colors")
parser.add_argument('--roi_pictures', default=True)

args = parser.parse_args()
# where to save
SAVE_PATH = args.out_dir

args.clean_pictures = str2bool(args.clean_pictures)
args.bounding_box_pictures = str2bool(args.bounding_box_pictures)
args.roi_pictures = str2bool(args.roi_pictures)


img_index = 0


with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))

with open("../metacentrum_experiments/CRTL_multi_labeled_transfer.yaml", encoding="utf-8") as f:
    class_mapping = yaml.load(f, Loader=yaml.SafeLoader)

class_mapping = class_mapping["names"]
class_mapping = dict(zip(class_mapping.values(), class_mapping.keys()))


for i in traffic_lights["data"]:
    d_video = download_video(i["ytlink"], args.in_dir)
    get_pictures(d_video, i["timestamp in video"], args, class_mapping)
