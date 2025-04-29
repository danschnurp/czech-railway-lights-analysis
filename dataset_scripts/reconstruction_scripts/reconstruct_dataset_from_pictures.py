import argparse
import json

from utils.general_utils import str2bool, get_jpg_files
from utils.image_utils import annotate_pictures

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='yolov5mu.pt')
parser.add_argument('--sequences_jsom_path', default="../traffic_lights.json")
parser.add_argument('--sequence_seconds_before', type=float, default=0.002)
parser.add_argument('--sequence_seconds_after', type=float, default=0.002)
parser.add_argument('--clean_pictures', default=False)
parser.add_argument('--bounding_box_pictures', default=False)
parser.add_argument('--in-dir', default="/Volumes/zalohy/DIP_unannontated")
parser.add_argument('--out-dir', default="/Volumes/zalohy/dip/dataset")
parser.add_argument('--roi_pictures', default=True)

args = parser.parse_args()
# where to save
SAVE_PATH = args.out_dir

args.clean_pictures = str2bool(args.clean_pictures)
args.bounding_box_pictures = str2bool(args.bounding_box_pictures)
args.roi_pictures = str2bool(args.roi_pictures)


czech_railway_folder = "czech_railway_dataset"
img_index = 0



annotate_pictures(args, SAVE_PATH)
image_paths = get_jpg_files("/Volumes/zalohy/dip/dataset")
print(len(image_paths))
