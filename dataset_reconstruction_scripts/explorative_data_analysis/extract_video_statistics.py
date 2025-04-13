import argparse
import json
import os
from utils.general_utils import download_video, str2bool, analyze_video_characteristics, \
    visualize_brightness_distribution

parser = argparse.ArgumentParser(description='')

parser.add_argument('--sequences_jsom_path', default="../../railway_datasets/video_names_test.json")
parser.add_argument('--in-dir', default="../../videos")
parser.add_argument('--out-dir', default="../../reconstructed/stats")

args = parser.parse_args()
# where to save
SAVE_PATH = args.out_dir

with open(args.sequences_jsom_path, encoding="utf-8", mode="r") as f:
    traffic_lights = dict(json.load(f))["names"]
all_metadata = {}
for i in traffic_lights.values():
    d_video = download_video(i, args.in_dir)
    video_name = d_video
    metadata = analyze_video_characteristics(args.in_dir+ "/" + d_video, sample_interval=3056)
    all_metadata[video_name[:-4]] = metadata
    # visualize_brightness_distribution(output_path=f"{SAVE_PATH}/{video_name[:-4]}", metadata=metadata)
    del metadata["frame_count"]
    del metadata["brightness_histogram"]
with open(f"{SAVE_PATH}/summary.json", mode="w") as f:
        json.dump(all_metadata, f, indent=2)
