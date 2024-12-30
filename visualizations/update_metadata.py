import argparse
import json
import os
import unicodedata

from utils.general_utils import get_jpg_files
from utils.image_utils import detect_red_without_stats

parser = argparse.ArgumentParser()
parser.add_argument("--workdir", default="/Volumes/zalohy 1/dip/all_yolov5",
                    type=str, help="Path to the directory with images to process")
parser.add_argument("--output_dir", default="../dataset/reconstructed/",
                    type=str, help="Path to the output directory")
args = parser.parse_args()
workdir = args.workdir
output_dir = args.output_dir


def update_verified_metadata(metadata: dict, verified_dir="../dataset/reconstructed/green"):
    metadata = metadata["data"]
    picture_ids = []
    for i in os.listdir(verified_dir):
        try:

            #     strip .jpg extension
            picture_ids.append(int(i[:-4]))
        except ValueError:
            continue
    picture_ids.sort()
    verified_metadata = []
    for i in metadata:
        if i["color"] == verified_dir.split("/")[-1] and i["ID"]  in picture_ids:
            verified_metadata.append(i)
            print(i)

    with open(f"{output_dir}/today_results.json", mode="w", encoding="utf-8") as f:
        json.dump({"data":verified_metadata}, f, indent=2)


def add_yt_links():
    with open("../traffic_lights.json", encoding="utf-8", mode="r") as f:
        traffic_lights = dict(json.load(f))

    video_names = traffic_lights["names"]
    with open(f"{output_dir}/today_results.json", encoding='utf-8', mode="r") as f:
        colored_data = dict(json.load(f))["data"]

    video_names = [unicodedata.normalize('NFC', i.replace("⧸", "").replace("/", "").replace("#", "").replace(",", "").replace(".", "")) for i in  video_names.keys()]
    video_names = dict(zip(video_names, traffic_lights["names"].values()))
    for i in colored_data:
        i["ytlink"] = video_names[str(unicodedata.normalize('NFC',i["video name"])).replace("⧸", "").replace("/", "").replace("#", "").replace(",", "").replace(".", "")]

    with open(f"{output_dir}/today_results.json", mode="w", encoding="utf-8") as f:
        json.dump({"data":colored_data}, f, indent=2)

# def add_roi_index():
#     with open(f"{output_dir}/today_results.json", encoding='utf-8', mode="r") as f:
#         colored_data = dict(json.load(f))["data"]
#     files = get_jpg_files("/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed/roi_red")
#     for i in files:
#         split = i.split("/")
#         for j in colored_data:
#             if j["video name"] == split[-4]:
#                 if float(split[-1].split("_")[0]) == j['timestamp in video']:
#                     #  todo add multiple roi possibilities
#                     try:
#                         j["roi index"].append(int(split[-1].split("_")[1].split(".")[0][-1]))
#                     except KeyError:
#                         j["roi index"] = [int(split[-1].split("_")[1].split(".")[0][-1])]
#     with open(f"{output_dir}/today_results.json", mode="w", encoding="utf-8") as f:
#         json.dump({"data":colored_data}, f, indent=2)


def update_metadata(verified_dir = "../dataset/reconstructed/red"):
    """
    The function `update_metadata` updates the metadata by performing the following steps:
    1. Reads the metadata from a JSON file.
    2. Updates the verified metadata.
    3. Adds YouTube links to the metadata.
    4. Adds ROI (Region of Interest) indices to the metadata.
    """
    with open(f"{output_dir}/metadata.json", encoding='utf-8', mode="r") as f:
        metadata = dict(json.load(f))
    update_verified_metadata(metadata, verified_dir = verified_dir)
    add_yt_links()




print()
update_metadata()
# print()
# update_metadata(verified_dir = "../dataset/reconstructed/green")