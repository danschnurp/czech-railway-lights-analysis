
import os
import json

from utils.general_utils import print_statistics

base_dir = "./videos"
yolo_v = "yolov10m"

results = {}

print_statistics()


#  For each item in
# the directory, it checks if it is a subdirectory. If it is a subdirectory, it then iterates over the
# contents of the "traffic light" directory within the specific subdirectory.
for i in os.listdir(base_dir):
    if os.path.isdir(f"{base_dir}/{i}"):
        results[i] = []
        for j in os.listdir(f"{base_dir}/{i}/{yolo_v}/traffic light/"):
            if ".DS_Store" == j:
                continue
            results[i].append((float(j.replace('_box.jpg', ''))))
        results[i] = sorted(results[i])
with open("today_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
