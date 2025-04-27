
import os
import json

from utils.general_utils import print_statistics

base_dir = "../../reconstructed/test_yolo"
yolo_v = ""

results = {}

#  For each item in
# the directory, it checks if it is a subdirectory. If it is a subdirectory, it then iterates over the
# contents of the "traffic light" directory within the specific subdirectory.
for i in os.listdir(base_dir):
    if os.path.isdir(f"{base_dir}/{i}"):
        for j in os.listdir(f"{base_dir}/{i}/"):
            if ".DS_Store" == j:
                continue
            elif j.find('box') != -1:
                res = tuple(j.replace('box', '').split("_"))
                timestamp = res[0]
                try:
                    results[i].append(float(timestamp))
                except KeyError:
                    results[i] = [float(timestamp)]
    results[i] = sorted(results[i])
with open("test_times.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
