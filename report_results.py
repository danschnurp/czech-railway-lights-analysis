
import os
import json

base_dir = "./videos"
yolo_v = "yolov10m"

results = {}

for i in os.listdir(base_dir):
    if os.path.isdir(f"{base_dir}/{i}"):
        results[i] = []
        for j in os.listdir(f"{base_dir}/{i}/yolov5mu/traffic light/"):
            results[i].append((float(j.replace('_box.jpg', ''))))
        results[i] = sorted(results[i])
with open("today_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
