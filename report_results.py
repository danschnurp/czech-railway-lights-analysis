from directory_tree import display_tree

import os
import json

base_dir = "./videos"

display_tree(base_dir)

results = {}

for i in os.listdir(base_dir):
    if os.path.isdir(f"{base_dir}/{i}"):
        results[i] = []
        for j in os.listdir(f"{base_dir}/{i}/yolov5mu/traffic light/"):
            results[i].append((float(j.replace('_box.jpg', ''))))
        results[i] = sorted(results[i])
with open("experiment_results.md", "w", encoding="utf-8") as f:
    f.write("""- yolov5mu with 720p, 30 fps\n
- traffic light detections as seconds from start\n
````
    """)
    json.dump(results, f, indent=2, ensure_ascii=False)
