# Czech Railway Traffic light detections from YouTube channels Parníci CZ and strojvedoucicom
This dataset provides traffic light detections extracted 
from YouTube videos using the **YOLOv5mu** object detection
model. The videos are in **720p resolution at 30 frames per
second** (fps). The dataset includes a JSON file named 
traffic_lights.json that contains information about the 
videos and the detected traffic lights. Each entry in the 
JSON file specifies a video name, its URL, and a list of 
timestamps (in seconds) where traffic lights were identified
within the video.

https://www.youtube.com/@parnicicz4773/videos
 
https://www.youtube.com/@strojvedouci_com

### Folder structure
- important files
```
.
├── CzechRailwayTrafficLights.yaml - config for yolo training
├── dataset/prepare_coco_format.py - prepares coco format from traffic_lights.json
├── process_video_with_yolo.py - script for new semi manual video annotations
├── dataset/reconstruct_dataset.py - reconstructs dataset by video names, detection method (yolov8n...) and classes
├── report_results.py - creates today_results.json with current detections in ./videos folder 
├── train_yolo.py - training script
└── traffic_lights.json
```


## Prerequisites

- ffmpeg
  - (windows) - install via chocolatey
    - ```
      Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
      choco install ffmpeg -y
      ```
- Python libraries
``pip install -r requirements.txt``
- run - ``yt-dlp --cookies-from-browser chrome -j  'https://www.youtube.com/watch?v=1CuJmlU0rzM'`` - to ensure u are not robot

## Reconstructing the dataset

- use script `reconstruct_dataset.py`
- script parameters:
  - **--nett_name** (default: yolov5mu.pt): Name of the pre-trained neural network model (default: yolov5mu.pt).
  - **--sequences_jsom_path** (default: ./traffic_lights.json): Path to a JSON file containing video sequences information (default: ./traffic_lights.json).
  - **--sequence_seconds_before** (default: 0.001): Number of seconds of inference to include before each timestamp (default: 0.001 seconds).
  - **--sequence_seconds_after** (default: 0.001): Number of seconds of inference to include after each timestamp (default: 0.001 seconds).
    - used for blinking states
  - **--clean_pictures** (default: True): Generate images without markings (original frames) (default: on).
  - **--bounding_box_pictures** (default: True): Generate images with bounding boxes around objects of interest (default: on).
  - **--roi_pictures** (default: True): Generate images containing only regions of interest (default: on).
