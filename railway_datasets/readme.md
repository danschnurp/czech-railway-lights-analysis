# Czech Railway Traffic light detections from YouTube channels Parníci CZ and strojvedoucicom
This dataset provides traffic light detections extracted 
from YouTube videos using the **YOLOv5mu** and **YOLOv10m** object detection
model. The videos are in **1080p at 60 frames per
second** (fps). The dataset is stored in the JSON file format.

- datasets on Google Drive - https://drive.google.com/drive/folders/1NIlhyW1fIZfiFyOGCTeXyf0pdiRoDbbF?usp=share_link


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

## Metadata

### Preliminary experiment statistics
Moments are timestamps in video. In one timestamp can be multiple railway signals.


- traffic_lights_raw predicted by yolo
    - current size of traffic lights dataset is: 	 **6485 moments**

- true predicted by yolo and checked by human
  -  current size of traffic lights dataset is: 	 **848 moments**

    - **acc: 	 0.13** =  848 / 6485

- **yolov5m with movement detection:** 6485 moments
- **yolov10m:** 10404 moments
- **yolov5m:** 10895 moments 

### CRL - extended with CVAT hand made annotation
- train 1502 objects
- valid 552 objects