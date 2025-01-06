# Czech Railway Traffic light detections from YouTube channels Parníci CZ and strojvedoucicom
This dataset provides traffic light detections extracted 
from YouTube videos using the **YOLOv5mu** and **YOLOv10m** object detection
model. The videos are in ** 1080p and  720p resolution at 60 and 30 frames per
second** (fps). The dataset is stored in the JSON file format.


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

