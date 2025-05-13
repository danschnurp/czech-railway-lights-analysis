# Computer Vision Applications in Video Recordings for Traffic Signal Detection and Classification on Czech Railways  


https://github.com/danschnurp/czech-railway-lights-analysis


https://huggingface.co/spaces/danschnurp/czech-railway-trafic-lights


Computer vision utils for detection and classification of railway traffic signals from videos on Czech Railways. Implements YOLO or RTDetr based detection pipeline. Includes custom dataset of annotated frames capturing diverse signal types according to Czech Railways standards.


This repository focuses on the application of computer vision techniques to detect and classify railway traffic signals in video recordings from Czech railways. The project aims to improve railway safety and efficiency by leveraging advanced deep learning methodologies.  

## Key Contributions  
- **Dataset Creation:** Curated and annotated a high-quality dataset of Czech railway traffic signals for training and evaluation.  
- **Detection Enhancement:** Implemented improvements to YOLO (You Only Look Once) to enhance the accuracy and performance of traffic signal detection in video recordings.  
- **Signal Classification:** Developed a classification pipeline to identify and categorize various types of railway traffic signals with high precision.  

## Methodology  
The project is structured in two key phases:  
1. **Detection Phase:**  
   - Focused on fine-tuning YOLO models for robust and real-time detection of railway traffic signals in complex environments.  
2. **Classification Phase:**  
   - Designed and trained models to accurately classify the detected signals into predefined categories, enabling better signal interpretation and action planning.  

## Repository Contents  
- **Dataset:** annotated dataset for railway traffic signals.  
- **Codebase:** Implementation of detection and classification pipelines using YOLO and other supporting tools.  

This repository provides resources for researchers and practitioners interested in the intersection of computer vision and railway signal processing. Contributions and feedback are welcome!  

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

