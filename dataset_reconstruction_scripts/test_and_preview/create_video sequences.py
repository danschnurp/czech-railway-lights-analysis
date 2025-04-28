import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

from classifiers.combined_model import CzechRailwayLightModel

print(CzechRailwayLightModel)


def get_jpg_files(path):
    """
    The function `get_jpg_files` retrieves a list of all JPG and JPEG files within a specified directory
    and its subdirectories.

    :param path: The `get_jpg_files` function you provided is designed to retrieve a list of all JPG and
    JPEG files within a specified directory and its subdirectories. To use this function, you need to
    provide the `path` parameter, which should be the directory path where you want to search for JPG
    and
    :return: The function `get_jpg_files(path)` returns a list of full file paths for all the JPG/JPEG
    files found in the specified directory `path` and its subdirectories.
    """
    jpg_files = []

    # Walk through directory and subdirectories
    for root, dirs, files in os.walk(path):
        # Find all jpg/jpeg files in current directory
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                # Create full file path and add to list
                full_path = os.path.join(root, file)
                jpg_files.append(full_path)

    return jpg_files


# Load YOLOv5 model
model = torch.load('./two_stage_czech_railway_lights_model.pt')

# Open video file
video_path = "/Volumes/zalohy/test_videos/"
times_path = "../../reconstructed/test_yolo"

to_process = {}
for j in os.listdir(times_path):
    to_process[j] = sorted([float(i[i.rfind("/")+1:].replace("_box.jpg", "")) for i in get_jpg_files(times_path + "/" + j) if "_box.jpg" in i])
    try:
        cap = cv2.VideoCapture(video_path + j + ".mp4")
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(j + '.mp4', fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            for i in tqdm(to_process[j]):

                frame_number = int(fps * (i - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        frame_number)
                frames = []
                for _ in range(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), int(np.ceil(cap.get(cv2.CAP_PROP_POS_FRAMES) + (fps * 2)))):
                    ret, frame = cap.read()
                    frames.append(frame)
                    if not ret:
                        break
                [out.write(fr)
                 for fr in [model(frame, conf=0.75, iou=0.45, verbose=False)[0].plot()
                  for frame in frames]]
            cap.release()
            out.release()
    except:
        cap.release()
        out.release()
