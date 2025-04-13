import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from general_utils import get_jpg_files

# Load YOLOv5 model
model = YOLO('../../reconstructed/120_lights_0_yolov8n.pt_0.5/weights/best.pt')

# Open video file
video_path = "../../videos/"
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
                 for fr in [model(frame, conf=0.75, iou=0.2, verbose=False)[0].plot()
                  for frame in tqdm(frames)]]
        cap.release()
        out.release()
    except:
        cap.release()
        out.release()
