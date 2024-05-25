from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator 
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', '--movie', default="4K  Bezdružice - Plzeň  842 Kvatro.mp4")

args = parser.parse_args()

interesting_labels = {'traffic sign', 'traffic light'}

name = args.movie

if name[:-4] not in os.listdir("./videos/"):
    os.mkdir(f"./videos/{name[:-4]}/")
    for i in interesting_labels:
        os.mkdir(f"./videos/{name[:-4]}/{i}/")

# Load a model
model = YOLO('yolov8m.pt')  # load an official model



# Load video
video_path = 'videos/' + name
cap = cv2.VideoCapture(video_path)

import time

image_index = 0

dropout_time = 0



t1 = 10
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if time.time() - t1 - dropout_time > 0.05:
        dropout_time = 0
        results = model.predict(frame)

           # Iterate over the results
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            class_indices = boxes.cls  # Class indices of the detections
            class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
            print(class_names)

            if len(interesting_labels & set(class_names)) > 0:

                dropout_time = 1

                for r in results:
                    
                    annotator = Annotator(frame)
                    
                    boxes = r.boxes
                    for box in boxes:
                        
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                        c = box.cls
                        annotator.box_label(b, model.names[int(c)])
                
                img = annotator.result()  

       

                # Display the result
                cv2.imwrite(f"./videos/{name[:-4]}/{list(interesting_labels & set(class_names))[0]}/{image_index}.jpg", img)   
                image_index += 1 
 
        t1 = time.time()

cap.release()


     
