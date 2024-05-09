from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator 
import os

if "predicted" not in os.listdir("./videos/"):
    os.mkdir("./videos/predicted/")

# Load a model
model = YOLO('yolov5n.pt')  # load an official model



# Load video
video_path = 'videos/4K  Broumov - Meziměstí  854 Hydra + 954 Bfbrdtn794.mp4' 
cap = cv2.VideoCapture(video_path)

import time

image_index = 0

dropout_time = 0

t1 = 10
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if time.time() - t1 - dropout_time > 0.1:
        dropout_time = 0
        results = model.predict(frame)

           # Iterate over the results
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            class_indices = boxes.cls  # Class indices of the detections
            class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
            print(class_names)

            if "traffic" in  str(class_names) or 'stop sign' in  str(class_names):

                dropout_time = 2

                for r in results:
                    
                    annotator = Annotator(frame)
                    
                    boxes = r.boxes
                    for box in boxes:
                        
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                        c = box.cls
                        annotator.box_label(b, model.names[int(c)])
                
                img = annotator.result()  

       

                # Display the result
                cv2.imwrite(f"./videos/predicted/{image_index}.jpg", img)   
                image_index += 1 
 
        t1 = time.time()


cap.release()
cv2.destroyAllWindows()


     
