from ultralytics import YOLO
import cv2
# Load a model
model = YOLO('yolov5l.pt')  # load an official model



# Load video
video_path = 'path to video' 
cap = cv2.VideoCapture(video_path)

import time

t1 = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if time.time() - t1 > 0.1:
        
        pred = model(frame)

        # Iterate over the results
        for result in pred:
            boxes = result.boxes  # Boxes object for bbox outputs
            class_indices = boxes.cls  # Class indices of the detections
            class_names = [result.names[int(cls)] for cls in class_indices]  # Map indices to names
            print(class_names)

        if "traffic" in  str(class_names):

                # Display the result
                
            cv2.imshow('YOLOv5', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        t1 = time.time()


cap.release()
cv2.destroyAllWindows()


