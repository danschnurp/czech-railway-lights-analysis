import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

""""
 create_video_sequences with Temporal Confidence Smoothing 
"""

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

# Load YOLO model
model = YOLO('../../reconstructed/100_lights_2_yolov10n.pt_0.55/weights/best.pt')

# Open video file
video_path = "/Volumes/zalohy/test_videos"
times_path = "../../reconstructed/test"

to_process = {}
for j in reversed(os.listdir(times_path)):
    to_process[j] = sorted([float(i[i.rfind("/")+1:].replace("_box.jpg", "")) for i in get_jpg_files(times_path + "/" + j) if "_box.jpg" in i])
    try:
        cap = cv2.VideoCapture(video_path + j + ".mp4")
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(j + '_smoothing.mp4', fourcc, fps, (frame_width, frame_height))

        # Dictionary to store confidence history for each object
        confidence_history = {}
        # Number of frames to keep in history for smoothing
        history_length = 5
        # Confidence threshold
        confidence_threshold = 0.5

        while cap.isOpened():
            for i in tqdm(to_process[j]):
                frame_number = int(fps * (i - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                                    int(np.ceil(cap.get(cv2.CAP_PROP_POS_FRAMES) + (fps * 3))))):
                    ret, frame = cap.read()

                    if not ret:
                        break

                    # Run YOLO detection but don't plot yet
                    results = model(frame, conf=0.5, iou=0.5, verbose=False)[0]

                    # Dictionary to store current frame's detections for processing
                    current_detections = {}

                    # Process each detection
                    for det_idx, detection in enumerate(results.boxes.data):
                        x1, y1, x2, y2, conf, cls = detection

                        # Create a unique identifier for this object based on its class and position
                        # This is a simple approach - you might need a more sophisticated tracking method
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        obj_class = int(cls)

                        # Create an object ID (simple version - in practice use a tracker)
                        obj_id = f"{obj_class}_{int(center_x // 20)}_{int(center_y // 20)}"

                        # Store this detection
                        current_detections[obj_id] = {
                            'box': [x1, y1, x2, y2],
                            'conf': conf,
                            'cls': obj_class
                        }

                        # Update confidence history
                        if obj_id not in confidence_history:
                            confidence_history[obj_id] = []

                        confidence_history[obj_id].append(float(conf))

                        # Keep only the last N frames
                        if len(confidence_history[obj_id]) > history_length:
                            confidence_history[obj_id].pop(0)

                    # Create a new results object with smoothed confidence scores
                    import copy

                    smoothed_results = copy.deepcopy(results)
                    filtered_boxes = []
                    filtered_cls = []
                    filtered_conf = []

                    # Apply temporal smoothing for each object
                    for obj_id, detection in current_detections.items():
                        # Calculate average confidence
                        avg_confidence = sum(confidence_history[obj_id]) / len(confidence_history[obj_id])

                        # Only keep detections with smoothed confidence above threshold
                        if avg_confidence > confidence_threshold:
                            x1, y1, x2, y2 = detection['box']
                            filtered_boxes.append([x1, y1, x2, y2])
                            filtered_cls.append(detection['cls'])
                            filtered_conf.append(avg_confidence)

                    # Clean up old object histories that haven't been seen recently
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    keys_to_remove = []
                    for obj_id in confidence_history:
                        if obj_id not in current_detections:
                            confidence_history[obj_id].append(0)  # Add zero confidence if not detected
                            if len(confidence_history[obj_id]) > history_length:
                                confidence_history[obj_id].pop(0)

                            # If object hasn't been confidently detected for several frames, remove it
                            if sum(confidence_history[obj_id]) / len(confidence_history[obj_id]) < 0.1:
                                keys_to_remove.append(obj_id)

                    for key in keys_to_remove:
                        del confidence_history[key]

                    # Create a frame with the smoothed detections
                    frame_with_boxes = frame.copy()
                    for i in range(len(filtered_boxes)):
                        x1, y1, x2, y2 = map(int, filtered_boxes[i])
                        cls = int(filtered_cls[i])
                        conf = filtered_conf[i]

                        # Draw bounding box with class name and confidence
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f"{results.names[cls]}: {conf:.2f}"
                        cv2.putText(frame_with_boxes, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Write the frame with smoothed boxes
                    out.write(frame_with_boxes)

            cap.release()
            out.release()
    except Exception as e:
        print(e)
        cap.release()
        out.release()