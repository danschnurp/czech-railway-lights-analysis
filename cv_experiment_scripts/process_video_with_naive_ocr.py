import easyocr

import cv2
import os
import argparse

from utils.general_utils import download_video
import time

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='yolov5mu.pt')
parser.add_argument('-l', '--link', default="https://youtu.be/1rLqhB7VlYE")
parser.add_argument('--skip_seconds', type=int, default=0)

args = parser.parse_args()

# where to save
SAVE_PATH = "./videos"  # to_do

# link of the video to be downloaded
link = args.link

d_video = download_video(link, SAVE_PATH)

interesting_labels = {'ocr'}


video_name = d_video
# creating folder with video name
if video_name[:-4] not in os.listdir("./videos/"):
    os.mkdir(f"./videos/{video_name[:-4]}")
    for i in interesting_labels:
        os.mkdir(f"./videos/{video_name[:-4]}/{i}/")



# Load video
video_path = 'videos/' + video_name
cap = cv2.VideoCapture(video_path)


# Set confidence threshold
confidence_threshold = 0.3

image_index = 0
dropout_time = 0.01
skip_seconds = args.skip_seconds
#
fps = cap.get(cv2.CAP_PROP_FPS)
frame_number = int(fps * skip_seconds)
cap.set(cv2.CAP_PROP_POS_FRAMES,
                        frame_number)

reader = easyocr.Reader(['en'])  # Specify language(s) as needed

t1 = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if time.time() - t1 - dropout_time > 0.05:
        # timestamp seconds from video beginning
        timestamp = f"{float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.):.3f}"
        # Perform OCR
        results = reader.readtext(frame)

        if len(results) == 0:
            print(f"{timestamp} no detections...")
            t1 = time.time()
            continue

        # Process and display results
        for (bbox, text, prob) in results:
            index_for_save = 0
            if prob > confidence_threshold:
                index_for_save += 1
                # Unpack the bounding box
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))

                # Draw the bounding box
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                # Put the text and probability
                cv2.putText(frame, f"{text}", (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                print(f"Text: {text}")
                print(f"Bounding Box: {bbox}")
                print(f"Confidence: {prob}")
                print("---")
        save_name = f"./videos/{video_name[:-4]}/" \
                    f"ocr/{timestamp}"
        if index_for_save > 0:
            cv2.imwrite(save_name + "_box.jpg", frame)
        t1 = time.time()
cap.release()
