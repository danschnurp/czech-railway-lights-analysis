from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import os
import argparse
from pytube import YouTube
import time

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='yolov5mu.pt')
parser.add_argument('-l', '--link', default="https://youtu.be/8rtVqE2yclo")
parser.add_argument('--skip_minutes', type=int, default=0)


args = parser.parse_args()

# where to save
SAVE_PATH = "./videos"  # to_do

# link of the video to be downloaded
link = args.link
try:
    # object creation using YouTube
    yt = YouTube(link)
except:
    # to handle exception
    print("Connection Error")

# Get all streams and filter for mp4 files
mp4_streams = yt.streams.filter(resolution="720p")

# get the video with the highest resolution
d_video = mp4_streams[0]

try:
    print('downloading the video' + d_video.default_filename)

    d_video.download(output_path=SAVE_PATH)
    print('Video downloaded successfully!')
except Exception:
    print("Some Error!")

interesting_labels = {'traffic light'}

nett_name = args.nett_name

video_name = d_video.default_filename
# creating folder with video name
if video_name[:-4] not in os.listdir("./videos/"):
    os.mkdir(f"./videos/{video_name[:-4]}")

# creating folder with yolo type and label folders
if nett_name[:-3] not in os.listdir(f"./videos/{video_name[:-4]}"):
    os.mkdir(f"./videos/{video_name[:-4]}/{nett_name[:-3]}/")
    for i in interesting_labels:
        os.mkdir(f"./videos/{video_name[:-4]}/{nett_name[:-3]}/{i}/")

# Load a model
model = YOLO(nett_name)  # load an official model


# Load video
video_path = 'videos/' + video_name
cap = cv2.VideoCapture(video_path)


image_index = 0
dropout_time = 0
seek_minutes = args.skip_minutes
#
cap.set(cv2.CAP_PROP_POS_MSEC,
        seek_minutes * 1000 * 60
        )

t1 = 10
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if time.time() - t1 - dropout_time > 0.05:
        dropout_time = 0
        # timestamp seconds from video beginning
        timestamp = f"{float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.):.3f}"
        results = model.predict(frame)
        # Iterate over the results
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            class_indices = boxes.cls  # Class indices of the detections
            class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
            print(class_names)
            if len(interesting_labels & set(class_names)) > 0:
                # saves the result
                save_name = f"./videos/{video_name[:-4]}/{nett_name[:-3]}/" \
                            f"{list(interesting_labels & set(class_names))[0]}/{timestamp}"
                dropout_time = 0.1
                # cv2.imwrite(
                #     save_name + "_clean.jpg",
                #     frame)
                for r in results:
                    annotator = Annotator(frame)
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                        c = box.cls
                        annotator.box_label(b, model.names[int(c)])

                img = annotator.result()
                # saves the result with bounding box
                cv2.imwrite(save_name + "_box.jpg", img)

                image_index += 1
        t1 = time.time()
cap.release()
