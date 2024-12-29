import argparse

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from utils.image_utils import crop_bounding_box, calculate_aspect_ratio, crop_sides_percentage, detect_color, red, \
    check_content_centered, calculate_nonzero_percent

parser = argparse.ArgumentParser(description='')

parser.add_argument('--nett_name', default='yolov10m.pt')
parser.add_argument('--sequences_jsom_path', default="../traffic_lights.json")
parser.add_argument('--sequence_seconds_before', type=float, default=0.002)
parser.add_argument('--sequence_seconds_after', type=float, default=0.002)
parser.add_argument('--clean_pictures', default=False)
parser.add_argument('--bounding_box_pictures', default=True)
parser.add_argument('--roi_pictures', default=True)

args = parser.parse_args()

interesting_label = "traffic light"
model = YOLO(args.nett_name)
save_name = "railway_crossing"
frame = cv2.imread("237.333_clean.jpg")

def detect_red(image):

    aspect_ratio, w, h = calculate_aspect_ratio(image)
    # image = replace_white_with_black(image)
    result_color = crop_sides_percentage(detect_color(image, color_filter=red))

    is_centered = check_content_centered(result_color)
    no_zero = calculate_nonzero_percent(result_color)
    print(f"Aspect ratio: {aspect_ratio}, width: {w}, height: {h}")
    print(f"Is centered: {is_centered}, Nonzero: {no_zero}")
    if is_centered and no_zero > 0.1:
        print("Found a red light!")
        exit(1)
    # cv2.imshow("result_color", result_color)
    # cv2.waitKey(0)


detect_red(cv2.imread("railway_crossing_roi0.jpg"))

exit(0)

results = model.predict(frame)
# Iterate over the results
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    class_indices = boxes.cls  # Class indices of the detections
    class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names

    if 1:
        # saves the result
        dropout_time = 0.1
        if args.clean_pictures:
            cv2.imwrite(
                save_name + "_clean.jpg",
                frame)
        for r in results:
            annotator = Annotator(frame, line_width=2)
            boxes = r.boxes
            for index, box in enumerate(boxes):
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                # new_top, new_bottom = enlarge_bounding_box(b)
                # b = [b[0], new_top, b[2], new_bottom]
                c = box.cls
                if args.roi_pictures:
                        cropped_roi = crop_bounding_box(b, frame)
                        try:
                            cv2.imwrite(f"{save_name}_roi{index}.jpg", cropped_roi)
                        except cv2.error as e:
                            print(e)
                annotator.box_label(b, model.names[int(c)])

        img = annotator.result()
        if args.bounding_box_pictures:
            # saves the result with bounding box
            cv2.imwrite(save_name + "_box.jpg", img)
