import os
import shutil
import cv2

from simple_color_detection_from_roi import detect_color, red, yellow, green, orange, yellow_orange, \
    crop_top_half, crop_sides_percentage
from utils import get_jpg_files, calculate_nonzero_percent

workdir = "/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed/"


color = yellow

bad_colors = {yellow, red, orange, yellow_orange, green}

bad_colors -= {color}

if os.path.exists(
        f"{workdir}{str(color.__name__)}"):
    shutil.rmtree(
        f"{workdir}{str(color.__name__)}")
os.mkdir(
    f"{workdir}{str(color.__name__)}")

counter = 0
files = get_jpg_files(
    f"{workdir}roi_unanotated")
for i in files:
    image = cv2.imread(i)
    # image = replace_white_with_black(image)
    result_color = crop_sides_percentage(crop_top_half(detect_color(image, color_filter=color)))
    bad_colors_result_perc = [calculate_nonzero_percent(detect_color(image, i)) for i in bad_colors]

    if calculate_nonzero_percent(result_color) > 0.2 \
            and 0.4 > max(bad_colors_result_perc):
        counter += 1
        print(counter)
        cv2.imwrite(
            f"{workdir}/{str(color.__name__)}/{counter}_mini_roi.jpg",
            result_color)
        cv2.imwrite(
            f"{workdir}/{str(color.__name__)}/{counter}.jpg",
            image)
