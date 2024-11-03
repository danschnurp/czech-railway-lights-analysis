import os
import shutil
import cv2

from simple_color_detection_from_roi import detect_color, red, yellow, green, orange, yellow_orange, \
    crop_top_half, crop_sides_percentage
from utils import get_jpg_files, calculate_nonzero_percent, check_content_centered, calculate_aspect_ratio

workdir = "/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed/roi_unanotated/"
output_dir = "/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed/"


def prepare_dirs(color):
    if os.path.exists(
            f"{output_dir}{str(color.__name__)}"):
        shutil.rmtree(
            f"{output_dir}{str(color.__name__)}")
    os.mkdir(
        f"{output_dir}{str(color.__name__)}")


def save_image(counter, output_dir, color, image, result_color, original=True, mini_roi=False):
    if mini_roi:
        cv2.imwrite(
            f"{output_dir}/{str(color.__name__)}/{counter}_mini_roi.jpg",
            result_color)
    if original:
        cv2.imwrite(
            f"{output_dir}/{str(color.__name__)}/{counter}.jpg",
            image)


def detect_red(color=red):
    bad_colors = {yellow, red, orange, yellow_orange, green}

    bad_colors -= {color}
    prepare_dirs(color)

    stats = {}
    counter = 0
    files = get_jpg_files(
        f"{workdir}")
    for i in files:
        image = cv2.imread(i)
        aspect_ratio, w, h = calculate_aspect_ratio(image)
        # image = replace_white_with_black(image)
        result_color = crop_sides_percentage(crop_top_half(detect_color(image, color_filter=color)))
        bad_colors_result_perc = [calculate_nonzero_percent(detect_color(image, i)) for i in bad_colors]

        is_centered = check_content_centered(result_color)
        if calculate_nonzero_percent(result_color) > 0.2 \
                and 0.4 > max(bad_colors_result_perc) and is_centered and aspect_ratio > 0.35:
            counter += 1
            stats[counter] = {"aspect ratio": aspect_ratio,"path": i[len(workdir):] }

            save_image(counter, output_dir, color, image, result_color)
    print(f"{str(color.__name__)} found:", counter, "from total", len(files), )
    print("stats:")
    [print(i, stats[i]) for i in stats]


def detect_green(color=green):
    bad_colors = {yellow, red, orange, yellow_orange, green}

    bad_colors -= {color}

    prepare_dirs(color)
    stats = {}
    counter = 0
    files = get_jpg_files(
        f"{workdir}")
    for i in files:
        image = cv2.imread(i)
        aspect_ratio, w, h = calculate_aspect_ratio(image)
        # image = replace_white_with_black(image)
        result_color = crop_sides_percentage(detect_color(image, color_filter=color))
        bad_colors_result_perc = [calculate_nonzero_percent(detect_color(image, i)) for i in bad_colors]

        if calculate_nonzero_percent(result_color) > 0.2 \
                and 0.4 > max(bad_colors_result_perc) and check_content_centered(result_color) and aspect_ratio > 0.35:
            counter += 1
            stats[counter] = {"aspect ratio": aspect_ratio,"path": i[len(workdir):]}

            save_image(counter, output_dir, color, image, result_color)
    print(f"{str(color.__name__)} found:", counter, "from total", len(files), )
    print("stats:")
    [print(i, stats[i]) for i in stats]



if __name__ == '__main__':
    detect_red()
    print("-----------------------------------------------")
    detect_green()
    print("-----------------------------------------------")
    # detect_yellow()
