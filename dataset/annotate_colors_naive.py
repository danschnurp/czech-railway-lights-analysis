import json
import os
import shutil
import sys
import time

import cv2
import argparse

import numpy as np
import yaml
from tqdm import tqdm
from ultralytics import YOLO

from detect_white_triangles import detect_white_triangles
from utils.image_utils import detect_color, red, yellow, green, orange, yellow_orange, \
    crop_sides_percentage, calculate_nonzero_percent, check_content_centered, calculate_aspect_ratio, \
    crop_top_bottom_percentage, black, visualize_dos_picturos, white, crop_bounding_box
from utils.general_utils import get_jpg_files


def log_metadata(path_attributes, aspect_ratio, detected_color, counter, roi):
    """
    The function `log_metadata` creates a dictionary with metadata information such as aspect ratio,
    video name, detection method, class, and timestamp in video based on input parameters.
    
    :param path_attributes: The `path_attributes` parameter is expected to be a list containing
    information about a video file. The elements in the list are as follows:
    :param aspect_ratio: Aspect ratio is the ratio of the width to the height of an image or video. It
    is commonly expressed as two numbers separated by a colon, such as 16:9 or 4:3
    :return: A dictionary containing metadata information about a video file. The dictionary includes
    the aspect ratio, video name, detection method, class, and timestamp in the video.
    """
    return {
        "ID": counter,
        "aspect ratio": float(f"{float(aspect_ratio):.3f}"),
        "video name": path_attributes[-4],
        "detection method": path_attributes[-3],
        "class": path_attributes[-2],
        "timestamp in video": float(f"{float(path_attributes[-1][:path_attributes[-1].find('_')]):.3f}"),
        "color": detected_color,
        "roi coordinates": roi
    }


def prepare_dirs(color):
    """
    The function `prepare_dirs` checks if a directory exists, removes it if it does, and then creates a
    new directory with the same name.
    
    :param color: It looks like the `color` parameter is a class or a function since `color.__name__` is
    being used to get its name as a string. The `prepare_dirs` function seems to be preparing
    directories based on the name of the color
    """
    if not os.path.exists(
            f"{output_dir}/{str(color)}"):
        os.mkdir(
            f"{output_dir}/{str(color)}")


def save_image(counter, output_dir, color, image, result_color, original=True, mini_roi=False):
    """
    The function `save_image` saves images with specified parameters to a specified output directory.
    
    :param counter: The `counter` parameter is used to specify a unique identifier or index for the
    image being saved. It is typically an integer value that helps in naming the saved image files
    uniquely
    :param output_dir: The `output_dir` parameter is a string that represents the directory where the
    images will be saved
    :param color: The `color` parameter in the `save_image` function is likely a color space conversion
    function or a color representation, such as BGR, RGB, HSV, etc. It is used to specify the color
    space or representation of the image being saved
    :param image: The `image` parameter in the `save_image` function is typically a NumPy array
    representing an image that you want to save to a file. It could be an image in various color spaces
    such as RGB, BGR, grayscale, etc. The function uses this parameter to save the image to
    :param result_color: The `result_color` parameter in the `save_image` function is the color image
    that you want to save. It is used to save the color image in the specified output directory
    :param original: The `original` parameter in the `save_image` function is a boolean flag that
    determines whether to save the original image or not. If `original` is set to `True`, the original
    image will be saved using the specified file path. If `original` is set to `False`, the, defaults to
    True (optional)
    :param mini_roi: The `mini_roi` parameter is a boolean flag that indicates whether to save a smaller
    region of interest (ROI) image. If `mini_roi` is set to `True`, the function will save the
    `result_color` image with the filename `{output_dir}/{color_name}/{counter}_mini_roi, defaults to
    False (optional)
    """
    if original:
        cv2.imwrite(
            f"{output_dir}/{str(color)}/{counter}.jpg",
            image)


def get_box_coordinates(image, model):
    result = model.predict(image)
    result = result[0]
    # Iterate over the results
    boxes = result.boxes  # Boxes object for bbox outputs
    class_indices = boxes.cls  # Class indices of the detections
    class_names = [result.names[int(i)] for i in class_indices]  # Map indices to names
    results = []
    cropped_rois = []
    if "traffic light" in class_names:
        for box in boxes:
            if result.names[int(box.cls)] == "traffic light":
                cropped_rois.append(crop_bounding_box(box.xyxy[0], image))
                results.append(f"{box.xywhn.tolist()[0][0]} {box.xywhn.tolist()[0][1]} {box.xywhn.tolist()[0][2]} {box.xywhn.tolist()[0][3]}")
    return results, cropped_rois


def detect_single_color(colors={yellow, red, orange, yellow_orange, green, black}, class_names = ["stop"],
                        crop_sides_value_percentage=0,
                        crop_top_bottom_value_percentage=0,
                        files=None):
    """
    This Python function detects images with a specified color and extracts metadata and statistics for
    the detected images.
    
    :param color: It seems like the code snippet you provided is a function named `detect_green` that
    detects green color in images. The function takes a color parameter, which is set to `green` by
    default. The function then processes a list of image files, reads each image, calculates aspect
    ratio, detects color
    :return: The function `detect_green` is returning a dictionary containing statistics for images that
    meet certain criteria for the color green. The key of the dictionary is the string representation of
    the color name "green", and the value is a dictionary of statistics for each image that meets the
    criteria.
    """
    bad_colors = {yellow, red, orange, yellow_orange, green}
    for i in colors:
        bad_colors -= {i}
    [prepare_dirs(class_name) for class_name in class_names]
    processed = []
    for i in tqdm(files):
        if i.split("_")[-1] == "clean.jpg":
            i = i.replace("\\", "/")
            try:
                image_clean = cv2.imread(i)
                image_box = cv2.imread(f"{i[:i.find('clean')]}box.jpg")
            except FileNotFoundError as ex:
                print("neco se posralo ", ex)
                continue
            aspect_ratio, w, h = calculate_aspect_ratio(image_clean)

            rois, this_roi_imgs = get_box_coordinates(image_clean, model)

            for (one_roi_index, this_roi), this_roi_img in zip(enumerate(rois), this_roi_imgs):
                # image = replace_white_with_black(image)
                result_colors = [
                    crop_top_bottom_percentage(crop_sides_percentage(detect_color(this_roi_img, color_filter=colo),
                                                                     crop_percentage=crop_sides_value_percentage),
                                               crop_percentage=crop_top_bottom_value_percentage) for colo in colors]
                # bad_colors_result_perc = [calculate_nonzero_percent(detect_color(image_roi, i)) for i in bad_colors]
                result_color_perc = [calculate_nonzero_percent(result_color) for result_color in result_colors]
                # centered = [check_content_centered(result_color) for result_color in result_colors]
                # white_trinagles = [detect_white_triangles(image_roi, image_clean)]
                verdict_colors = [c > 0.15 for c in result_color_perc]
                if any(verdict_colors):
                    h, w, _ = tuple([*this_roi_img.shape])
                    cv2.imshow(str(class_names), cv2.resize(image_box, (int(1960 / 1.9), int(1080 / 1.9))))
                    cv2.imshow(str(one_roi_index), cv2.resize(this_roi_img, (int(w * 5), int(h * 5))))
                    res = cv2.waitKey(0)
                    for index, class_name in enumerate(class_names):
                        if res == ord("x"):
                            cv2.destroyAllWindows()
                            return stats, False,  [image_path for image_path in  files if image_path not in processed]
                        try:
                            cls_id = int(chr(res))
                        except ValueError:
                            processed.append(i)
                            processed.append(f"{i[:i.find('clean')]}box.jpg")
                            cv2.destroyAllWindows()
                            continue
                        if cls_id == int(str(index)):
                            path_attributes = i[len(workdir):].split("/")
                            img_id = time.time_ns()
                            save_image(img_id, output_dir, class_name, this_roi_img, 1, )
                            stats.append(log_metadata(path_attributes, aspect_ratio, class_name, counter=img_id,
                                                      roi=this_roi))
                            processed.append(i)
                            processed.append(f"{i[:i.find('clean')]}box.jpg")
                            break
                    cv2.destroyAllWindows()
            processed.append(i)
            processed.append(f"{i[:i.find('clean')]}box.jpg")


    print(f"{class_names} found:", counter, "from total", len(files), )
    print("stats:")
    [print(i) for i in stats]
    return stats, True, list(set(files) - set(processed))


def save_to_vis(vis_type= "_aspect_ratio_based_on_videos", r=None, g=None, y=None):
    with open(f"./visualizations/visualization{vis_type}.json") as f:
        vis = json.load(f)
        vis["data"]["values"] = [*r, *g, *y]
    with open(f"{output_dir}/vis{vis_type}.json", mode="w", encoding="utf-8") as f:
        json.dump(vis, f, indent=2, ensure_ascii=False)


dataset_yaml = '../metacentrum/CRL.yaml'
with open(dataset_yaml, encoding="utf-8") as f:
    class_mapping = yaml.load(f, Loader=yaml.SafeLoader)

class_mapping = class_mapping["names"]
class_mapping = list(class_mapping.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="../reconstructed/all",
                        type=str, help="Path to the directory with images to process")
    parser.add_argument("--output_dir", default="./reconstructed",
                        type=str, help="Path to the output directory")
    args = parser.parse_args()
    workdir = args.workdir
    output_dir = args.output_dir
    model = YOLO("../yolov10n.pt")
    stats = []
    counter = 0
    files = get_jpg_files(
            f"{workdir}")
    if os.path.exists(f"{output_dir}/metadata_part.json"):
        with open(f"{output_dir}/metadata_part.json", mode="r", encoding="utf-8") as f:
                d = dict(json.load(f))
                files = d["files todo"]
                stats = d["data"]


        # r = detect_single_color(color=red,  crop_sides_value_percentage=15, crop_top_bottom_value_percentage=20)
    print("-----------------------------------------------")
    data, finished, files = detect_single_color(class_names=class_mapping,
                                                  files=files
                                                  )
    print("-----------------------------------------------")

    if finished:
        with open(f"{output_dir}/metadata.json", mode="w", encoding="utf-8") as f:
            json.dump({"data":[ *data]}, f, indent=2, ensure_ascii=False)
        os.remove(f"{output_dir}/metadata_part.json")
    else:
        with open(f"{output_dir}/metadata_part.json", mode="w", encoding="utf-8") as f:
            json.dump({"data": [*data], "files todo": files }, f, indent=2, ensure_ascii=False)

    # save_to_vis()
    # save_to_vis(vis_type="_aspect_ratio_based_on_colors", r=r, g=g, y=y)





