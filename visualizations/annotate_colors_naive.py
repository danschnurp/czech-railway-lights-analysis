import json
import os
import shutil
import cv2

from utils.image_utils import detect_color, red, yellow, green, orange, yellow_orange, \
    crop_top_half, crop_sides_percentage, calculate_nonzero_percent, check_content_centered, calculate_aspect_ratio
from utils.general_utils import get_jpg_files

workdir = "/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed/roi_unanotated/"
output_dir = f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed/"


def log_metadata(path_attributes, aspect_ratio, detected_color):
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
        "aspect ratio": float(f"{float(aspect_ratio):.3f}"),
        "video name": path_attributes[-4],
        "detection method": path_attributes[-3],
        "class": path_attributes[-2],
        "timestamp in video": float(f"{float(path_attributes[-1][:path_attributes[-1].find('_')]):.3f}"),
        "color": detected_color
    }


def prepare_dirs(color):
    """
    The function `prepare_dirs` checks if a directory exists, removes it if it does, and then creates a
    new directory with the same name.
    
    :param color: It looks like the `color` parameter is a class or a function since `color.__name__` is
    being used to get its name as a string. The `prepare_dirs` function seems to be preparing
    directories based on the name of the color
    """
    if os.path.exists(
            f"{output_dir}{str(color.__name__)}"):
        shutil.rmtree(
            f"{output_dir}{str(color.__name__)}")
    os.mkdir(
        f"{output_dir}{str(color.__name__)}")


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
    if mini_roi:
        cv2.imwrite(
            f"{output_dir}/{str(color.__name__)}/{counter}_mini_roi.jpg",
            result_color)
    if original:
        cv2.imwrite(
            f"{output_dir}/{str(color.__name__)}/{counter}.jpg",
            image)


def detect_red(color=red):
    """
    function detects and processes images with a specified color, providing statistics and
    saving the results.
    
    :param color: It looks like the code snippet you provided is a function named `detect_red` that
    seems to be part of an image processing script. The function is designed to detect images with a
    specific color (red in this case) and perform various operations on them
    :return: The function `detect_red` returns a dictionary containing statistics for images that meet
    certain criteria for the color red. The key of the dictionary is the string representation of the
    color name ("red" in this case), and the value is a dictionary of image statistics.
    """
    bad_colors = {yellow, red, orange, yellow_orange, green}

    bad_colors -= {color}
    prepare_dirs(color)

    stats = []
    counter = 0
    files = get_jpg_files(
        f"{workdir}")
    for i in files:
        i = i.replace("\\", "/")
        image = cv2.imread(i)
        aspect_ratio, w, h = calculate_aspect_ratio(image)
        # image = replace_white_with_black(image)
        result_color = crop_sides_percentage(crop_top_half(detect_color(image, color_filter=color)))
        bad_colors_result_perc = [calculate_nonzero_percent(detect_color(image, i)) for i in bad_colors]

        is_centered = check_content_centered(result_color)
        if calculate_nonzero_percent(result_color) > 0.2 \
                and 0.4 > max(bad_colors_result_perc) and is_centered:
            counter += 1
            stats.append(log_metadata(path_attributes=i[len(workdir):].split("/"), aspect_ratio=aspect_ratio, detected_color=str(color.__name__)))

            save_image(counter, output_dir, color, image, result_color)
    print(f"{str(color.__name__)} found:", counter, "from total", len(files), )
    print("stats:")
    [print(i, stats) for i in stats]
    return stats


def detect_green(color=green):
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

    bad_colors -= {color}

    prepare_dirs(color)
    stats = []
    counter = 0
    files = get_jpg_files(
        f"{workdir}")
    for i in files:
        i = i.replace("\\", "/")
        image = cv2.imread(i)
        aspect_ratio, w, h = calculate_aspect_ratio(image)
        # image = replace_white_with_black(image)
        result_color = crop_sides_percentage(detect_color(image, color_filter=color))
        bad_colors_result_perc = [calculate_nonzero_percent(detect_color(image, i)) for i in bad_colors]

        if calculate_nonzero_percent(result_color) > 0.2 \
                and 0.4 > max(bad_colors_result_perc) and check_content_centered(result_color):
            counter += 1
            path_attributes = i[len(workdir):].split("/")
            stats.append(log_metadata(path_attributes, aspect_ratio, str(color.__name__)))

            save_image(counter, output_dir, color, image, result_color)
    print(f"{str(color.__name__)} found:", counter, "from total", len(files), )
    print("stats:")
    [print(i, stats) for i in stats]
    return stats


def save_to_vis(vis_type= "_aspect_ratio_based_on_videos"):
    with open(f"visualization{vis_type}.json") as f:
        vis = json.load(f)
        vis["data"]["values"] = [*r, *g, *y]
    with open(f"{output_dir}/vis{vis_type}.json", mode="w", encoding="utf-8") as f:
        json.dump(vis, f, indent=2)


if __name__ == '__main__':
    r = detect_red()
    print("-----------------------------------------------")
    g = detect_green()
    print("-----------------------------------------------")
    y = detect_green(color=yellow)
    save_to_vis()
    save_to_vis(vis_type="_aspect_ratio_based_on_colors")

    with open(f"{output_dir}/metadata.json", mode="w", encoding="utf-8") as f:
        json.dump([*r, *g, *y], f, indent=2)

