import json

import os

import argparse
import numpy as np
from pytube import YouTube
import subprocess


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


def split_train_val_test(
        path="/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed"
             "/czech_railway_dataset/train/images/metadata.txt"):
    with open(path) as f:
        metadata = f.readlines()

    metadata = np.array(metadata)
    np.random.shuffle(metadata)
    train = metadata[:int(metadata.shape[0] * 0.9)]
    test = metadata[int(metadata.shape[0] * 0.9):]

    assert (len(train) + len(test)) == len(metadata)

    # [os.replace(f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed"
    #          f"/czech_railway_dataset/train/images/traffic light/{i[:-1]}",
    #             f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed"
    #             f"/czech_railway_dataset/val/images/traffic light/{i[:-1]}"
    #             ) for i in test]

    [os.replace(f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed"
                f"/czech_railway_dataset/train/images/traffic light/{i[:-1]}",
                f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset/reconstructed"
                f"/czech_railway_dataset/val/images/traffic light/{i[:-1]}"
                ) for i in test]


#
# for i in os.listdir("/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset"
#                     "/reconstructed/czech_railway_dataset/val/images/traffic light"):
#     # Full paths for old and new names
#     old_path = os.path.join(f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset"
#                             f"/reconstructed"
#              f"/czech_railway_dataset/train/labels/traffic light/", i[:-4] + ".txt")
#     new_path = os.path.join(f"/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection1/dataset"
#                             f"/reconstructed"
#              f"/czech_railway_dataset/val/labels/traffic light/",  i[:-4] + ".txt")
#
#     # Rename the file
#     os.replace(old_path, new_path)


def str2bool(v):
    """
    The function `str2bool` converts a string representation of a boolean value to a boolean type in
    Python.
    
    :param v: The function `str2bool` takes a string `v` as input and converts it to a boolean value. If
    `v` is already a boolean, it returns `v` as is. If `v` is a string that represents a boolean value
    (e.g., 'yes', '
    :return: The `str2bool` function is designed to convert a string representation of a boolean value
    to an actual boolean value. If the input string `v` matches any of the recognized true values
    ('yes', 'true', 't', 'y', '1'), it will return `True`. If the input string matches any of the
    recognized false values ('no', 'false', 'f', '
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def download_video_pytube(link, SAVE_PATH):
    """
    The function `download_video_pytube` downloads a YouTube video with a specified resolution and saves
    it to a specified path.
    
    :param link: The `link` parameter in the `download_video_pytube` function should be the URL of the
    YouTube video that you want to download. This link will be used to create a `YouTube` object to
    access the video streams
    :param SAVE_PATH: The `SAVE_PATH` parameter in the `download_video_pytube` function is the directory
    path where you want to save the downloaded video file. You should provide the full path to the
    directory where you want the video to be saved on your local machine. For example, it could be
    something like
    :return: the video stream object `d_video` after attempting to download the video from the provided
    link to the specified `SAVE_PATH`.
    """
    try:
        # object creation using YouTube
        yt = YouTube(link)
    except:
        # to handle exception
        print(" YouTube Connection Error")

        # Get all streams and filter for mp4 files
    mp4_streams = yt.streams.filter(resolution="720p")

    # get the video with the highest resolution
    d_video = mp4_streams[0]

    if d_video in os.listdir(SAVE_PATH):
        return d_video

    try:
        print('downloading the video' + d_video.default_filename)

        d_video.download(output_path=SAVE_PATH)
        print('Video downloaded successfully!')
    except Exception:
        print("Some Error!")
    return d_video


def get_youtube_video_info(youtube_url):
    """
    The function `get_youtube_video_info` extracts the title of a YouTube video using the yt-dlp tool.
    
    :param youtube_url: The function `get_youtube_video_info` takes a YouTube video URL as input and
    uses the `yt-dlp` command-line tool to retrieve information about the video. The function then
    extracts and returns the title of the video from the obtained information
    :return: The function `get_youtube_video_info` is returning the title of the YouTube video specified
    by the `youtube_url` parameter.
    """
    command = [
        'yt-dlp',
        '-j',
        youtube_url
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    video_info = json.loads(result.stdout)
    return video_info['title']


def download_video(link, SAVE_PATH):
    """
    The function `download_video` downloads a video from a given link in either full HD or 720p quality
    and saves it to a specified path after processing the video name.
    
    :param link: The `link` parameter in the `download_video` function is the URL link to the video that
    you want to download. This link should point to a video on a platform like YouTube
    :param SAVE_PATH: SAVE_PATH is the directory path where the downloaded video will be saved. It
    should be a valid path on your system where you have write permissions
    :return: The function `download_video` returns the name of the downloaded video file after
    attempting to download it in full HD quality (1080p, 60fps) and falling back to 720p if the full HD
    quality download fails.
    """
    video_name = get_youtube_video_info(link)
    video_name = video_name.strip().replace("⧸", "").replace("/", "").replace("#", "").replace(",", "").replace(".", "")
    video_name += ".mp4"

    if video_name in os.listdir(SAVE_PATH):
        return video_name
    try:
        command = [
            'yt-dlp',
            link,
            '-f', 'bestvideo[height=1080][fps=60][ext=mp4]/best[height=1080][fps=60][ext=mp4]',
            '-o', f'{SAVE_PATH}/' + video_name,
        ]
        subprocess.run(command, check=True)
    except:
        print("full HD quality failed... trying 720p")
        command = [
            'yt-dlp',
            link,
            '-f', 'bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]',
            '-o', f'{SAVE_PATH}/' + video_name,
        ]
        subprocess.run(command, check=True)

    return video_name
