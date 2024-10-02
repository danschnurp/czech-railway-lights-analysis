import json

import os

import argparse
from pytube import YouTube
import subprocess

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def crop_bounding_box(box, img):
    x, y, w, h = box
    x, y, w, h = int(x), int(y), int(w), int(h)
    roi = img[y:h, x:w]
    return roi


def download_video_pytube(link, SAVE_PATH):
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
    command = [
        'yt-dlp',
        '-j',
        youtube_url
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    video_info = json.loads(result.stdout)
    return video_info['title']


def download_video(link, SAVE_PATH):
    video_name = get_youtube_video_info(link)
    video_name = video_name.strip().replace("⧸", "").replace("/", "").replace("#", "").replace(",", "").replace(".", "")
    video_name += ".mp4"

    if video_name in os.listdir(SAVE_PATH):
        return video_name

    command = [
        'yt-dlp',
        link,
        '-f', 'bestvideo[height=1080][fps=60][ext=mp4]/best[height=1080][fps=60][ext=mp4]',
        '-o', f'{SAVE_PATH}/' + video_name,
    ]
    subprocess.run(command, check=True)
    return video_name
