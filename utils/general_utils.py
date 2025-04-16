import codecs
import json
import locale

import os

import argparse
import sys
from collections import defaultdict

import unicodedata

from pytube import YouTube
import subprocess

from tqdm import tqdm

from utils.utf8_encoder import UTF8StringEncoder

import cv2
import numpy as np
import matplotlib.pyplot as plt


def normalize_text(text):
    return unicodedata.normalize('NFC', text.replace("⧸", "")
                          .replace("/", "")
                          .replace("#", "")
                          .replace(",", "")
                          .replace(".", ""))

def normalize_list_of_texts(texts):
    return [normalize_text(text) for text in texts]


def analyze_video_characteristics(video_path, sample_interval=24):
    """
    Comprehensive analysis of video content characteristics with emphasis on
    brightness distribution and temporal variation.

    Parameters:
    -----------
    video_path : str
        Path to the video file for analysis
    sample_interval : int
        Frame sampling interval for histogram computation

    Returns:
    --------
    dict
        Metadata dictionary containing technical parameters and statistical measures
    """
    metadata = {}
    cap = cv2.VideoCapture(video_path)

    # Extract fundamental technical parameters
    metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
    metadata['resolution'] =str( (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    metadata['duration'] = str(metadata['frame_count'] / metadata['fps'])
    metadata['fps'] = str(cap.get(cv2.CAP_PROP_FPS))
    # Initialize brightness histogram aggregation
    brightness_histograms = []
    brightness_means = []
    brightness_stds = []

    # Temporal sampling for brightness analysis
    frame_indices = range(0, metadata['frame_count'], sample_interval)

    for frame_idx in tqdm(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert to grayscale for luminance analysis
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute frame brightness histogram
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize to probability distribution
        brightness_histograms.append(hist)

        # Calculate statistical measures
        mean_brightness = np.mean(gray_frame)
        std_brightness = np.std(gray_frame)
        brightness_means.append(mean_brightness)
        brightness_stds.append(std_brightness)

    # Aggregate brightness statistical measures
    metadata['brightness_mean'] = str(np.mean(brightness_means))
    metadata['brightness_std'] = str(np.mean(brightness_stds))
    metadata['brightness_temporal_variation'] = str(np.std(brightness_means))

    # Compute average histogram across sampled frames
    if brightness_histograms:
        avg_histogram = np.mean(brightness_histograms, axis=0)
        metadata['brightness_histogram'] = str(avg_histogram)

        # Derive histogram statistical measures
        metadata['histogram_entropy'] = str(-np.sum(avg_histogram * np.log2(avg_histogram + 1e-10)))
        metadata['histogram_dynamic_range'] = str(np.sum(avg_histogram > 0.0001) / 256.0)

        # Analyze histogram modality for lighting characteristics
        peaks = find_histogram_peaks(avg_histogram)
        metadata['histogram_modality'] =str( len(peaks))
        metadata['histogram_peaks'] = str(peaks)

    cap.release()
    return metadata

def find_histogram_peaks(histogram, min_distance=10, threshold=0.01):
    """
    Identify significant peaks in brightness histogram to characterize
    lighting conditions and potential bimodality.

    Parameters:
    -----------
    histogram : ndarray
        Normalized brightness histogram
    min_distance : int
        Minimum distance between peaks
    threshold : float
        Minimum amplitude threshold for peak detection

    Returns:
    --------
    list
        Detected peak positions
    """
    # Smooth histogram with Gaussian filter to reduce noise
    smoothed = cv2.GaussianBlur(histogram, (5, 0), 2)

    # Find local maxima
    peaks = []
    for i in range(1, len(smoothed)-1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and smoothed[i] > threshold:
            peaks.append((i, smoothed[i]))

    # Filter peaks by minimum distance
    filtered_peaks = []
    peaks.sort(key=lambda x: x[1], reverse=True)

    for peak in peaks:
        if not filtered_peaks or min(abs(peak[0] - p[0]) for p in filtered_peaks) >= min_distance:
            filtered_peaks.append(peak)

    return [p[0] for p in filtered_peaks]

def visualize_brightness_distribution(metadata, output_path=None):
    """
    Generate visualization of brightness distribution characteristics.

    Parameters:
    -----------
    metadata : dict
        Video metadata containing brightness histogram
    output_path : str, optional
        Path to save visualization output
    """
    if 'brightness_histogram' not in metadata:
        return

    plt.figure(figsize=(12, 6))

    # Plot brightness histogram
    plt.subplot(2, 1, 1)
    plt.plot(metadata['brightness_histogram'], 'b-')

    # Mark detected peaks
    if 'histogram_peaks' in metadata:
        for peak in metadata['histogram_peaks']:
            plt.axvline(x=peak, color='r', linestyle='--', alpha=0.5)

    plt.title(f"Brightness Histogram Analysis: {metadata['name']}\n"
              f"Mean: {metadata['brightness_mean']:.2f}, " +
              f"Entropy: {metadata['histogram_entropy']:.2f}, " +
              f"FPS: {metadata['fps']:.2f}, " +
              f"resolution: {metadata['resolution'][0]} x {metadata['resolution'][0]}, " +
              f"duration: {metadata['duration']:.2f} sec, " +

              f"Modality: {metadata['histogram_modality']}")
    plt.xlabel('Brightness Level')


    # Plot temporal brightness variation
    if 'brightness_means' in metadata:
        plt.subplot(2, 1, 2)
        plt.plot(metadata['brightness_means'], 'g-')
        plt.title(f"Temporal Brightness Variation\nStd: {metadata['brightness_temporal_variation']:.2f}")
        plt.xlabel('Frame Index')


    plt.tight_layout()
    plt.ylim(0, 0.02)
    plt.yticks([])
    if output_path:
        plt.savefig(output_path+ ".pdf", bbox_inches='tight')

    plt.close()

def remove_annotated_duplicates(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]
    unique_entries = {}
    duplicates = defaultdict(list)

    # Iterate through the JSON array
    for entry in data:

        key = (normalize_text(entry["video name"]), entry["timestamp in video"])
        if key in unique_entries:
            duplicates[key].append(entry)
        else:
            unique_entries[key] = entry

    # Print duplicates
    print("Duplicates:")
    for key, duplicate_list in duplicates.items():
        for duplicate in duplicate_list:
            print(json.dumps(duplicate, indent=4))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({duplicates["data"]}, f)


def get_system_encoding():
    """
    Get the system's preferred encoding.
    Returns 'utf-8' for non-Windows systems.
    """
    if sys.platform.startswith('win'):
        return locale.getpreferredencoding()
    return 'utf-8'


def print_statistics(json_path="../experiments/today_results.json"):
    with open(json_path, encoding="utf-8", mode="r") as f:
        traffic_lights = dict(json.load(f))
    print("current size of traffic lights dataset is: \t", sum([len(traffic_lights[i]) for i in traffic_lights]), "pictures")
    return sum([len(traffic_lights[i]) for i in traffic_lights])



def save_json_unicode(data, filename, format_unicode=True):
    """
    Save data to JSON file with options for Unicode handling.
    Handles Windows encoding issues.

    Args:
        data: Data to save
        filename: Output filename
        format_unicode: If True, saves Unicode characters as readable characters
                      If False, saves as escaped Unicode
    """
    try:
        # First attempt: Try with utf-8
        if format_unicode:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=True, indent=4)

    except UnicodeEncodeError:
        # Second attempt: Use system encoding (for Windows)
        try:
            if format_unicode:
                with codecs.open(filename, 'w', encoding='utf-8-sig') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            else:
                with codecs.open(filename, 'w', encoding='utf-8-sig') as f:
                    json.dump(data, f, ensure_ascii=True, indent=4)
        except UnicodeEncodeError as e:
            # If all else fails, force ASCII encoding
            print(f"Warning: Falling back to ASCII encoding due to: {str(e)}")
            with open(filename, 'w', encoding='ascii') as f:
                json.dump(data, f, ensure_ascii=True, indent=4)


def get_video_name(video_link, names_jsom_path):
    with open(names_jsom_path, encoding="utf-8", mode="r") as f:
        video_names = dict(json.load(f))
    video_names = video_names["names"]
    return dict(zip(list(video_names.values()), list(video_names.keys())))[video_link]



def get_times_by_video_name(sequences_jsom_path, names_jsom_path="../railway_datasets/video_names.json", reverse=False):
    with open(sequences_jsom_path, encoding="utf-8", mode="r") as f:
        traffic_lights = dict(json.load(f))
    with open(names_jsom_path, encoding="utf-8", mode="r") as f:
        video_names = dict(json.load(f))

    video_names = video_names["names"]
    encoder = UTF8StringEncoder()
    if reverse:
        normalized_video_names = {
            v:
           encoder.decode_string(encoder.encode_string( unicodedata.normalize('NFC', k.replace("⧸", "")
                                  .replace("/", "")
                                  .replace("#", "")
                                  .replace(",", "")
                                  .replace(".", ""))))[0]
            for k, v in video_names.items()
        }
    else:
        normalized_video_names = {
            encoder.decode_string(encoder.encode_string(unicodedata.normalize('NFC', k.replace("⧸", "")
                                                                              .replace("/", "")
                                                                              .replace("#", "")
                                                                              .replace(",", "")
                                                                              .replace(".", ""))))[0]: v
            for k, v in video_names.items()
        }

    items_to_modify = []
    for ytlink, times in traffic_lights.items():
        if ytlink in normalized_video_names.values():
            video_name = list(normalized_video_names.keys())[list(normalized_video_names.values()).index(ytlink)]
            items_to_modify.append((ytlink, video_name))

    for ytlink, video_name in items_to_modify:
        traffic_lights[video_name] = traffic_lights.pop(ytlink)

    return traffic_lights


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


def download_video(link, SAVE_PATH, use_internet=True, names_jsom_path="../railway_datasets/video_names.json"):
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
    if not  use_internet:
        video_name = get_video_name(link, names_jsom_path=names_jsom_path)
    else:
        video_name = get_youtube_video_info(link)
    video_name = (video_name.strip().replace("⧸", "")
                  .replace("/", "").replace("#", "")
                  .replace(",", "").replace(".", "").replace("\"", ""))
    video_name += ".mp4"

    if video_name in os.listdir(SAVE_PATH):
        return video_name
    elif  not  use_internet:
        return None
    try:
        command = [
            'yt-dlp',
            link,
            '-f', 'bestvideo[height=1080][fps=60][ext=mp4]/best[height=1080][fps=60][ext=mp4]',
            '-o', f'{SAVE_PATH}/' + video_name,
        ]
        subprocess.run(command, check=True)
    except Exception as e:
        print(f"{e}\n full HD quality withs 60fps failed... trying 50fps", file=sys.stderr)
        command = [
            'yt-dlp',
            link,
            '-f', 'bestvideo[height=1080][fps=50][ext=mp4]/best[height=1080][fps=50][ext=mp4]',
            '-o', f'{SAVE_PATH}/' + video_name,
        ]
        subprocess.run(command, check=True)

    return video_name

