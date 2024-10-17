import sys

import cv2
import numpy as np
import argparse

# args setting
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', "--input", help="input file video")
parser.add_argument('--skip_seconds', type=int, default=0)
args = parser.parse_args()


def main():
    # load video class
    cap = VideoCapture(args.input)
    cap.video.set(cv2.CAP_PROP_POS_MSEC,
                  args.skip_seconds * 1000
                  )
    r, frame = cap.read()
    # initialization for line detection
    offset = 400
    expt_startLeft = int(frame.shape[1] / 2 - offset * 1.1)
    expt_startRight = expt_startLeft + offset
    expt_startTop = frame.shape[0] - offset

    # value initialize
    left_maxpoint = [0] * 50
    right_maxpoint = [195] * 50

    # convolution filter
    kernel = np.array([
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1],
        [-1, 1, 0, 1, -1]
    ])

    # Next frame availability
    r = True
    first = True
    direction_left = False
    offset_step = 60
    line_mess = 200

    while r is True:
        r, frame = cap.read()
        if frame is None:
            break

        # cut away invalid frame area
        valid_frame = frame[expt_startTop:, expt_startLeft:expt_startRight]

        # gray scale transform
        gray_frame = cv2.cvtColor(valid_frame, cv2.COLOR_BGR2GRAY)

        # histogram equalization image
        histeqaul_frame = cv2.equalizeHist(gray_frame)

        # apply gaussian blur
        blur_frame = cv2.GaussianBlur(histeqaul_frame, (5, 5), 5)

        # merge current frame and last frame
        if first is True:
            merge_frame = blur_frame
            first = False
            old_valid_frame = merge_frame.copy()
        else:
            merge_frame = cv2.addWeighted(blur_frame, 0.2, old_valid_frame, 0.8, 0)
            old_valid_frame = merge_frame.copy()

        # convolution filter
        conv_frame = cv2.filter2D(merge_frame, -1, kernel)

        # initialization for sliding window property
        sliding_window = [20, 190, 200, 370]
        slide_interval = 15
        slide_height = 15
        slide_width = 60

        # initialization for bezier curve variables
        left_points = []
        right_points = []

        # define count value
        count = 0
        for i in range(340, 40, -slide_interval):
            # get edges in sliding window
            left_edge = conv_frame[i:i + slide_height, sliding_window[0]:sliding_window[1]].sum(axis=0)
            right_edge = conv_frame[i:i + slide_height, sliding_window[2]:sliding_window[3]].sum(axis=0)

            # left railroad line processing
            if left_edge.argmax() > 0:
                left_maxindex = sliding_window[0] + left_edge.argmax()
                left_maxpoint[count] = left_maxindex
                cv2.line(frame, (left_maxindex + expt_startLeft, i + int(slide_height / 2) + expt_startTop),
                         (left_maxindex + expt_startLeft, i + int(slide_height / 2) + expt_startTop), (255, 255, 255),
                         5, cv2.LINE_AA)
                left_points.append([left_maxindex, i + int(slide_height / 2)])
                sliding_window[0] = max(0, left_maxindex - int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                sliding_window[1] = min(390, left_maxindex + int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                cv2.rectangle(frame, (sliding_window[0] + expt_startLeft, i + slide_height + expt_startTop),
                              (sliding_window[1] + expt_startLeft, i + expt_startTop), (0, 255, 0),
                              1)

            # right railroad line processing
            if right_edge.argmax() > 0:
                right_maxindex = sliding_window[2] + right_edge.argmax()
                right_maxpoint[count] = right_maxindex
                cv2.line(frame, (right_maxindex + expt_startLeft, i + int(slide_height / 2) + expt_startTop),
                         (right_maxindex + expt_startLeft, i + int(slide_height / 2) + expt_startTop), (255, 255, 255),
                         5, cv2.LINE_AA)
                right_points.append([right_maxindex, i + int(slide_height / 2)])
                sliding_window[2] = max(0, right_maxindex - int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                sliding_window[3] = min(390, right_maxindex + int(slide_width / 4 + (slide_width + 10) / (count + 1)))
                cv2.rectangle(frame, (sliding_window[2] + expt_startLeft, i + slide_height + expt_startTop),
                              (sliding_window[3] + expt_startLeft, i + expt_startTop), (0, 0, 255),
                              1)
            count += 1
        try:
            # ########
            # following code sets parameters left and right automatically
            # difference between max and min X coord ... this works well for straight lines
            if max(np.array(left_points)[:, 0]) - min(np.array(left_points)[:, 0]) > line_mess or \
                    max(np.array(right_points)[:, 0]) - min(np.array(right_points)[:, 0]) > line_mess:
                # changes direction to left in 2/3 of frame
                if frame.shape[1] * 2 / 3.5 <= expt_startRight:
                    direction_left = True
                # changes direction to right in 1/3 of frame
                if frame.shape[1] / 3.5 >= expt_startRight:
                    direction_left = False
                # moves in direction
                if direction_left:
                    expt_startLeft -= offset_step
                    expt_startRight -= offset_step
                else:
                    expt_startLeft += offset_step
                    expt_startRight += offset_step
        except IndexError:
            print(f"black screen in {cap.video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0}", file=sys.stderr,)
            continue
        cv2.imshow('Video', frame)
        cv2.waitKey(1)
    print('finish')


# class for reading video
class VideoCapture:
    def __init__(self, path):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(path)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def read(self):
        # Grab a single frame of video
        ret, frame = self.video.read()
        return frame is not None, frame


if __name__ == '__main__':
    main()
