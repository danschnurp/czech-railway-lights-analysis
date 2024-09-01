import time

import cv2
import numpy as np

def detect_movement(video_path, refresh_time):
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_MSEC,
                  220 * 1000
                  )

    # Read the first frame as the reference frame
    ret, frame = cap.read()
    reference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t1 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the current frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the difference between the current frame and the reference frame
        diff = cv2.absdiff(gray_frame, reference_frame)

        # Threshold the difference image to create a binary mask
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if time.time() - t1 > refresh_time:
            reference_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t1 = time.time()

        # Draw contours on the original frame
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Adjust the area threshold as needed
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('Difference', diff)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Replace 'your_video.mp4' with the path to your video file
video_path = 'reconstructed/4K  Broumov - Meziměstí  854 Hydra + 954 Bfbrdtn794.mp4'
detect_movement(video_path, refresh_time=5)
