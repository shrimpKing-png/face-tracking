# -*- coding: utf-8 -*-
"""
utils/general.py
Created on Wed Jun 26 01:16:25 2025
Last Update: 25JUNE2025
@author: GPAULL
"""

import os
import tkinter as tk
from tkinter import filedialog
from face_tracking import normalize_frame
import cv2 as cv
import numpy as np


def filebrowser(select_directory=False):
    """
    Allows you to browse to a file or directory and return its absolute path as a string

    Parameters
    ----------
    select_directory : bool, optional
        If True, opens directory selection dialog. If False, opens file selection dialog.
        Default is False (file selection).

    Returns
    -------
    str
        File or directory relative path. Returns empty string if no selection made.
    """

    core = tk.Tk()
    core.withdraw()

    if select_directory:
        filepath = filedialog.askdirectory()
    else:
        filepath = filedialog.askopenfilename()

    core.destroy()

    if filepath:
        return os.path.relpath(filepath)
    else:
        return ""


def video_to_list(video_path):
    """
    This func will extract the frames from a video into a list object
    Args:
        video_path: a directory path with tif files or a .mp4 filepath.
    Returns:frames - a list of all the frames read from the video

    """
    frames = []
    if not os.path.isdir(video_path):
        print(
            "This has been detected as mp4 file.")
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
        else:
            print("Entering Video Processing loop, if this lasts too long press --> q <-- to exit")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame, Exiting...")
                    break
                frames.append(frame)
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
            cap.release()
            cv.destroyAllWindows()
            return frames
    else:
        tiff_files = [f for f in os.listdir(video_path) if f.endswith('.tif')]
        sorted_tiff_files = sorted(tiff_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
        for tif in sorted_tiff_files:
            frames.append(cv.imread(os.path.join(video_path, tif), cv.IMREAD_UNCHANGED))
        return frames


def list_to_video(frames_lst: list, vid_name: str, fps: int=30):
    """
    This function will take a list of frames and save them as video file.
    Args:
        frames_lst: The list of all frames in order to save into the video file
        vid_name: the label for the video file, just a string. No need to add .mp4. that is added in this func
        fps: self-explanatory, the fps of the video. If you don't know what fps is use google.com

    Returns:

    """
    # convert to bgr if frames are grayscale
    if frames_lst[0].ndim == 2:
        frames_lst = [normalize_frame(frame, np.ones_like(frame)) * 255 for frame in frames_lst]
        frames_lst = [frame.astype(np.uint8) for frame in frames_lst]
        frames_lst = [cv.cvtColor(frame, cv.COLOR_GRAY2BGR) for frame in frames_lst]
    # setup opencv videoWriter
    output_path = vid_name + '.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or try 'XVID' if mp4v doesn't work
    frame_width, frame_height = frames_lst[0].shape[1], frames_lst[0].shape[0]
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    for frame in frames_lst:
        out.write(frame)
    out.release()
    return


def input_to_bool(request_string):
    return input(request_string).strip().lower() == 'y'
