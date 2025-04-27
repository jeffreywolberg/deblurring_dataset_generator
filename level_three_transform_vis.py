from abc import ABC, abstractmethod
import copy
from matplotlib.backend_bases import KeyEvent
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import glob
import cv2
from os.path import join, basename, abspath, exists, splitext, isdir

from scipy.interpolate import RegularGridInterpolator

import numpy as np

VALID_IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".pgm", ".ppm", ".npy"]

CROP_BORDER = 30

class LevelThreeTransformVis:
    def __init__(self):
        self._orig_interp_state = {
            "st_frame": [0, -1, 1, 0],
            "n_avg": [1, 21, 1, 1],
        }
        
        self.interp_state = copy.deepcopy(self._orig_interp_state)

        self.video_no = 0
        self.video_directory = "./data/level3/videos"
        self.video_paths = sorted([join(self.video_directory, dirname) for dirname in os.listdir(self.video_directory)])
        self.video_frame_paths = [sorted([join(vid_path, fname) for fname in os.listdir(vid_path)]) for vid_path in self.video_paths]
        self.interp_state["st_frame"][1] = len(self.video_frame_paths[self.video_no]) - 1
        self.video_frames = [np.array([]) for vid in self.video_frame_paths]

        # Setup matplotlib figure and axes
        self.fig = plt.figure(figsize=(12, 8))

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.setup_plot()

    def on_slider_changed(self, param, val):
        self.interp_state[param][3] = val

        if param == "st_frame":
            self.interp_state["n_avg"][1] = min(len(self.video_frame_paths[self.video_no]) - val, max(self._orig_interp_state["n_avg"][1], self.interp_state["n_avg"][1]))
            self.interp_state["n_avg"][3] = min(self.interp_state["n_avg"][1], self.interp_state["n_avg"][3])
        # elif param == "n_avg":
            # self.interp_state["st_frame"][1]

        self.setup_plot()
        self.update_plot()

    def setup_plot(self):
        print(self.interp_state)
        self.fig.clear()
        # Create main image display on the right
        self.img_ax = plt.subplot2grid((1, 5), (0, 1), colspan=4)
        self.img_ax.cla()
        self.img_ax.set_title("Transformed Image")
        self.img_ax.axis('off')  # Remove axes and numbers
        
        self.slider_ax = plt.subplot2grid((1, 5), (0, 0))
        self.slider_ax.clear()  # Clear everything in the axes
        self.slider_ax.set_title(f"Motion")
        self.slider_ax.axis('off')  # Remove axes and numbers

        self.sliders = {}
        y_pos = 0.90
        for param_name, values in self.interp_state.items():
            ax = plt.axes([0.05, y_pos, 0.15, 0.03])
            # plt.cla()
            slider = Slider(
                ax=ax,
                label=f"{param_name}",
                valmin=values[0],
                valmax=values[1],
                valstep=values[2],
                valinit=values[3],
            )
            slider.on_changed(lambda val, param=param_name: self.on_slider_changed(param, val))
            self.sliders[f"{param_name}"] = slider
            y_pos -= 0.05
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    def update_plot(self):
        self.img_ax.axis('off')
        im = self.transform()
        # im_crop = im[CROP_BORDER:-CROP_BORDER, CROP_BORDER:-CROP_BORDER] # to conceal transform artifacts
        self.img_ax.imshow(im)
        plt.draw()

    def on_key(self, key_event : KeyEvent):
        key = key_event.key
        print(key)
        if key == "n":
            self.video_no = (self.video_no + 1) % len(self.video_paths)
        elif key == "b":
            self.video_no = (self.video_no - 1) % len(self.video_paths)
        elif key == "r":
            self.interp_state = copy.deepcopy(self._orig_interp_state)
        
        self.interp_state["st_frame"][1] = len(self.video_frame_paths[self.video_no]) - self.interp_state["n_avg"][3]
        self.setup_plot()
        self.update_plot()

    def transform(self):
        start_frame_no = self.interp_state["st_frame"][3]
        n_avg = self.interp_state["n_avg"][3]
        end_frame_no = self.interp_state["st_frame"][3] + n_avg

        frame_paths = self.video_frame_paths[self.video_no][start_frame_no : end_frame_no]
        self.video_frames[self.video_no] = np.array([cv2.imread(path)[:, :, ::-1] for path in frame_paths])

        N, h, w, c = self.video_frames[self.video_no].shape
        assert N == n_avg

        img_out = np.average(self.video_frames[self.video_no], axis=0).astype(np.uint8)
            
        return img_out
    
    def __call__(self):
        self.update_plot()
        plt.show()