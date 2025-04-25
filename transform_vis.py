from abc import ABC, abstractmethod
import copy
from matplotlib.backend_bases import KeyEvent
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import glob
import cv2
from os.path import join, basename, abspath, exists, splitext

import numpy as np

VALID_IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]

class TransformVis(ABC):
    def __init__(self):
        self.image_paths = [join(self.data_directory, fname) for fname in os.listdir(self.data_directory) if splitext(fname)[1] in VALID_IMG_EXTENSIONS]
        self.images = [None for _ in self.image_paths]
        self.image_no = 0

        # Setup matplotlib figure and axes
        self.fig = plt.figure(figsize=(12, 8))

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    @property
    @abstractmethod
    def data_directory(self):
        ...

    @abstractmethod
    def on_key(self, key_event : KeyEvent):
        ...

    def __call__(self):
        self.update_plot()
        plt.show()
        


    
