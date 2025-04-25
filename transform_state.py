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

class TransformState(ABC):
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

    def __call__(self):
        self.update_plot()
        plt.show()
        

class NonblindDeblurringTransformState(TransformState):
    def __init__(self):
        super().__init__()
        self._filters = ["box", "triangle", "gaussian"]
        self.filter_no = 0
        self._orig_filter_state = {
            "filters": {
                self._filters[0]: 
                    {
                        "size": [1, 21, 2, 5] # min, max, step, cur
                    }, 
                self._filters[1]: 
                    {
                        "size": [1, 21, 2, 5] # min, max, step, cur
                    },
                self._filters[2]: 
                    {
                        "size": [1, 21, 2, 5], # min, max, step, cur
                        "sigma": [1, 20, 1, 2] # min, max, step, cur
                    }
            }

        }
        
        self.filter_state = copy.deepcopy(self._orig_filter_state)
        self.setup_plot()

    def on_slider_changed(self, param, val):
        self.filter_state["filters"][self._filters[self.filter_no]][param][3] = val
        self.update_plot()

    def setup_plot(self):
        # Create main image display on the right
        self.img_ax = plt.subplot2grid((1, 5), (0, 1), colspan=4)
        self.img_ax.cla()
        self.img_ax.set_title("Transformed Image")
        self.img_ax.axis('off')  # Remove axes and numbers
        
        filter_name = self._filters[self.filter_no]

        # Create slider panel on the left
        if hasattr(self, 'slider_ax'):
            self.slider_ax.remove()  # Remove the old axes completely
        self.slider_ax = plt.subplot2grid((1, 5), (0, 0))
        self.slider_ax.clear()  # Clear everything in the axes
        self.slider_ax.set_title(f"{filter_name.capitalize()} Filter")
        self.slider_ax.axis('off')  # Remove axes and numbers

        if hasattr(self, 'sliders'):
            for slider in self.sliders.values():
                slider.ax.remove()

        # Create sliders for each parameter in state
        self.sliders = {}
        y_pos = 0.90
        params = self.filter_state["filters"][filter_name]
        print(filter_name)
        for param_name, values in params.items():
            ax = plt.axes([0.05, y_pos, 0.15, 0.03])
            plt.cla()
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
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

    def update_plot(self):
        self.img_ax.imshow(self.transform())
        plt.draw()

    @property
    def data_directory(self):
        return "./data/level1"
    

    def on_key(self, key_event : KeyEvent):
        key = key_event.key
        print(key)
        if key == "n":
            self.image_no = (self.image_no + 1) % len(self.image_paths)
        elif key == "b":
            self.image_no = (self.image_no - 1) % len(self.image_paths)
        elif key.lower() == "t":
            self.filter_no = (self.filter_no + (1 if key == "t" else -1)) % len(self._filters)
            self.setup_plot()
        elif key == "r":
            self.filter_no = 0
            self.filter_state = copy.deepcopy(self._orig_filter_state)
            self.setup_plot()

        self.update_plot()

    def transform(self):
        if self.images[self.image_no] is None:
            self.images[self.image_no] = cv2.imread(self.image_paths[self.image_no])[:, :, ::-1]
            print(self.images[self.image_no].shape)

        img = np.copy(self.images[self.image_no])

        filter_name = self._filters[self.filter_no]
        if filter_name == "box":
            ksize = self.filter_state["filters"][filter_name]["size"][3]
            print(ksize)
            img = cv2.boxFilter(img, 3, (ksize, ksize))
        elif filter_name == "triangle":
            ksize = self.filter_state["filters"][filter_name]["size"][3]
            triangle_func = 1 - np.abs(np.linspace(-1, 1, ksize))
            pyramid_filter = np.outer(triangle_func, triangle_func)
            pyramid_filter /= np.sum(pyramid_filter, axis=None)
            print(np.sum(pyramid_filter, axis=None))
            img = cv2.filter2D(img, 3, pyramid_filter)
        elif filter_name == "gaussian":
            ksize = self.filter_state["filters"][filter_name]["size"][3]
            sigma = self.filter_state["filters"][filter_name]["sigma"][3]
            img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        
        return img
    
