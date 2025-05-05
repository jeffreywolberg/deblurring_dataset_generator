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

VALID_IMG_EXTENSIONS = [".png", ".jpeg", ".jpg"]

class LevelOneTransformVis:
    def __init__(self):
        self.image_directory = "./data/level1"
        self.image_paths = [join(self.image_directory, fname) for fname in os.listdir(self.image_directory) if splitext(fname)[1] in VALID_IMG_EXTENSIONS]
        self.images = [None for _ in self.image_paths]
        self.image_no = 0

        self.save_dir ='./data/level1/saved_images'
        os.makedirs(self.save_dir, exist_ok=True)

        # Setup matplotlib figure and axes
        self.fig = plt.figure(figsize=(12, 8))

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self._filters = ["box", "triangle", "gaussian", "pixel motion"]
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
                    },
                self._filters[3]:
                    {
                        "umot": [0, 50, 1, 0],
                        "vmot": [0, 50, 1, 0],
                    }
            }

        }
        
        self.filter_state = copy.deepcopy(self._orig_filter_state)
        self.setup_plot()

    def on_slider_changed(self, param, val):
        self.filter_state["filters"][self._filters[self.filter_no]][param][3] = val
        self.update_plot()

    def setup_plot(self):
        self.fig.clear()
        # Create main image display on the right
        self.img_ax = plt.subplot2grid((1, 5), (0, 1), colspan=4)
        self.img_ax.cla()
        self.img_ax.set_title("Transformed Image")
        self.img_ax.axis('off')  # Remove axes and numbers
        
        filter_name = self._filters[self.filter_no]

        self.slider_ax = plt.subplot2grid((1, 5), (0, 0))
        self.slider_ax.clear()  # Clear everything in the axes
        self.slider_ax.set_title(f"{filter_name.capitalize()} Filter")
        self.slider_ax.axis('off')  # Remove axes and numbers

        self.sliders = {}
        y_pos = 0.90
        params = self.filter_state["filters"][filter_name]
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
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    def update_plot(self):
        self.img_ax.cla()
        self.img_ax.axis('off')
        self.img_ax.imshow(self.transform())
        plt.draw()

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
        elif key == "c":
            im = self.transform()
            filter_name = self._filters[self.filter_no]
            filter_state = self.filter_state["filters"][filter_name]
            filter_state_str = '_'.join([f"{k}-{v[3]}" for k, v in filter_state.items()])
            img_basename, img_ext = splitext(basename(self.image_paths[self.image_no]))
            save_path = join(self.save_dir, img_basename + f"_{filter_name}_{filter_state_str}{img_ext}")
            cv2.imwrite(save_path, im[:, :, ::-1])
            print(f"Saved to {save_path}")

            im = cv2.imread(self.image_paths[self.image_no])
            gt_save_path = join(self.save_dir, img_basename + f"_{filter_name}_{filter_state_str}_gt{img_ext}")
            cv2.imwrite(gt_save_path, im)
            print(f"Saved gt to {gt_save_path}")
        else:
            return
        
        self.update_plot()

    def transform(self):
        if self.images[self.image_no] is None:
            self.images[self.image_no] = cv2.imread(self.image_paths[self.image_no])[:, :, ::-1]

        img = np.copy(self.images[self.image_no])

        filter_name = self._filters[self.filter_no]
        if filter_name == "box":
            ksize = self.filter_state["filters"][filter_name]["size"][3]
            img = cv2.boxFilter(img, 3, (ksize, ksize))
        elif filter_name == "triangle":
            ksize = self.filter_state["filters"][filter_name]["size"][3]
            triangle_func = 1 - np.abs(np.linspace(-1, 1, ksize))
            pyramid_filter = np.outer(triangle_func, triangle_func)
            pyramid_filter /= np.sum(pyramid_filter, axis=None)
            img = cv2.filter2D(img, 3, pyramid_filter)
        elif filter_name == "gaussian":
            ksize = self.filter_state["filters"][filter_name]["size"][3]
            sigma = self.filter_state["filters"][filter_name]["sigma"][3]
            img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        elif filter_name == "pixel motion":
            umot = self.filter_state["filters"][filter_name]["umot"][3]
            vmot = self.filter_state["filters"][filter_name]["vmot"][3]

            if umot == vmot == 0:
                return img
            
            ang = np.atan2(vmot, umot) # ccw angle from x-axis
            h, w = vmot*2|1, umot*2|1
            kernel = np.zeros((h, w))
            ch, cw = h//2, w//2

            pt1 = (int(cw - (w/2) * np.cos(ang)), int(ch - (h/2) * np.sin(ang)))
            pt2 = (int(cw + (w/2) * np.cos(ang)), int(ch + (h/2) * np.sin(ang)))
            cv2.line(kernel, pt1, pt2, color=1, thickness=1)
            kernel /= np.sum(kernel) # normalize to maintain brightness level
            
            img = cv2.filter2D(img, -1, kernel)
        
        return img
    
    
    def __call__(self):
        self.update_plot()
        plt.show()