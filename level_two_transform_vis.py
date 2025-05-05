from abc import ABC, abstractmethod
import copy
from matplotlib.backend_bases import KeyEvent
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import glob
import cv2
from os.path import join, basename, abspath, exists, splitext

from scipy.interpolate import RegularGridInterpolator

import numpy as np

VALID_IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".pgm", ".ppm", ".npy"]

CROP_BORDER = 30

# NYU depth v2 dataset
fx = 518.8579
fy = 519.4696
cx = 325.5824
cy = 253.7362

# # iPhone X intrinsics
# fx = 4032
# fy = 4032
# cx = 2016
# cy = 1512

class LevelTwoTransformVis:
    def __init__(self):
        self._orig_motion_state = {
            "roll": [-5, 5, 1, 0],
            "pitch": [-5, 5, 1, 0],
            "yaw": [-5, 5, 1, 0],
            "xmot": [0, 4, .2, 0],
            "ymot": [0, 4, .2, 0],
        }
        
        self.motion_state = copy.deepcopy(self._orig_motion_state)

        self.save_dir ='./data/level2/saved_images'
        os.makedirs(self.save_dir, exist_ok=True)

        self.image_no = 0
        self.image_directory = "./data/level2/images"
        self.image_paths = sorted([join(self.image_directory, fname) for fname in os.listdir(self.image_directory) if splitext(fname)[1] in VALID_IMG_EXTENSIONS])
        self.images = [None for _ in self.image_paths]

        self.depth_map_directory = "./data/level2/depth_maps"
        self.depth_map_paths = sorted([join(self.depth_map_directory, fname) for fname in os.listdir(self.depth_map_directory) if splitext(fname)[1] in VALID_IMG_EXTENSIONS])
        self.depth_maps = [None for _ in self.depth_map_paths]

        assert len(self.depth_map_paths) == len(self.image_paths)

        # Setup matplotlib figure and axes
        self.fig = plt.figure(figsize=(12, 8))

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.setup_plot()

    def on_slider_changed(self, param, val):
        self.motion_state[param][3] = val
        self.update_plot()

    def setup_plot(self):
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
        for param_name, values in self.motion_state.items():
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
        im_crop = im[CROP_BORDER:-CROP_BORDER, CROP_BORDER:-CROP_BORDER] # to conceal transform artifacts
        self.img_ax.imshow(im_crop)
        plt.draw()

    def on_key(self, key_event : KeyEvent):
        key = key_event.key
        print(key)
        if key == "n":
            self.image_no = (self.image_no + 1) % len(self.image_paths)
        elif key == "b":
            self.image_no = (self.image_no - 1) % len(self.image_paths)
        elif key == "r":
            self.motion_state = copy.deepcopy(self._orig_motion_state)
            self.setup_plot()
        elif key == "c":
            im = self.transform()
            im_crop = im[CROP_BORDER:-CROP_BORDER, CROP_BORDER:-CROP_BORDER] # to conceal transform artifacts
            motion_state_str = '_'.join([f"{k}{v[3]}" for k, v in self.motion_state.items()])
            img_basename, img_ext = splitext(basename(self.image_paths[self.image_no]))
            save_path = join(self.save_dir, img_basename + f"_{motion_state_str}{img_ext}")
            cv2.imwrite(save_path, im_crop[:, :, ::-1])
            print(f"Saved transformed img to {save_path}")

            im_crop = cv2.imread(self.image_paths[self.image_no])[CROP_BORDER:-CROP_BORDER, CROP_BORDER:-CROP_BORDER] # to match transformed image
            gt_save_path = join(self.save_dir, img_basename + f"_{motion_state_str}_gt{img_ext}")
            cv2.imwrite(gt_save_path, im_crop)
            print(f"Saved gt to {gt_save_path}")
        else:
            return

        self.update_plot()

    def transform(self):
        if self.images[self.image_no] is None:
            self.images[self.image_no] = cv2.imread(self.image_paths[self.image_no])[:, :, ::-1]
            if splitext(self.depth_map_paths[self.image_no])[1] == ".npy":
                self.depth_maps[self.image_no] = np.load(self.depth_map_paths[self.image_no])

        img = np.copy(self.images[self.image_no])
        Z = self.depth_maps[self.image_no]
        # Z = np.clip(self.depth_maps[self.image_no], 0, 1000)
        h, w = img.shape[:2]

        v, u = np.arange(h), np.arange(w)

        X = (u[None, :] - cx) * Z / fx
        Y = (v[:, None] - cy) * Z / fy

        roll = np.deg2rad(self.motion_state["roll"][3])
        pitch = np.deg2rad(self.motion_state["pitch"][3])
        yaw = np.deg2rad(self.motion_state["yaw"][3])
        xmot = np.deg2rad(self.motion_state["xmot"][3])
        ymot = np.deg2rad(self.motion_state["ymot"][3])

        T = 10

        # omega = np.array([pitch, yaw, roll])
        # omega_skew_sym = np.array([
        #     [0, -yaw, pitch],
        #     [yaw, 0, -roll],
        #     [-pitch, roll, 0]
        # ])

        # omega_norm = np.linalg.norm(omega)
        # R = np.eye(3) + np.sin(omega_norm) * omega_skew_sym + (1 - np.cos(omega_norm)) * np.linalg.matrix_power(omega_skew_sym, 2)
        # XYZ = np.stack([X, Y, Z], axis=-1)
        # Xtrans = XYZ @ R.T # right multiplication, 

        img_out = np.zeros(img.shape, dtype=np.float64)
        
        interp = RegularGridInterpolator((v, u), img, bounds_error=False, fill_value=0)
        
        for t in range(T):
            dX = (t / T) * (xmot + -yaw * Y + pitch * Z)
            dY = (t / T) * (ymot + yaw * X - roll * Z)
            dZ = (t / T) * (-pitch * X + roll * Y)
            u_trans = fx * (X + dX) / (Z + dZ) + cx
            v_trans = fy * (Y + dY) / (Z + dZ) + cy

            # v_ = np.clip(np.round(v_trans).astype(int), 0, h-1)
            # u_ = np.clip(np.round(u_trans).astype(int), 0, w-1)
            # img_out[v_, u_] += img / T
            
            img_out += interp((v_trans, u_trans)) / T
        
        img_out = img_out.astype(np.uint8)

        return img_out
    
    def __call__(self):
        self.update_plot()
        plt.show()