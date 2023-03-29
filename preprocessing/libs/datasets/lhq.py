from __future__ import absolute_import, print_function

import os.path as osp
from glob import glob

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

from .base import _BaseDataset

class LHQ(_BaseDataset):

    def __init__(self, **kwargs):
        super(LHQ, self).__init__(**kwargs)

    def _set_files(self):

        file_list = sorted(glob(osp.join(self.root,"images","*.png"))) #or jpg
        file_list = [f.split("/")[-1].replace(".png", "") for f in file_list]
        self.files = file_list

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.root, "images",  image_id + ".png")
        # Load an image and label
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label= np.random.rand(1,1)
        return image_id, image, label
