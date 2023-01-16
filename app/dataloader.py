#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tarfile
import requests
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset

def download_model_weights():

    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0JV6EN/LargeData/u2net.tgz"
    target_path = "u2net.tgz"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())

    file = tarfile.open("u2net.tgz")
    file.extractall()
    file.close()


class LoadDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_name_list = [x for x in os.listdir(img_dir) if not x.startswith(".")]

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.img_name_list[idx])
        image = io.imread(img_name)
        imidx = np.array([idx])
        sample = {'imidx': imidx, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

