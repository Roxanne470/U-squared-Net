#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
# from skimage import transform
# from torch.utils.data import DataLoader
# from dataloader import LoadDataset
# from torchvision import transforms
# from torch.autograd import Variable
from PIL import Image


def normPRED(d):
    """
    This function normalizes a prediction

    """
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def prepare(inp):

    image = inp.resize((512,512))
    image_arr = np.array(image)

    tmp_image = np.zeros((512, 512, 3))
    tmp_image[:, :, 0] = (image_arr[:, :, 0] - 0.485) / 0.229
    tmp_image[:, :, 1] = (image_arr[:, :, 1] - 0.456) / 0.224
    tmp_image[:, :, 2] = (image_arr[:, :, 2] - 0.406) / 0.225
    tmp_image = tmp_image.transpose((2,0,1))

    prepared_img = torch.from_numpy(tmp_image).unsqueeze(0)
    prepared_img = prepared_img.type("torch.FloatTensor")

    return prepared_img






