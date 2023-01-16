#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import requests
import tarfile
import torch
# from skimage import transform
# from torch.utils.data import DataLoader
# from dataloader import LoadDataset
# from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from model import U2NET
from preprocess import normPRED, prepare
from dataloader import download_model_weights
import gradio as gr



download_model_weights()
net = U2NET(3, 1)
net.load_state_dict(torch.load("u2net.pth", map_location='cpu'))

def predict(inp):

    processed_image = prepare(inp)
    input_test = Variable(processed_image)

    d = net(input_test)

    # normalization
    pred = 1.0 - d[:, 0, :, :]
    pred = normPRED(pred)

    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    img = Image.fromarray(predict_np * 255).convert('RGB')
    del input_test, d, pred, predict, predict_np

    return img

demo = gr.Interface(fn = predict,
                    inputs = gr.Image(type="pil"),
                    outputs = gr.Image(type="pil"),
                    examples = ["/samples/w1.jpg", "/samples/w2.jpg", "/samples/w3.jpg", "/samples/w4.jpg", "/samples/m1.jpg"]
                    )

demo.launch(server_name="0.0.0.0")
