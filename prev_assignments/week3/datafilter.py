import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torch.utils.data import Dataset

ROWS = 64
COLS = 64
CHANNELS = 3

num_epochs = 2
learning_rate = 0.01
batch_size = 100
datapath = "../PetImages/"

cats = [datapath + "Cat/" + i for i in os.listdir(datapath + "Cat/")]
dogs = [datapath + "Dog/" + i for i in os.listdir(datapath + "Dog/")]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if(type(img) == type(None)):
        print(file_path)

for i, image_file in enumerate(cats):
    read_image(image_file)

for i, image_file in enumerate(dogs):
    read_image(image_file)