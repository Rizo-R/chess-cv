import numpy as np
import cv2
import glob
from pathlib import Path
from matplotlib import pyplot as plt
from shutil import copy
import os
import random

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import preprocess_input


from rescale import *
from slid import detect_lines
from laps import LAPS
from llr import LLR, llr_pad

def split_data(train=0.7, validation=0.2, test=0.1):
    '''Splits the data stored in data/labeled and stores it in data/train
    according to the ratios for training, validation, and test data.'''
    assert (train + validation + test) - 1 < 1e-8
    # Find all images in data/labeled and its subdirectories
    total_size = 0
    for dirpath, dirnames, filenames in os.walk("data/labeled/"):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            # Randomly decide whether an instance is used to train, validate, or test
            rand = random.uniform(0,1)
            label = dirpath.split('/')[-1]
            if rand < train:
                data_type = 'train'
            elif rand < train+validation:
                data_type = 'validation'
            else:
                data_type = 'test'
            path = 'data/CNN/%s/%s/' % (data_type, label)
            # Make a directory if it doesn't exist
            Path(path).mkdir(parents=True, exist_ok=True)
            copy(dirpath+'/'+filename, path)





if __name__ == '__main__':
    split_data(0.7, 0.2, 0.1)
    print("Done splitting data!")
    