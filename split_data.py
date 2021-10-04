# This script processes all the labeled data, previously created by
# running create_labels.py, into train/validation/test subsamples, which
# are then used in training the CNN by train.py.

from pathlib import Path
from shutil import copy
import os
import random
from train import DATA_FOLDER

TRAIN = 0.7
VALIDATION = 0.2
TEST = 0.1


def split_data():
    '''Splits the data stored in "data/labeled/" and stores it in "data/CNN/"
    according to the ratios for training, validation, and test data.'''
    assert (TRAIN + VALIDATION + TEST) - 1 < 1e-8
    # Find all images in data/labeled and its subdirectories
    total_size = 0
    for dirpath, dirnames, filenames in os.walk("data/labeled/"):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            # Randomly decide whether an instance is used to train, validate, or test
            rand = random.uniform(0, 1)
            label = dirpath.split('/')[-1]
            if rand < TRAIN:
                data_type = 'train'
            elif rand < TRAIN+VALIDATION:
                data_type = 'validation'
            else:
                data_type = 'test'
            path = DATA_FOLDER + '%s/%s/' % (data_type, label)
            # Make a directory if it doesn't exist
            Path(path).mkdir(parents=True, exist_ok=True)
            copy(dirpath+'/'+filename, path)


if __name__ == '__main__':
    split_data()
