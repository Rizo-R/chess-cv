# This script preprocess esoriginal pictures and turns them into 2D-projections.
# The data is then used in create_labels.py.

import numpy as np
import cv2
import glob
from pathlib import Path
from matplotlib import pyplot as plt

from rescale import *
from slid import detect_lines
from laps import LAPS
from llr import LLR, llr_pad

RAW_DATA_FOLDER = './data/raw/games/'
PREPROCESSED_FOLDER = './data/preprocessed/games/'


def preprocess_image(path, final_folder="", filename="",  save=False):
    ''' Reads and preprocesses image from [path] and saves it as [filename] in the [final_folder] is [save] is enabled.'''
    res = cv2.imread(path)[..., ::-1]
    # Crop twice, just like Czyzewski et al. did
    for _ in range(2):
        img, shape, scale = image_resize(res)
        lines = detect_lines(img)
        # filter_lines(lines)
        lattice_points = LAPS(img, lines)
        # Sometimes LLR() or llr_pad() will produce an error. In this case,
        # the picture needs to be retaken
        inner_points = LLR(img, lattice_points, lines)
        four_points = llr_pad(inner_points, img)  # padcrop

        # print(four_points)
        try:
            res = crop(res, four_points, scale)
        except:
            print("WARNING: couldn't crop around outer points")
            res = crop(
                res, inner_points, scale)
    if save:
        # Create the folder if it doesn't exist
        Path(final_folder).mkdir(parents=True, exist_ok=True)
        plt.imsave("%s/%s" % (final_folder, filename), res)
    return res


def preprocess_games(game_list):
    '''Preprocesses all games in the given list. Assuming there are two 
    versions of each: original and reversed; in reversed, the board is flipped.
    I included this to improve the performance of CNN in situations when
    White has pieces on ranks 5-8 or Black has pieces on ranks 1-4.'''
    for game_name in game_list:
        for ver in ['orig', 'rev']:
            img_filename_list = []
            folder_name = RAW_DATA_FOLDER + '%s/%s/*' % (game_name, ver)
            for path_name in glob.glob(folder_name):
                img_filename_list.append(path_name)

            count = 0
            img_filename_list.sort(key=lambda s: int(
                s.split('/')[-1].split('.')[0]))
            for path in img_filename_list:
                count += 1
                final_folder = PREPROCESSED_FOLDER + \
                    "%s/%s/" % (game_name, ver)
                preprocess_image(path, final_folder=final_folder,
                                 filename="%i.png" % count, save=True)
            print("Done saving in %s." % final_folder)


if __name__ == '__main__':
    game_list = ['runau_schmidt', 'hewitt_steinitz', 'bertok_fischer', 'karpov_kasparov',
                 'alekhine_nimzowitsch', 'rossolimo_reissmann', 'anderssen_dufresne', 'thorsteinsson_karlsson']
    preprocess_games(game_list)
