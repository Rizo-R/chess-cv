# This script uses the CNN architecture defined in train.py to analyze
# the given image (can be used autonomously, i.e. without explicitly running
# train.py). Given an image path or multiple image paths separated by space,
# turns it/them into a chess position/positions and saves a .png file/files.

# References:
# https://stackoverflow.com/questions/56754543/generate-chess-board-diagram-from-an-array-of-positions-in-python
# https://chess.stackexchange.com/questions/28870/render-a-chessboard-from-a-pgn-file
# https://www.programcreek.com/python/?code=yaqwsx%2FPcbDraw%2FPcbDraw-master%2Fpcbdraw%2Fpcbdraw.py#

import chess
import chess.svg
import io
import numpy as np
import os
import sys
import wand.color
import wand.image
from matplotlib import pyplot as plt
from pathlib import Path
from preprocess import preprocess_image
from train import create_model
from wand.api import library

PIECES = ['Empty', 'Rook_White', 'Rook_Black', 'Knight_White', 'Knight_Black', 'Bishop_White',
          'Bishop_Black', 'Queen_White', 'Queen_Black', 'King_White', 'King_Black', 'Pawn_White', 'Pawn_Black']
# PIECES = ['Empty','Rook','Knight','Bishop','Queen','King','Pawn']
PIECES.sort()
LABELS = {
    'Empty': '.',
    'Rook_White': 'R',
    'Rook_Black': 'r',
    'Knight_White': 'N',
    'Knight_Black': 'n',
    'Bishop_White': 'B',
    'Bishop_Black': 'b',
    'Queen_White': 'Q',
    'Queen_Black': 'q',
    'King_White': 'K',
    'King_Black': 'k',
    'Pawn_White': 'P',
    'Pawn_Black': 'p',
}
TEMP_SVG_FOLDER = './'
PNG_FOLDER = './results/'


def classify_image(img):
    '''Given an image of a single piece, classifies it into one of the classes
    defined in PIECES.'''
    y_prob = model.predict(img.reshape(1, 300, 150, 3))
    y_pred = y_prob.argmax()
    return PIECES[y_pred]


def analyze_board(img):
    '''Given an image of an entire board, returns an array representing 
    the predicted chess position. Note that the first row of the array
    corresponds to the 8th rank of the chess board (i.e. where all the 
    black non-pawn pieces are located initially).'''
    arr = []
    M = img.shape[0]//8
    N = img.shape[1]//8
    # for y in range(img.shape[0]-1, -1, -M):
    for y in range(M-1, img.shape[1], M):
        row = []
        for x in range(0, img.shape[1], N):
            sub_img = img[max(0, y-2*M):y, x:x+N]
            if y-2*M < 0:
                sub_img = np.concatenate(
                    (np.zeros((2*M-y, N, 3)), sub_img))
                sub_img = sub_img.astype(np.uint8)

            piece = classify_image(sub_img)
            row.append(LABELS[piece])
        arr.append(row)

    # If there is a Queen but not a King then replace it with a King since
    # the King was probably misclassified as a Queen because the two look
    # very similar.
    blackKing = False
    whiteKing = False
    whitePos = (-1, -1)
    blackPos = (-1, -1)
    for i in range(8):
        for j in range(8):
            if arr[i][j] == 'K':
                whiteKing = True
            if arr[i][j] == 'k':
                blackKing = True
            if arr[i][j] == 'Q':
                whitePos = (i, j)
            if arr[i][j] == 'q':
                blackPos = (i, j)
    if not whiteKing and whitePos[0] >= 0:
        arr[whitePos[0]][whitePos[1]] = 'K'
    if not blackKing and blackPos[0] >= 0:
        arr[blackPos[0]][blackPos[1]] = 'k'

    return arr


def board_to_fen(board):
    '''Given an array representing a board position (from analyze_board()),
    converts it to FEN representation with default additional parameters
    (white to move, can castle both ways, etc).
    Returns a string representing a FEN position.'''
    # Use StringIO to build string more efficiently than concatenating
    with io.StringIO() as s:
        for row in board:
            empty = 0
            for cell in row:
                if cell != '.':
                    if empty > 0:
                        s.write(str(empty))
                        empty = 0
                    s.write(cell)
                else:
                    empty += 1
            if empty > 0:
                s.write(str(empty))
            s.write('/')
        # Move one position back to overwrite last '/'
        s.seek(s.tell() - 1)
        # If you do not have the additional information choose what to put
        s.write(' w KQkq - 0 1')
        return s.getvalue()


def fen_to_svg(fen):
    '''Converts a given string to a SVG file and saves it temporarily.'''
    board = chess.Board(fen)
    boardsvg = chess.svg.board(board=board)
    f = open(TEMP_SVG_FOLDER + 'temp.SVG', "w")
    f.write(boardsvg)
    f.close()


def svg_to_png(infile, outfile, dpi=300):
    '''Loads the temporarily greated SVG file, converts it to PNG, and saves
    it. Removes the temporary SVG file.'''
    with wand.image.Image(resolution=300) as image:
        with wand.color.Color('transparent') as background_color:
            library.MagickSetBackgroundColor(image.wand,
                                             background_color.resource)
        image.read(filename=infile, resolution=300)
        png_image = image.make_blob("png32")
        with open(outfile, "wb") as out:
            out.write(png_image)
        # Once done, remove the temporary SVG file
        os.remove(infile)


if __name__ == '__main__':
    for IMAGE_PATH in sys.argv[1:]:
        image_name = IMAGE_PATH.split('/')[-1]
        file_name = './results/' + image_name
        # Create folder if it doesn't exist
        Path('./results/').mkdir(parents=True, exist_ok=True)
        # Create a CNN architecture and load pre-trained weights.
        model = create_model()
        model.load_weights('./model_weights.h5')
        # Load image and convert it to array, then to FEN, then to SVG, then to PNG.
        img = preprocess_image(IMAGE_PATH, save=False)
        arr = analyze_board(img)
        fen = board_to_fen(arr)
        fen_to_svg(fen)
        svg_to_png(infile=TEMP_SVG_FOLDER+'temp.SVG',
                   outfile=file_name)
        # plt.imshow(cv2.imread(file_name))
        # plt.show()
    print('Done!')
