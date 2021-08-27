import os
import glob
import chess.pgn
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from slid import detect_lines

LABELS = {
    'p': 'Pawn',
    'r': 'Rook',
    'n': 'Knight',
    'b': 'Bishop',
    'q': 'Queen',
    'k': 'King'
}

TOTAL_COUNT = 0


def label_game(game_name, reversed=False):
    '''Reads preprocessed images for the game and labels each square 
    according to the .pgn file.'''

    global TOTAL_COUNT

    # read the game file and set board
    pgn = open("data/pgns/%s.pgn" % game_name)
    game = chess.pgn.read_game(pgn)
    board = game.board()

    move_count = 0
    for move in game.mainline_moves():
        board.push(move)
        move_count += 1

        if not reversed:
            img = cv2.imread(
                "data/preprocessed/games/%s/orig/%i.png" % (game_name, move_count))
        else:
            img = cv2.imread(
                "data/preprocessed/games/%s/rev/%i.png" % (game_name, move_count))

        if img is None:
            return

        imheight, imwidth, _ = img.shape

        # Split each image into 64 equal squares and go through each of them
        # Note that I look at the square above the given square, too (except
        # for the 8th rank).
        M = imheight // 8
        N = imwidth // 8
        i = 0

        # [reversed] determines whether we start from bottom left and go right
        # and then up (if False), or if we start from top right (i.e. the board
        # is flipped) and go left and then down (if True).
        if not reversed:
            for y in range(imheight-1, -1, -M):
                for x in range(0, imwidth, N):
                    piece = str(board.piece_at(i))
                    sub_img = img[max(0, y-2*M):y, x:x+N]
                    if y-2*M < 0:
                        sub_img = np.concatenate((np.zeros((2*M-y, N, 3)), sub_img))
                        sub_img = sub_img.astype(np.uint8)
                    if piece == 'None':
                        final_folder = 'data/labeled/Empty/'
                    # else:
                    #     piece_type = LABELS[piece.lower()]
                    #     if piece.isupper():
                    #         color = 'White'
                    #     else:
                    #         color = 'Black'
                    #     final_folder = 'data/labeled/%s/%s/' % (
                    #         piece_type, color)
                    else:
                        piece_type = LABELS[piece.lower()]
                        final_folder = 'data/labeled/%s/' % piece_type

                    # Create folder if it doesn't exist
                    Path(final_folder).mkdir(parents=True, exist_ok=True)
                    TOTAL_COUNT += 1
                    i += 1
                    plt.imsave(final_folder+'%i.jpg' % TOTAL_COUNT, sub_img)
        else:
            for y in range(M, imheight+1, M):
                # for y in range(0, imheight, M):
                for x in range(imwidth, N-1, -N):
                    piece = str(board.piece_at(i))
                    sub_img = img[max(0, y-2*M):y, x-N:x]
                    if y-2*M < 0:
                        sub_img = np.concatenate((np.zeros((2*M-y, N, 3)), sub_img))
                        sub_img = sub_img.astype(np.uint8)
                    # sub_img = img[y:min(imheight, y+2*M), x-N:x]
                    if piece == 'None':
                        final_folder = 'data/labeled/Empty/'
                    # else:
                    #     piece_type = LABELS[piece.lower()]
                    #     if piece.isupper():
                    #         color = 'White'
                    #     else:
                    #         color = 'Black'
                    #     final_folder = 'data/labeled/%s/%s/' % (
                    #         piece_type, color)
                    else:
                        piece_type = LABELS[piece.lower()]
                        final_folder = 'data/labeled/%s/' % piece_type

                    # Create folder if it doesn't exist
                    Path(final_folder).mkdir(parents=True, exist_ok=True)
                    TOTAL_COUNT += 1
                    i += 1
                    plt.imsave(final_folder+'%i.jpg' % TOTAL_COUNT, sub_img)
        print("Move %i finished." % move_count)

    if not reversed:
        print('Finished labeling game "%s" (original)' % game_name)
    else:
        print('Finished labeling game "%s" (reversed)' % game_name)


if __name__ == '__main__':
    game_list = ['runau_schmidt']
    ver_list = ['orig', 'rev']
    for game_name in game_list:
        for ver in ver_list:
            if ver == 'orig':
                label_game(game_name, reversed=False)
            elif ver == 'rev':
                label_game(game_name, reversed=True)
            else:
                raise ValueError("Wrong version. Need either 'orig' or 'rev'")
