import functools
import numpy as np
import cv2
import math
arr = np.array


def image_scale(pts, scale):
    """scale to original image size"""
    def __loop(x, y): return [x[0] * y, x[1] * y]
    return list(map(functools.partial(__loop, y=1/scale), pts))


def image_resize(img, height=500):
    """resize image to same normalized area (height**2)"""
    pixels = height * height
    shape = list(np.shape(img))
    scale = math.sqrt(float(pixels)/float(shape[0]*shape[1]))
    shape[0] *= scale
    shape[1] *= scale
    img = cv2.resize(img, (int(shape[1]), int(shape[0])))
    img_shape = np.shape(img)
    return img, img_shape, scale


def image_transform(img, points, square_length=150):
    """crop original image using perspective warp"""
    board_length = square_length * 8
    def __dis(a, b): return np.linalg.norm(arr(a)-arr(b))
    def __shi(seq, n=0): return seq[-(n % len(seq)):] + seq[:-(n % len(seq))]
    best_idx, best_val = 0, 10**6
    for idx, val in enumerate(points):
        val = __dis(val, [0, 0])
        if val < best_val:
            best_idx, best_val = idx, val
    pts1 = np.float32(__shi(points, 4 - best_idx))
    pts2 = np.float32([[0, 0], [board_length, 0],
                       [board_length, board_length], [0, board_length]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    W = cv2.warpPerspective(img, M, (board_length, board_length))
    return W


def crop(img, pts, scale):
    """crop using 4 points transform"""
    pts_orig = image_scale(pts, scale)
    img_crop = image_transform(img, pts_orig)
    return img_crop
