# My implementation of the SLID module from
# https://github.com/maciejczyzewski/neural-chessboard/

from typing import Tuple
import numpy as np
import cv2


arr = np.array
# Four parameters are taken from the original code and
# correspond to four possible cases that need correction:
# low light, overexposure, underexposure, and blur
CLAHE_PARAMS = [[3,   (2, 6),    5],  # @1
                [3,   (6, 2),    5],  # @2
                [5,   (3, 3),    5],  # @3
                [0,   (0, 0),    0]]  # EE


def slid_clahe(img, limit=2, grid=(3, 3), iters=5):
    """repair using CLAHE algorithm (adaptive histogram equalization)"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(iters):
        img = cv2.createCLAHE(clipLimit=limit,
                              tileGridSize=grid).apply(img)
    if limit != 0:
        kernel = np.ones((10, 10), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def slid_detector(img, alfa=150, beta=2):
    """detect lines using Hough algorithm"""
    __lines, lines = [], cv2.HoughLinesP(img, rho=1, theta=np.pi/360*beta,
                                         threshold=40, minLineLength=50, maxLineGap=15)  # [40, 40, 10]
    if lines is None:
        return []
    for line in np.reshape(lines, (-1, 4)):
        __lines += [[[int(line[0]), int(line[1])],
                     [int(line[2]), int(line[3])]]]
    return __lines


def slid_canny(img, sigma=0.25):
    """apply Canny edge detector (automatic thresh)"""
    v = np.median(img)
    img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (7, 7), 2)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)


def pSLID(img, thresh=150):
    """find all lines using different settings"""
    segments = []
    i = 0
    for key, arr in enumerate(CLAHE_PARAMS):
        tmp = slid_clahe(img, limit=arr[0], grid=arr[1], iters=arr[2])
        curr_segments = list(slid_detector(slid_canny(tmp), thresh))
        segments += curr_segments
        i += 1
        # print("FILTER: {} {} : {}".format(i, arr, len(curr_segments)))
    return segments


all_points = []


def SLID(img, segments):
    global all_points
    all_points = []

    pregroup, group, hashmap, raw_lines = [[], []], {}, {}, []

    dists = {}

    def dist(a, b):
        h = hash("dist"+str(a)+str(b))
        if h not in dists:
            dists[h] = np.linalg.norm(arr(a)-arr(b))
        return dists[h]

    parents = {}

    def find(x):
        if x not in parents:
            parents[x] = x
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]

    def union(a, b):
        par_a = find(a)
        par_b = find(b)
        parents[par_a] = par_b
        group[par_b] |= group[par_a]

    def height(line, pt):
        v = np.cross(arr(line[1])-arr(line[0]), arr(pt)-arr(line[0]))
        # Using dist() to speed up distance look-up since the 2-norm
        # is used many times
        return np.linalg.norm(v)/dist(line[1], line[0])

    def are_similar(l1, l2):
        '''See Sec.3.2.2 in Czyzewski et al.'''
        a = dist(l1[0], l1[1])
        b = dist(l2[0], l2[1])

        x1 = height(l2, l1[0])
        x2 = height(l2, l1[1])
        y1 = height(l1, l2[0])
        y2 = height(l1, l2[1])

        if x1 < 1e-8 and x2 < 1e-8 and y1 < 1e-8 and y2 < 1e-8:
            return True

        # print("l1: %s, l2: %s" % (str(l1), str(l2)))
        # print("x1: %f, x2: %f, y1: %f, y2: %f" % (x1, x2, y1, y2))
        gamma = 0.25 * (x1+x2+y1+y2)
        # print("gamma:", gamma)

        img_width = 500
        img_height = 282
        p = 0.
        A = img_width*img_height
        w = np.pi/2 / np.sqrt(np.sqrt(A))
        t_delta = p*w
        t_delta = 0.0625
        # t_delta = 0.05

        delta = (a+b) * t_delta

        return (a/gamma > delta) and (b/gamma > delta)

    def generate_line(a, b, n):
        points = []
        for i in range(n):
            x = a[0] + (b[0] - a[0]) * (i/n)
            y = a[1] + (b[1] - a[1]) * (i/n)
            points += [[int(x), int(y)]]
        return points

    def analyze(group):
        global all_points
        points = []
        for idx in group:
            points += generate_line(*hashmap[idx], 10)
        _, radius = cv2.minEnclosingCircle(arr(points))
        w = radius * np.pi / 2
        vx, vy, cx, cy = cv2.fitLine(arr(points), cv2.DIST_L2, 0, 0.01, 0.01)
        all_points += points
        return [[int(cx-vx*w), int(cy-vy*w)], [int(cx+vx*w), int(cy+vy*w)]]

    for l in segments:
        h = hash(str(l))
        # Initialize the line
        hashmap[h] = l
        group[h] = set([h])
        parents[h] = h

        wid = l[0][0] - l[1][0]
        hei = l[0][1] - l[1][1]

        # Divide lines into more horizontal vs more vertical
        # to speed up comparison later
        if abs(wid) < abs(hei):
            pregroup[0].append(l)
        else:
            pregroup[1].append(l)

    for lines in pregroup:
        for i in range(len(lines)):
            l1 = lines[i]
            h1 = hash(str(l1))
            # We're looking for the root line of each disjoint set
            if parents[h1] != h1:
                continue
            for j in range(i+1, len(lines)):
                l2 = lines[j]
                h2 = hash(str(l2))
                if parents[h2] != h2:
                    continue
                if are_similar(l1, l2):
                    # Merge lines into a single disjoint set
                    union(h1, h2)

    for h in group:
        if parents[h] != h:
            continue
        raw_lines += [analyze(group[h])]

    return raw_lines


def slid_tendency(raw_lines, s=4):
    lines = []
    def scale(x, y, s): return int(x * (1+s)/2 + y * (1-s)/2)
    for a, b in raw_lines:
        a[0] = scale(a[0], b[0], s)
        a[1] = scale(a[1], b[1], s)
        b[0] = scale(b[0], a[0], s)
        b[1] = scale(b[1], a[1], s)
        lines += [[a, b]]
    return lines


def detect_lines(img):
    segments = pSLID(img)
    raw_lines = SLID(img, segments)
    lines = slid_tendency(raw_lines)
    return lines
