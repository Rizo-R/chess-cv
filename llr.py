# Code taken from
# https://github.com/maciejczyzewski/neural-chessboard/

from laps import laps_intersections, laps_cluster
from slid import slid_tendency
import scipy
import cv2
import pyclipper
import numpy as np
import matplotlib.path
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import collections
import itertools
import random
import math
import sklearn.cluster
from copy import copy
na = np.array


################################################################################


def llr_normalize(points): return [[int(a), int(b)] for a, b in points]


def llr_correctness(points, shape):
    __points = []
    for pt in points:
        if pt[0] < 0 or pt[1] < 0 or \
            pt[0] > shape[1] or \
                pt[1] > shape[0]:
            continue
        __points += [pt]
    return __points


def llr_unique(a):
    indices = sorted(range(len(a)), key=a.__getitem__)
    indices = set(next(it) for k, it in
                  itertools.groupby(indices, key=a.__getitem__))
    return [x for i, x in enumerate(a) if i in indices]


def llr_polysort(pts):
    """sort points clockwise"""
    mlat = sum(x[0] for x in pts) / len(pts)
    mlng = sum(x[1] for x in pts) / len(pts)

    def __sort(x):  # main math --> found on MIT site
        return (math.atan2(x[0]-mlat, x[1]-mlng) +
                2*math.pi) % (2*math.pi)
    pts.sort(key=__sort)
    return pts


def llr_polyscore(cnt, pts, cen, alfa=5, beta=2):
    a = cnt[0]
    b = cnt[1]
    c = cnt[2]
    d = cnt[3]

    area = cv2.contourArea(cnt)
    t2 = area < (4 * alfa * alfa) * 5
    if t2:
        return 0

    gamma = alfa/1.5

    pco = pyclipper.PyclipperOffset()
    pco.AddPath(cnt, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    pcnt = matplotlib.path.Path(pco.Execute(gamma)[0])  # FIXME: alfa/1.5
    wtfs = pcnt.contains_points(pts)
    pts_in = min(np.count_nonzero(wtfs), 49)
    t1 = pts_in < min(len(pts), 49) - 2 * beta - 1
    if t1:
        return 0

    A = pts_in
    B = area

    def nln(l1, x, dx): return \
        np.linalg.norm(np.cross(na(l1[1])-na(l1[0]),
                                na(l1[0])-na(x)))/dx
    pcnt_in = []
    i = 0
    for pt in wtfs:
        if pt:
            pcnt_in += [pts[i]]
        i += 1

    def __convex_approx(points, alfa=0.001):
        hull = scipy.spatial.ConvexHull(na(points)).vertices
        cnt = na([points[pt] for pt in hull])
        return cnt

    cnt_in = __convex_approx(na(pcnt_in))

    points = cnt_in
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    cen2 = (sum(x) / len(points),
            sum(y) / len(points))

    G = np.linalg.norm(na(cen)-na(cen2))

    """
	cnt_in = __convex_approx(na(pcnt_in))
	S = cv2.contourArea(na(cnt_in))
	if S < B: E += abs(S - B)
	cnt_in = __convex_approx(na(list(cnt_in)+list(cnt)))
	S = cv2.contourArea(na(cnt_in))
	if S > B: E += abs(S - B)
	"""

    a = [cnt[0], cnt[1]]
    b = [cnt[1], cnt[2]]
    c = [cnt[2], cnt[3]]
    d = [cnt[3], cnt[0]]
    lns = [a, b, c, d]
    E = 0
    F = 0
    for l in lns:
        d = np.linalg.norm(na(l[0])-na(l[1]))
        for p in cnt_in:
            r = nln(l, p, d)
            if r < gamma:
                E += r
                F += 1
    if F == 0:
        return 0
    E /= F

    if B == 0 or A == 0:
        return 0

    # See Eq.11 and Sec.3.4 in the paper

    C = 1+(E/A)**(1/3)
    D = 1+(G/A)**(1/5)
    R = (A**4)/((B**2) * C * D)

    # print(R*(10**12), A, "|", B, C, D, "|", E, G)

    return R

################################################################################

# LAPS, SLID


def LLR(img, points, lines):
    old = points

    def __convex_approx(points, alfa=0.01):
        hull = scipy.spatial.ConvexHull(na(points)).vertices
        cnt = na([points[pt] for pt in hull])
        approx = cv2.approxPolyDP(cnt, alfa *
                                  cv2.arcLength(cnt, True), True)
        return llr_normalize(itertools.chain(*approx))

    __cache = {}

    def __dis(a, b):
        idx = hash("__dis" + str(a) + str(b))
        if idx in __cache:
            return __cache[idx]
        __cache[idx] = np.linalg.norm(na(a)-na(b))
        return __cache[idx]

    def nln(l1, x, dx): return \
        np.linalg.norm(np.cross(na(l1[1])-na(l1[0]),
                                na(l1[0])-na(x)))/dx

    pregroup = [[], []]
    S = {}

    points = llr_correctness(llr_normalize(points), img.shape)

    __points = {}
    points = llr_polysort(points)
    __max, __points_max = 0, []
    alfa = math.sqrt(cv2.contourArea(na(points))/49)
    X = sklearn.cluster.DBSCAN(eps=alfa*4).fit(points)
    for i in range(len(points)):
        __points[i] = []
    for i in range(len(points)):
        if X.labels_[i] != -1:
            __points[X.labels_[i]] += [points[i]]
    for i in range(len(points)):
        if len(__points[i]) > __max:
            __max = len(__points[i])
            __points_max = __points[i]
    if len(__points) > 0 and len(points) > 49/2:
        points = __points_max
    # print(X.labels_)

    ring = __convex_approx(llr_polysort(points))

    n = len(points)
    beta = n*(5/100)
    alfa = math.sqrt(cv2.contourArea(na(points))/49)

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points),
                sum(y) / len(points))

    # print(alfa, beta, centroid)

    def __v(l):
        y_0, x_0 = l[0][0], l[0][1]
        y_1, x_1 = l[1][0], l[1][1]

        x_2 = 0
        t = (x_0-x_2)/(x_0-x_1+0.0001)
        a = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)][::-1]

        x_2 = img.shape[0]
        t = (x_0-x_2)/(x_0-x_1+0.0001)
        b = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)][::-1]

        poly1 = llr_polysort([[0, 0], [0, img.shape[0]], a, b])
        s1 = llr_polyscore(na(poly1), points, centroid, beta=beta, alfa=alfa/2)
        poly2 = llr_polysort([a, b,
                              [img.shape[1], 0], [img.shape[1], img.shape[0]]])
        s2 = llr_polyscore(na(poly2), points, centroid, beta=beta, alfa=alfa/2)

        return [a, b], s1, s2

    def __h(l):
        x_0, y_0 = l[0][0], l[0][1]
        x_1, y_1 = l[1][0], l[1][1]

        x_2 = 0
        t = (x_0-x_2)/(x_0-x_1+0.0001)
        a = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)]

        x_2 = img.shape[1]
        t = (x_0-x_2)/(x_0-x_1+0.0001)
        b = [int((1-t)*x_0+t*x_1), int((1-t)*y_0+t*y_1)]

        poly1 = llr_polysort([[0, 0], [img.shape[1], 0], a, b])
        s1 = llr_polyscore(na(poly1), points, centroid, beta=beta, alfa=alfa/2)
        poly2 = llr_polysort([a, b,
                              [0, img.shape[0]], [img.shape[1], img.shape[0]]])
        s2 = llr_polyscore(na(poly2), points, centroid, beta=beta, alfa=alfa/2)

        return [a, b], s1, s2

    for l in lines:
        for p in points:
            t1 = nln(l, p, __dis(*l)) < alfa
            t2 = nln(l, centroid, __dis(*l)) > alfa * 2.5

            if t1 and t2:
                tx, ty = l[0][0]-l[1][0], l[0][1]-l[1][1]
                if abs(tx) < abs(ty):
                    ll, s1, s2 = __v(l)
                    o = 0
                else:
                    ll, s1, s2 = __h(l)
                    o = 1
                if s1 == 0 and s2 == 0:
                    continue
                pregroup[o] += [ll]

    pregroup[0] = llr_unique(pregroup[0])
    pregroup[1] = llr_unique(pregroup[1])

    # print("---------------------")
    # print(pregroup)
    for v in itertools.combinations(pregroup[0], 2):
        for h in itertools.combinations(pregroup[1], 2):
            poly = laps_intersections([v[0], v[1], h[0], h[1]])
            poly = llr_correctness(poly, img.shape)
            if len(poly) != 4:
                continue
            poly = na(llr_polysort(llr_normalize(poly)))
            if not cv2.isContourConvex(poly):
                continue
            # print("Poly:", -llr_polyscore(poly, points, centroid,
            #                               beta=beta, alfa=alfa/2))
            S[-llr_polyscore(poly, points, centroid,
                             beta=beta, alfa=alfa/2)] = poly

    # print(bool(S))
    S = collections.OrderedDict(sorted(S.items()))
    K = next(iter(S))
    # print("key --", K)
    four_points = llr_normalize(S[K])

    # print("POINTS:", len(points))
    # print("LINES:", len(lines))

    return four_points


def llr_pad(four_points, img):
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(four_points, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)

    padded = pco.Execute(60)[0]

    # 60,70/75 is best (with buffer/for debug purpose)
    return pco.Execute(60)[0]
