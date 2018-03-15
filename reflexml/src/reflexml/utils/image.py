from __future__ import division

import numpy as np
import cv2
import dlib
from skimage import transform


def mean_kernel(shape=(3, 3)):
    return np.ones(shape) / float(np.prod(shape))


def grayscale(im):
    '''
    RGB --> grayscale
    '''
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def bgr2rgb(im):
    '''
    BGR --> RGB
    '''
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def rgb2bgr(im):
    '''
    RGB --> BGR
    '''
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def resize(img, new_width=500, backend='cv2'):
    assert backend in {'cv2', 'skimage'}
    if(len(img.shape) > 2):
        h, w, c = img.shape
    else:
        h, w = img.shape

    if isinstance(new_width, int):
        r = new_width / float(w)
        dim = (int(h * r), new_width)
    else:
        dim = new_width
    # img = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
    if backend == 'skimage':
        return transform.resize(img, dim)
    return cv2.resize(img, dim[::-1])


def grab_rectangular_region(img, r):
    x, y = r.top(), r.left()
    w, h = r.width(), r.height()
    return img[x:(x + h), y:(y + w)]


def center_of_mass(coords, weights=None):
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    if weights is not None:
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        weights /= weights.sum()
        return tuple(weights.dot(coords))
    return tuple(coords.mean(axis=0))


def bounding_rectangle(coords, horiz_scale=1.0, vert_scale=1.0):
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    r, b = coords.max(axis=0)
    l, t = coords.min(axis=0)

    dv = b - t
    dh = r - l

    dv = (vert_scale * dv - dv) / 2.
    dh = (horiz_scale * dh - dh) / 2.

    return dlib.rectangle(int(max(0, l - dh)), int(max(0, t - dv)), int(r + dh), int(b + dv))


def bounding_square(coords, horiz_scale=1.0, vert_scale=1.0):
    if not isinstance(coords, dlib.rectangle):
        rect = bounding_rectangle(coords, horiz_scale, vert_scale)
    else:
        rect = coords

    center = rect.center()

    c_x, c_y = center.x, center.y

    side_length = max(rect.width() / 2, rect.height() / 2)

    return dlib.rectangle(
        int(max(0, c_x - side_length)),
        int(max(0, c_y - side_length)),
        int(c_x + side_length),
        int(c_y + side_length)
    )
