import os
import cPickle as pickle
import cv2
import dlib
import numpy as np
import base64
import re

import logging

from joblib import Parallel, delayed

from ..detection.face import get_eye_landmarks, eyelid_distance, ingest_image
from ..detection.eye import find_pupil, relative_gaze_vector
from ..utils.io import im_to_b64, im_from_b64, load_img_as_base64
from ..utils.image import bounding_rectangle, grayscale, center_of_mass, grab_rectangular_region, bounding_square

# from ..models import GAZENET_MODEL, BLINKNET_MODEL
# from ..models import BLINKFINDER'

from skimage.color import rgb2gray
from skimage import exposure

# build logger
logger = logging.getLogger(__name__)


def movingaverage(x, window):
    """
    Calculate moving average across windows of size window_size
    using convolutions.
    """
    if isinstance(window, int):
        window = np.ones(int(window)) / float(window)
    return np.convolve(x, window, mode='same')


def greedy_contraction(a):
    """
    Perform greedy contraction across a 1D array.
    """
    currval = a[0]
    ret = [currval]
    i = 1
    while i < len(a):
        if np.abs(a[i] - currval) > 0.5:
            ret.append(a[i])
            currval = a[i]
        i += 1
    return ret


def blink_info(dists, convolved=False):
    d = dists[dists > 0.0001]

    if not len(d):
        return {
            'nb_blinks': 0,
            'nb_dropped_frames': 0
        }
    if convolved:
        s, e = d[0], d[-1]
        d = movingaverage(d, 2)
        d[0], d[-1] = s, e
    thresh = np.percentile(d, 5)
    x = 1 * (d < thresh)
    return {
        'nb_blinks': np.sum(greedy_contraction(x)),
        'nb_dropped_frames': len(dists) - len(d)
    }


def replace_invalid(l, b, replace_with=None, hook=None):
    """ Expects two lists of equal length

    Args:
    -----
        l: a list of things
        b: a list of bools, where we replace if the bool is true
        replace_with: the thing to replace stuff with
    """
    if hook is not None:
        hook.send([(el if not cond else replace_with)
                   for el, cond in zip(l, b)])
        return

    if replace_with is None:
        return [el for el, cond in zip(l, b) if not cond]
    return [(el if not cond else replace_with) for el, cond in zip(l, b)]


def crop_callback(im, parms):
    """ A callback, used to ensure that eye / face crops are returned from
    reflexml.detection.face.ingest_image

    Args:
    -----
        im: a 2D array (grayscale image)

    Returns:
    --------
        a dictionary with keys 'left_eye', 'right_eye', 'face', 
        and the corresponding crop out of @a im
    """

    # get the eye and face crop dimensions
    crop_rects = {
        'left_eye': bounding_square(parms['left_eye'],
                                    horiz_scale=2.5, vert_scale=2.5),
        'right_eye': bounding_square(parms['right_eye'],
                                     horiz_scale=2.5, vert_scale=2.5),
        # the bounding box comes as [left, right, top, bottom], so we need to
        # convert that into a dlib.rectangle for the bounding_square method to
        # be valid
        'face': bounding_square(dlib.rectangle(*parms['bounding_box'])),
    }

    # slice the crops on the im
    crops = {
        k: grab_rectangular_region(im, v)
        for k, v in crop_rects.iteritems()
    }

    return crops


def ingest_frames(frames, n_jobs=50, backend='multiprocessing'):
    logger.info('Ingested {} frames, processing in {} jobs'.format(
        len(frames), n_jobs))
    logger.debug('Launching jobs...')

    # do the ingestion in parallel to speed things up a bit
    # the callback will yank out the crops needed to run the
    # deep nets.
    blinks = Parallel(n_jobs=n_jobs, verbose=11, backend=backend)(
        # postprocess=False, silence_errors=False
        delayed(ingest_image)(im, crop_callback, True) for im in frames
    )
    nb_frames = len(blinks)

    logger.debug('Collected {} completed jobs'.format(n_jobs))
    logger.debug('Collecting recorded blink distances')

    blink_dist = np.array([float(blink.get('blink_distance', 0)) for blink in blinks])
    # failures = map(lambda x: x < 0.0001, blink_dist)
    failures = map(lambda f: not f['success'], blinks)

    if sum(failures) > 0:
        logger.warn('Found {} face detection failures. '
                    'Will smooth out estimation'.format(sum(failures)))
        _replace_invalid = replace_invalid
    else:
        logger.debug('No failures detected!')

        # if we dont have any failures, we dont have to do replacement
        def _f(l, b, *args, **kwargs):
            # though we want to ensure that we dont have a generator hanging
            # out
            return list(l)

        _replace_invalid = _f

    blinks = _replace_invalid(blinks, failures)

    # TODO: no need for pupils for the demo, but will need later
    # r_pupils = [blink['right_pupil'] for blink in blinks]
    # l_pupils = [blink['left_pupil'] for blink in blinks]

    r_palp_sep = [blink['right_sep'] for blink in blinks]
    l_palp_sep = [blink['left_sep'] for blink in blinks]

    # r_pupils = _replace_invalid(
    #     ([blink['right_pupil'] for blink in blinks]), failures)
    # l_pupils = _replace_invalid(
    #     ([blink['left_pupil'] for blink in blinks]), failures)

    logger.debug('Building payload')
    payload = {
        'failures': sum(failures),
        'palpebral': {
            'left': l_palp_sep,
            'right': r_palp_sep
        },
        'blink': {
            'blink_info': blink_info(blink_dist),
            'blink_info_conv': blink_info(blink_dist, True),
            'blink_distances': _replace_invalid(blink_dist, failures),
            'nb_frames': nb_frames
        },
        # 'pupils': {
        #     'left': l_pupils,
        #     'right': r_pupils
        # }
    }

    # logger.debug('Batching images for conv. net.')

    # crop_names = ['right_eye', 'left_eye', 'face']

    # image_crops = {
    #     crop: np.array([
    #         # resize the crop to 64x64, then normalize the intensity histogram,
    #         # then add a new axis and scale the pixel intensities to be
    #         # in [0, 1]
    #         cv2.equalizeHist(cv2.resize(
    #             # crops are contained in the 'postprocess' callback
    #             b['postprocess'][crop],
    #             (64, 64)
    #         ))[np.newaxis, :, :] / 255.
    #         for b in blinks
    #     ])
    #     for crop in crop_names  # get each crop
    # }

    # logger.debug('Feeding through conv net...')
    # the kwargs in the function map perfecty to the keys in `image_crops`
    # above
    # blink_proba = BLINKNET_MODEL(**image_crops)
    # gaze_info = GAZENET_MODEL(**image_crops)

    # payload['blink']['blink_info_dl'] = blink_info(
    #     blink_proba, convolved=True)

    # logger.debug('Conv net successful')

    # put the pose (i.e., head rotation) and gaze (h and v) information
    # seperate to make analysis easier

    # payload['pose'] = gaze_info['pose']

    # payload['gaze'] = {
    #     'horizontal': gaze_info['horiz'],
    #     'vertical': gaze_info['vertical']
    # }

    logger.debug('Full analysis successful')
    return payload
