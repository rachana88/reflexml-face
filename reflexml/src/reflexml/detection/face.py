from __future__ import division

import os
import base64
import itertools
import logging

import numpy as np
import dlib
import cv2

from joblib import Parallel, delayed

from ..utils.image import  grayscale, bgr2rgb, rgb2bgr, \
    grab_rectangular_region, bounding_rectangle, bounding_square

from ..detection.eye import find_pupil

from ..models.dlib_core import FACIAL_KEYPOINT_MODEL, FACIAL_CROP_MODEL

logger = logging.getLogger(__name__)

# slices from the cascading random forest facial keypoints...

JAW_SLICE = slice(0, 17)
R_EYEBROW_SLICE = slice(17, 22)
L_EYEBROW_SLICE = slice(22, 27)
NOSE_SLICE = slice(27, 31)
NOSE_BOTTOM_SLICE = slice(31, 36)
R_EYE_SLICE = slice(36, 42)
L_EYE_SLICE = slice(42, 48)
MOUTH_SLICE = slice(48, 69)


class FaceDetectionError(RuntimeError):
    pass


def crop_face(im, upsample=0, as_list=True):
    """ Crops the face out of an image using dlib

    Args:
    -----
        im: an **RGB** image!
        upsample: the number of times to upsample

    Returns:
    --------
        the bounding rectangle, (y, x, w, h)
    """
    try:
        rects = FACIAL_CROP_MODEL(im, upsample)
    except RuntimeError, e:
        raise FaceDetectionError('Error in image format passed')

    if len(rects) != 1:
        raise FaceDetectionError(
            'Found {} faces, expected 1'.format(len(rects)))

    r = rects[0]

    l, t, w, h = r.left(), r.top(), r.width(), r.height()

    if l < 0:
        w += l
        l = 0
    if t < 0:
        h += t
        t = 0
    if as_list:
        return [l, t, w, h]

    return dlib.rectangle(l, t, l + w, t + h)


def get_cropped_face(f):
    try:
        return grab_rectangular_region(f, crop_face(f, as_list=False))
    except FaceDetectionError:
        return f


def track_face(frames):
    """ Takes an iterator of frames and yields crops of the face

    Args:
    -----
        frames (iter): an iterator of numpy arrays (video frames)

    """

    tracker = dlib.correlation_tracker()
    for i, f in enumerate(frames):
        logger.debug('Tracking frame #{}'.format(i))
        if i == 0:
            l, t, w, h = crop_face(f)
            tracker.start_track(f, dlib.rectangle(l, t, l + w, t + h))
        else:
            tracker.update(f)
        yield grab_rectangular_region(f, tracker.get_position())


def chunked_face_tracker(frames, chunk_size):
    for ix, chunk in grouper(chunk_size, frames):
        logger.info('Launching chunk: {}'.format(ix))

    processed = Parallel(n_jobs=-1, verbose=100, backend='threading')(
        delayed(track_face_eyes)(chunk) for chunk in grouper(chunk_size, frames)
    )
    return processed


def ingest_image(buf, postprocess=False, silence_errors=False):
    """ Ingests a face and returns the blink distance as well 
    as whether or not a blink was detected

    Args:
    -----
        buf: a 2D grayscale image as a numpy array
    """
    try:
        landmarks = get_eye_landmarks(buf, precropped=False)
    except FaceDetectionError, err:
        if silence_errors:
            logger.warn('Failing silently')
            return {'success': False}
        raise err

    bounding_box = landmarks['bounding']
    l_eye, r_eye = landmarks['left_eye'], landmarks['right_eye']

    # def _find_pupils(img, eye):
    #     rect = bounding_rectangle(eye, vert_scale=2, horiz_scale=1.2)

    #     return find_pupil(img, rect, resize_width=25, equalize_img=True,
    #                       normalize_gradient=False, convolve=False)

    # l_pupil, r_pupil = _find_pupils(buf, l_eye), _find_pupils(buf, r_eye)

    logger.debug('Fetching blink distances')
    blink_distance = np.mean((landmarks['left_sep'], landmarks['right_sep']))

    payload = {
        'success': True,
        'blink_distance': blink_distance,
        'left_sep': landmarks['left_sep'],
        'right_sep': landmarks['right_sep'],
        # 'left_pupil': l_pupil,
        # 'right_pupil': r_pupil,
        'bounding_box': bounding_box,
        'left_eye': l_eye,
        'right_eye': r_eye,
        'keypoints': landmarks['facial_keypoints']
    }

    if postprocess is not False:
        assert hasattr(postprocess, '__call__')
        payload['postprocess'] = postprocess(buf, payload)
    return payload


def track_face_eyes(frames):
    face_tracker = dlib.correlation_tracker()
    left_eye_tracker = dlib.correlation_tracker()
    right_eye_tracker = dlib.correlation_tracker()

    res = []

    for i, f in enumerate(frames):
        logger.info('Tracking frame #{}'.format(i))
        if i == 0:
            l, t, w, h = crop_face(f)
            r = dlib.rectangle(l, t, l + w, t + h)
            face_tracker.start_track(f, r)

            keypoints = get_eye_landmarks(f, precropped=r)
            left_eye = bounding_rectangle(
                keypoints['left_eye'], horiz_scale=1.5, vert_scale=3)
            right_eye = bounding_rectangle(
                keypoints['right_eye'], horiz_scale=1.5, vert_scale=3)

            right_eye_tracker.start_track(f, right_eye)
            left_eye_tracker.start_track(f, left_eye)

        else:
            face_tracker.update(f)
            right_eye_tracker.update(f)
            left_eye_tracker.update(f)

        face_rect = face_tracker.get_position()
        face_rect = dlib.rectangle(int(face_rect.left()), int(face_rect.top()),
                                   int(face_rect.right()), int(face_rect.bottom()))

        left_eye_rect = right_eye_tracker.get_position()
        left_eye_rect = dlib.rectangle(int(left_eye_rect.left()),
                                       int(left_eye_rect.top()),
                                       int(left_eye_rect.right()),
                                       int(left_eye_rect.bottom()))

        right_eye_rect = left_eye_tracker.get_position()
        right_eye_rect = dlib.rectangle(int(right_eye_rect.left()),
                                        int(right_eye_rect.top()),
                                        int(right_eye_rect.right()),
                                        int(right_eye_rect.bottom()))

        # gr = grayscale(f)
        pupils = (_find_pupils(f, left_eye_rect),
                  _find_pupils(f, right_eye_rect))

        res.append((f, face_rect, left_eye_rect, right_eye_rect, pupils))
        # yield f, face_rect, left_eye_rect, right_eye_rect, pupils
        # yield f, r, left_eye, right_eye, l_pupil, r_pupil, keypoints


def get_eye_landmarks(im, precropped=True):
    """ ingest a grayscale image

    Args:
    -----
        im (2D np.array): a 2D, uint8 image
        precropped (bool | dlib.rectangle): whether or not the image has been 
            precropped, or, a dlib.rectable representing the exact facial bounding box to crop

    Returns:
    --------
        A dictionary with the following key paths:

        left_eye
        right_eye
        nose_tip
        bounding
        right_sep
        left_sep
        palpebral_fissure_ratio
        facial_keypoints/
            jaw_slice
            r_eyebrow_slice
            l_eyebrow_slice
            nose_slice
            nose_bottom_slice
            r_eye_slice
            l_eye_slice
            mouth_slice
    """

    if not precropped:
        rect = crop_face(im)
        (x, y, w, h) = rect
        r = dlib.rectangle(x, y, x + w, y + h)
    # case where the face is already cropped!
    elif precropped is True:
        rect = [0, 0, im.shape[1], im.shape[0]]
        r = dlib.rectangle(0, 0, im.shape[1], im.shape[0])
    else:
        r = precropped

    rect = [int(r.left()), int(r.top()), int(r.right()), int(r.bottom())]

    logger.debug('Fetching facial keypoints...')

    keypoints = [
        (p.x, p.y)
        for p in FACIAL_KEYPOINT_MODEL(im.astype('uint8'), r).parts()
    ]

    r_dist = eyelid_distance(keypoints[R_EYE_SLICE])
    l_dist = eyelid_distance(keypoints[L_EYE_SLICE])

    return {
        'left_eye': keypoints[L_EYE_SLICE],
        'right_eye': keypoints[R_EYE_SLICE],
        'nose_tip':  keypoints[NOSE_SLICE][-1],
        'bounding': rect,
        'right_sep': r_dist,
        'left_sep': l_dist,
        'palpebral_fissure_ratio': (r_dist + l_dist) / 4,
        'facial_keypoints': {
            'jaw_slice': keypoints[JAW_SLICE],
            'r_eyebrow_slice': keypoints[R_EYEBROW_SLICE],
            'l_eyebrow_slice': keypoints[L_EYEBROW_SLICE],
            'nose_slice': keypoints[NOSE_SLICE],
            'nose_bottom_slice': keypoints[NOSE_BOTTOM_SLICE],
            'r_eye_slice': keypoints[R_EYE_SLICE],
            'l_eye_slice': keypoints[L_EYE_SLICE],
            'mouth_slice': keypoints[MOUTH_SLICE]
        }
    }


def head_orientation(nose, left_eye, right_eye):
    nose_bridge = nose[0]
    nose_tip = nose[-1]

    nose_vec = np.array(nose_bridge) - np.array(nose_tip)

    def _point_distance(line, point):  # p3 is the point
        (x0, y0), (x1, y1) = line
        x2, y2 = point
        numer = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denom = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        result = numer / max(denom, 0.0001)
        return result

    left_eye_m = map(lambda (x, y): (x + y) / 2.,
                     zip(left_eye[0], left_eye[3]))
    right_eye_m = map(lambda (x, y): (x + y) / 2.,
                      zip(right_eye[0], right_eye[3]))

    left_dist = _point_distance(
        (nose_bridge, nose_tip), left_eye_m) / np.linalg.norm(nose_vec)
    right_dist = _point_distance(
        (nose_bridge, nose_tip), right_eye_m) / np.linalg.norm(nose_vec)

    return right_dist - left_dist


def eyelid_distance(eye):
    """ Returns the eyelid distance for a given eye

    Args:
    -----
        eye: a list of 6 tuples (x, y) representing the 
        keypoints from the ocular region

    Returns:
    --------
        the norm of vector length of the midpoint of the top and bottom eyelid

    Keypoints:
    ----------

    0, 3 are corners
    1, 2 are the top
    4, 5 are the bottom

    """

    left_corner, right_corner = map(np.array, (eye[0], eye[3]))
    top_left, top_right = map(np.array, (eye[1], eye[2]))
    bottom_right, bottom_left = map(np.array, (eye[4], eye[5]))

    denominator = 2 * np.linalg.norm(right_corner - left_corner)

    numerator = np.linalg.norm(top_left - bottom_left) + \
        np.linalg.norm(top_right - bottom_right)

    return numerator / denominator

    # top = np.mean(eye[1:3], axis=0)
    # bottom = np.mean(eye[4:], axis=0)

    # left, right = map(np.array, (eye[0], eye[3]))

    # return np.linalg.norm(top - bottom) / max(0.001, np.linalg.norm(right -
    # left))


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='run the face cropping service.')

    parser.add_argument('--aug', '-a', action="store", required=False,
                        help='Makes new file for each ingested /path/to/file.jpg '
                        'called /path/to/file{aug}.jpg', default='-processed')

    parser.add_argument('--quiet', '-q', action='store_true',
                        help='silence logging')

    parser.add_argument('images', help='images to ingest', nargs='+')

    results = parser.parse_args()

    if not results.quiet:
        level = logging.INFO
        stream = sys.stderr

        logger.setLevel(level)

        formatter = logging.Formatter(
            '[pid=%(process)d] %(asctime)s'
            ' - '
            '%(name)s(%(funcName)s)'
            ' - '
            '%(levelname)s: %(message)s'
        )

        console = logging.StreamHandler(stream)
        console.setFormatter(formatter)

        logger.addHandler(console)

    def outfmt(fname, aug):
        base, ext = os.path.splitext(fname)
        return base + aug + ext

    outfiles = [outfmt(im, results.aug) for im in sorted(results.images)]
    # outfiles = [outfmt(im, results.aug) for im in sorted(results.images)]
    frames = (cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
              for inpath in sorted(results.images))

    def grab_select(im, parms):
        crop_rects = {
            'left_eye': bounding_square(parms['left_eye'], horiz_scale=2.5, vert_scale=2.5),
            'right_eye': bounding_square(parms['right_eye'], horiz_scale=2.5, vert_scale=2.5),
            'bounding_box': dlib.rectangle(*parms['bounding_box'])
        }

        crops = {
            k: grab_rectangular_region(im, v)
            for k, v in crop_rects.iteritems()
        }

        return crops

    analyzed = Parallel(n_jobs=30, verbose=20)(
        delayed(ingest_image)(im, grab_select, True) for im in frames
    )

    for image, fp in zip(analyzed, outfiles):
        if image is not False:
            logger.info('writing to {}'.format(fp.replace(
                results.aug, results.aug + 'bounding_box')))
            cv2.imwrite(fp.replace(results.aug, results.aug +
                                   'bounding_box'), image['postprocess']['bounding_box'])

            for e in ['left_eye', 'right_eye']:
                logger.info('writing to {}'.format(
                    fp.replace(results.aug, results.aug + e)))
                cv2.imwrite(fp.replace(results.aug, results.aug + e),
                            image['postprocess'][e])
        else:
            logger.warn('Skipping write out to {}'.format(fp))
