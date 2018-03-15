from __future__ import division

import os
import base64
import itertools
import logging

import numpy as np
import dlib
import cv2

from joblib import Parallel, delayed

from ..utils import  grayscale, bgr2rgb, rgb2bgr, \
    grab_rectangular_region, bounding_rectangle, bounding_square

from ..models.dlib import FACIAL_KEYPOINT_MODEL, FACIAL_CROP_MODEL

if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
else:
    logger = logging.getLogger(__name__)


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='run the detection / cropping service.')

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
        delayed(ingest_image)(im, grab_select, True) for im in frames)

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
