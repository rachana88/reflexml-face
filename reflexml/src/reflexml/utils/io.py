#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
file: io.py
description: utilities related to image format I/O in the ReflexML ecosystem
author: Luke de Oliveira (lukedeo@vizzario.com)
"""
from __future__ import division

import base64
import tarfile
import StringIO
import os
import zipfile

import cv2
import numpy as np


def im_to_b64(im, order='rgb'):
    assert order in {'rgb', 'bgr'}
    if order == 'rgb':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    buf = cv2.imencode('.jpg', im)
    return base64.encodestring(buf[1].tostring())


def im_from_b64(buf, order='rgb'):
    """ 
    returns a cv2 image from a base64 encoded jpg
    """
    assert order in {'rgb', 'bgr'}
    im = cv2.imdecode(
        np.fromstring(
            base64.b64decode(buf),
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )
    if order == 'rgb':
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def im_to_binary(im, order='rgb'):
    assert order in {'rgb', 'bgr'}
    if order == 'rgb':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    buf = cv2.imencode('.jpg', im)
    return buf[1].tostring()


def im_from_binary(buf, order='rgb'):
    """ returns a cv2 image from a binary stream
    """
    assert order in {'rgb', 'bgr', 'gray'}
    im = cv2.imdecode(
        np.fromstring(
            buf,
            np.uint8
        ),
        cv2.IMREAD_COLOR if order != 'gray' else cv2.IMREAD_GRAYSCALE
    )
    if order == 'rgb':
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def load_img_as_base64(impath):
    with open(impath, 'rb') as f:
        return base64.b64encode(f.read())


def compressed_image_iterator(buf, ext={'.jpg', '.png'}, grayscale=True):
    """
    Args:
    -----
        buf: either a basestring object that is a read-in zip or gz file or an 
            actual stream or filelike object to be buffered

    Example:
    --------

    >>> buf = open('./CameraSample.zip').read()
    >>> images = map(my_etl, compressed_image_iterator(buf))
    >>> images = list(compressed_image_iterator(open('./CameraSample.zip')))
    """
    # if not isinstance(buf, basestring):
    #     raise TypeError('compressed_image_iterator relies on a binary stream '
    #                     'that is an instance of a basestring')

    if isinstance(buf, basestring):
        buf = StringIO.StringIO(buf)

    if zipfile.is_zipfile(buf):
        open_file = lambda buf: zipfile.ZipFile(buf)
        get_files = lambda h: h.namelist()
        read_buf = lambda h, p: h.read(p)
    else:
        open_file = lambda buf: tarfile.open(
            mode='r:gz',
            fileobj=buf
        )
        get_files = lambda h: h.getnames()
        read_buf = lambda h, p: h.extractfile(p).read()

    with open_file(buf) as handler:
        files = (fp for fp in sorted(get_files(handler))
                 if os.path.splitext(fp)[-1] in ext)
        for fp in files:
            yield im_from_binary(
                read_buf(handler, fp), 'gray' if grayscale else 'rgb')
