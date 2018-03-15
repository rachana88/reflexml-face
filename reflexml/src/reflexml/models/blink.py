#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: blink.py
description: file description
author: Luke de Oliveira (lukedeo@vaitech.io)
"""
import cPickle as pickle
import logging
import sys

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.preprocessing import PolynomialFeatures

from vaimodel.common import VaiBaseModel, register_model
from vaiutil.general import get_aware_filepath, enforce_return_type

from ..utils.general import window_select

logger = logging.getLogger('reflexml.models.blink')


@register_model('blinkfinder', '0.1.0')
class BlinkFinder(VaiBaseModel):

    """BlinkFinder assumes a 10 FPS framerate"""

    def __init__(self, estimating_object=None):
        super(BlinkFinder, self).__init__()
        self._model = estimating_object

    def save(self, prefix=None):
        outpath = self.savepath(prefix=prefix)
        logger.debug('assigned unique filepath: {}'.format(outpath))
        with get_aware_filepath(outpath, 'wb') as fileobj:
            pickle.dump((self._model, self._poly), fileobj)

    @classmethod
    @enforce_return_type
    def load(cls, prefix=None, specifier=None):
        filepath = cls.loadpath(prefix=prefix, specifier=specifier)
        logger.debug('retrieving from filepath: {}'.format(filepath))
        with get_aware_filepath(filepath, 'rb') as fileobj:
            model, poly = pickle.load(fileobj)
            o = cls(model)
            setattr(o, '_poly', poly)
            return o

    @staticmethod
    def _remove_samplewise_mean(X):
        return X - X.mean(axis=-1)[:, np.newaxis]

    def fit(self, X, y):
        self._poly = PolynomialFeatures(2)
        X = self._remove_samplewise_mean(
            self._poly.fit_transform(X.reshape(X.shape[0], -1))
        )
        self._model.fit(X, y)

        return self

    def fit_predict(self, X, y):
        self._poly = PolynomialFeatures(2)

        X = self._remove_samplewise_mean(
            self._poly.fit_transform(X.reshape(X.shape[0], -1))
        )

        return self._model.fit_predict(X, y)

    def predict(self, X):
        X = self._remove_samplewise_mean(
            self._poly.transform(X.reshape(X.shape[0], -1))
        )
        return self._model.predict(X)

    def predict_stream(self, stream):
        """ predict from a stream of palpebral fissure ratios for each eye

        Args:
        -----
            stream (np.array): a (nb_frames, 2) shaped array of contiguous, raw,
                palpebral fissure ratios

        Returns:
        --------
            Returns a nb_frames - ((2 * window_size) + 1) length list of predictions
        """

        X = np.array(window_select(stream, 6))
        X = self._remove_samplewise_mean(
            self._poly.transform(X.reshape(X.shape[0], -1))
        )

        y_prob = self._model.predict_proba(X)
        # y_label = self._model.predict(X)

        return np.convolve(y_prob[:, -1], [0.5, 0.5], mode='same')

        # return [
        #     {'label': l, 'probabilities': p.tolist()}
        #     for p, l in zip(y_prob, y_label)
        # ]


if __name__ == '__main__':

    interactive_logger = logging.getLogger('reflexml.models.blink.run')

    logcfg = [
        # ('', logging.INFO),
        ('reflexml', logging.DEBUG),
        ('vaiutil', logging.DEBUG),
        ('vaimodel', logging.DEBUG)
    ]

    for logname, level in logcfg:
        _logger = logging.getLogger(logname)
        _logger.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s(%(funcName)s)[pid=%(process)d]'
            ' - %(levelname)s: %(message)s'
        )
        hander = logging.StreamHandler(sys.stdout)
        hander.setFormatter(formatter)
        _logger.addHandler(hander)

    import argparse

    parser = argparse.ArgumentParser(description='blink training utility')

    parser.add_argument('--prefix', '-p', action="store",
                        help='prefix for saving', required=True)

    parser.add_argument('--data', '-d', action="store",
                        help='path to load a hdf5 file containing data', required=True)

    parser.add_argument('--features', '-x', action="store",
                        help='hdf5 path for features', default='X')

    parser.add_argument('--targets', '-y', action="store",
                        help='hdf5 path for targets', default='y')

    results = parser.parse_args()

    interactive_logger.info('Building model')

    clf = BlinkFinder(SGDClassifier(
        loss='modified_huber',
        verbose=100,
        class_weight='balanced',
        n_iter=50
    ))

    from h5py import File as HDF5File

    interactive_logger.info('Loading data, feature set: {}, '
                            'label set: {}'.format(results.features,
                                                   results.targets))

    df = HDF5File(results.data, 'r')

    X = df[results.features][:]
    y = df[results.targets][:]

    interactive_logger.info('Fitting')
    clf.fit(X, y)

    interactive_logger.info('serializing')
    clf.save(results.prefix)
