from glob import glob
import logging
import random
import os

import cv2
import numpy as np
from skimage import exposure
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import random_rotation
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ColumbiaIterator(object):
    """ Utility iterator to train models on the Columbia dataset. 
    Please consult [1] for details on dataset construction. This iterator assumes
    you have unzipped the downloaded dataset, and run:

    ubuntu@host$ python -m reflexml.detection.face --root-dir=/path/to/columbia_gaze

    such that under /path/to/columbia_gaze, the folders 00*/ exist, with images
    inside.

    This also assumes you didn't pass the --aug flag to reflexml.detection.face


    Args:
    -----
        root: the path to /path/to/columbia_gaze


    [1] http://www.cs.columbia.edu/CAVE/databases/columbia_gaze/
    """

    def __init__(self, root, batch_size=32, size=(64, 64), train_frac=0.6):
        self.root = root
        self.batch_size = batch_size
        self.size = size

        f = [
            sorted(glob(os.path.join(root, '*', '*{}*.jpg'.format(c))))
            for c in ['bounding_box', 'left_eye', 'right_eye']
        ]

        allowed_prefixes = set(map(lambda _: _.split('-proc')[0], f[0]))

        for ch in f[1:]:
            allowed_prefixes = allowed_prefixes & set(
                map(lambda _: _.split('-proc')[0], ch))

        f = [[fp for fp in ch if fp.split(
            '-proc')[0] in allowed_prefixes] for ch in f]

        self.train_files = zip(*f)
        self.train_files, self.test_files = train_test_split(
            self.train_files, train_size=train_frac)

        self.pose_classes, self.vertical_classes, self.horiz_classes = [
            {
                i: int(k)
                for i, k in enumerate(
                    sorted(list(set(
                        np.array([
                            self._process_filename(_[0], radians=False)
                            for _ in self.train_files
                        ])[:, pr]
                    )))
                )
            }
            for pr in xrange(3)
        ]

        def _(d):
            return {v: k for k, v in d.iteritems()}

        self.pose_classes_inv, \
            self.vertical_classes_inv, \
            self.horiz_classes_inv = map(_,
                                         (self.pose_classes,
                                          self.vertical_classes,
                                          self.horiz_classes)
                                         )

    def nb_train():
        def fget(self):
            return len(self.train_files)
        return locals()
    nb_train = property(**nb_train())

    def nb_test():
        def fget(self):
            return len(self.train_files)
        return locals()
    nb_test = property(**nb_test())

    @staticmethod
    def _process_filename(fp, radians=True):
        """
        the original format of the columbia dataset is
        {N}/{N}_2m_{X}P_{Y}V_{Z}H*.jpg

            * N is the subject ID
            * X is the pose
            * Y is the vertical gaze angle
            * X is the horizontal gaze angle

        optionally, convert the angles to radians, yay
        """
        return map(
            lambda _: (
                lambda th: th if not radians else np.deg2rad(th)
            )(float(_[:-1])),
            fp.split('_2m_')[-1].split('-proc')[0].split('_')
        )

    @staticmethod
    def _load_image(im, size, rotate=None):
        """
        load an image from a file, resize it to `size`, then rotate if passed.
        Then, returns 1/255., such that the pixels are in [0, 1]
        """
        g = cv2.imread(im, cv2.IMREAD_GRAYSCALE)

        # also, equalize the histogram to make training easier
        # TODO: make this optional
        resized = cv2.resize(cv2.equalizeHist(g), size)[np.newaxis, :, :]
        if rotate is not None:
            resized = random_rotation(resized, rotate)
        return resized / 255.

    def flow(self, mode='train', n=None, rotate=None):
        """ Flow a generator from the root directory of the dataset

        Args:
        -----
            mode: one of {train, test}
            n: the number of samples to flow (if n is None), will flow infinitely
                at random

        Returns:
        --------
            Yields tuples of the form:

                (
                    (face_img, left_eye_img, right_eye_img), 
                    (pose_label, vertical_label, horiz_label)
                )
        """

        assert mode in {'train', 'test'}

        files = self.train_files if mode == 'train' else self.test_files

        iters = 0
        while True:
            iters += 1
            if n is not None and iters > n:
                break

            batch_files = zip(*random.sample(files, self.batch_size))

            inputs = [
                np.array([
                    self._load_image(im, self.size, rotate)
                    for im in stream
                ]).astype('float32')
                for stream in batch_files
            ]

            labels = np.array([
                self._process_filename(fp, radians=False)
                for fp in batch_files[0]
            ]).astype('int')

            pose_labels = [self.pose_classes_inv[_] for _ in labels[:, 0]]
            vertical_labels = [self.vertical_classes_inv[_]
                               for _ in labels[:, 1]]
            horiz_labels = [self.horiz_classes_inv[_] for _ in labels[:, 2]]

            pose_labels = np.reshape(pose_labels, (-1, 1))
            vertical_labels = np.reshape(vertical_labels, (-1, 1))
            horiz_labels = np.reshape(horiz_labels, (-1, 1))

            yield (inputs, [pose_labels, vertical_labels, horiz_labels])
