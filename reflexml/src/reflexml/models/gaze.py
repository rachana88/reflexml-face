import logging

import numpy as np

from ..deepmodels import columbia_net

logger = logging.getLogger(__name__)


class GazeNet(object):
    """ Wrapper and nice interface around a three input, three headed deep net
    for identifying head pose, horizontal eye gaze, and vertical eye gaze

    Args:
    -----
        filename: a path to a h5 file of a network trained with the columbia_net
            architecture

    __call__ invokes GazeNet.predict. Consult the docstring for GazeNet.predict 
    for details

    """

    def __init__(self, filename):
        super(GazeNet, self).__init__()

        # TODO: no flexibility of network arch!!!
        self._net = columbia_net()
        self._net.load_weights(filename)

        # the pose / gaze classes that are available are from the actual
        # dataset
        self._classes = {
            'horiz': {
                0: -15, 1: -10, 2: -5, 3: 0, 4: 5, 5: 10, 6: 15
            },
            'pose': {
                0: -30, 1: -15, 2: 0, 3: 15, 4: 30
            },
            'vertical': {
                0: -10, 1: 0, 2: 10
            }
        }

        # fixed shape for now
        self._shape = (64, 64)

    def predict(self, face, left_eye, right_eye, *args, **kwargs):
        """ Given a face crop and two eye crops, determine gaze and pose from
        the given training

        NOTE: all images are expected to be scaled [0, 1]

        Args:
        -----
            face: a (nb_frames, 1, 64, 64) shaped crop of faces, grayscale

            left_eye: a (nb_frames, 1, 64, 64) shaped crop of left eye 
                region, grayscale

            right_eye: a (nb_frames, 1, 64, 64) shaped crop of right eye 
                region, grayscale

        Returns:
        --------
            dictionary with keys = {'pose', 'vertical', and 'horiz'} representing
            the pose angles (in degrees) from straight ahead the desired metric
            represents per frame

        Raises:
        -------
            ValueError if passed shapes of arrays do not match specifications
        """

        for sh in [face.shape, left_eye.shape, right_eye.shape]:
            if len(sh) != 4 or sh[1] != 1:
                raise ValueError('all passed arrays must be '
                                 'of shape (nb_samples, 1, nb_x, nb_y)')
            if sh[-2:] != self._shape:
                raise ValueError('Model needs (64 x 64) images')

        payload = [face, left_eye, right_eye]

        pose, vertical, horiz = map(
            lambda y: y.argmax(axis=-1), self._net.predict(payload))

        pose = [self._classes['pose'][code] for code in pose]
        vertical = [self._classes['vertical'][code] for code in vertical]
        horiz = [self._classes['horiz'][code] for code in horiz]

        return {
            'pose': pose,
            'vertical': vertical,
            'horiz': horiz
        }

    def __call__(self, face, left_eye, right_eye):
        return self.predict(face, left_eye, right_eye)
