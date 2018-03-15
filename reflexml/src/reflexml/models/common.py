import logging
import os
import dlib

from .gaze import GazeNet
from .blink import BlinkNet

logger = logging.getLogger(__name__)

logger.debug('Loading models...')

GAZENET_MODEL = GazeNet(os.environ.get(
    'GAZENET', '/Users/lukedeoliveira/vizzario/python/reflexml/ColumbiaNetAug.h5'))

BLINKNET_MODEL = BlinkNet(os.environ.get(
    'BLINKNET', '/Users/lukedeoliveira/vizzario/python/reflexml/BlinkDNN.h5'))

FACE_LANDMARK_PATH = os.path.join(
    os.environ.get('DLIB_ROOT', '.'), 'shape_predictor_68_face_landmarks.dat')

logger.info('Will fetch facial landmark model '
            'from FACE_LANDMARK_PATH = {}'.format(FACE_LANDMARK_PATH))

FACIAL_KEYPOINT_MODEL = dlib.shape_predictor(FACE_LANDMARK_PATH)
FACIAL_CROP_MODEL = dlib.get_frontal_face_detector()
logger.debug('Done')
