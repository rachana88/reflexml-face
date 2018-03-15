import logging
import os
import dlib

logger = logging.getLogger(__name__)

logger.debug('Loading Dlib models...')

FACE_LANDMARK_PATH = os.path.join(
    os.environ.get('DLIB_ROOT', '/opt/dlib/models'), 'shape_predictor_68_face_landmarks.dat')

logger.info('Will fetch facial landmark model '
            'from FACE_LANDMARK_PATH = {}'.format(FACE_LANDMARK_PATH))

FACIAL_KEYPOINT_MODEL = dlib.shape_predictor(FACE_LANDMARK_PATH)
FACIAL_CROP_MODEL = dlib.get_frontal_face_detector()
