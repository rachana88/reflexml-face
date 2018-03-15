import logging
import dlib
import numpy as np
from scipy import signal

from skimage.filters import scharr_h, scharr_v
from skimage import exposure

from ..utils.image import grab_rectangular_region, mean_kernel, resize

logger = logging.getLogger(__name__)


def relative_gaze_vector(reference, pupil):
    direction = np.array(pupil) - np.array(reference)
    direction = direction / np.linalg.norm(direction)
    return direction.tolist()


def find_pupil(img, rectangle=None, resize_width=30, equalize_img=False,
               normalize_gradient=True, convolve=False, gamma=0.5):
    """ Find a pupil center in a grayscale image. Formulates the problem as
    an optimzation problem over the eye region, where the theoretical optimum
    occurs at the center of the iris.

    Args:
    -----
        img: a 2D numpy array (i.e., a grayscale image)
        rectangle: if passed, look inside a region of the image for a pupil
        resize_width: int, how wide the search space should be for the 
            eye region. Note that this scales as O(resize_width ^ 2)
        equalize_img: boolean, do we want to equalize the histogram in the 
            eye region to make the gradients more defined  
        normalize_gradient: y/n to gradient normalization
        convolve: whether or not to take a 2D mean convolution over the 
            functional space before selecting the optimum (helps with noise)

    Returns:
    --------
        (pupil_loc_w, pupil_loc_h) w.r.t. the original image size and not 
        relative to the crop

    Raises:
    -------
        Exceptions
    """

    # keep track of all the dims
    dimensions = {}
    if rectangle is not None:
        assert isinstance(
            rectangle, dlib.rectangle), 'rectangle must be a dlib rectangle'
        logger.debug('will perform pupil search in constrained region')
        orig = grab_rectangular_region(img, rectangle)
        # stick the cropped eye region in img
        img, orig = orig, img
        dimensions['orig'] = orig.shape
        dimensions['rect'] = rectangle

        logger.debug('ingested image of size: {}'.format(orig.shape))
        logger.debug('searching in rectangle defined by {}'.format(str(rectangle)))

    if resize_width is not None:
        logger.debug('resizing search region to have width = {}'.format(resize_width))
        dimensions['pre_resize'] = img.shape
        img = resize(img, resize_width)

    if equalize_img:
        logger.debug('normalizing intensity histogram')

        img = exposure.equalize_hist(img)

    img = exposure.adjust_gamma(img, gamma)

    h, w = img.shape
    dimensions['img'] = img.shape

    # compute grads
    grad_x = scharr_v(img)
    grad_y = scharr_h(img)

    # get gradient vector for each pixel
    grad = np.array(zip(grad_y.ravel(), grad_x.ravel())).astype('float')

    if normalize_gradient:
        logger.debug('performing gradient normalization')

        # get gradient magnitude for each pixel
        grad_magnitude = np.linalg.norm(grad, axis=-1)
        mask = np.isnan(grad_magnitude)

        grad_mean = np.mean(grad_magnitude[~mask])
        grad_std = np.std(grad_magnitude[~mask])

        # make it one to divide easier
        grad_magnitude[mask] = 1
        # prune out noisy gradients
        grad_magnitude[grad_magnitude < (0.3 * grad_std + grad_mean)] = 1

        grad[grad_magnitude < 2] = 0

        # make the gradient vectors unit length
        # grad = grad / grad_magnitude[:, :, np.newaxis]
        grad = grad / grad_magnitude[:, np.newaxis]

    grad_coords = np.array([a for a in np.ndindex(h, w)]).astype('float')

    direction = (grad_coords[:, :, np.newaxis] - grad_coords.T)
    direction = direction / np.linalg.norm(direction, axis=1)[:, np.newaxis, :]

    acc_grad = (direction * grad[:, :, np.newaxis]).sum(axis=1)
    acc_grad[np.isnan(acc_grad) | (acc_grad < 0)] = 0
    acc_grad = acc_grad.sum(axis=0)

    if convolve:
        logger.debug('convolving final energy space')

        acc_grad = signal.convolve2d(
            acc_grad.reshape((h, w)), mean_kernel(), mode='same')

    # reverse so this is (w x h)
    pupil_loc_w, pupil_loc_h = grad_coords[
        acc_grad.argmax()].astype('int')[::-1]

    if resize_width is not None:

        h_old, w_old = dimensions['pre_resize']
        # get the upsample factor
        h_factor, w_factor = h_old / float(h), w_old / float(w)

        logger.debug('upsampling found pupil '
                     'with ({}, {}) factor'.format(h_factor, w_factor))

        # get the coords in the original system
        pupil_loc_w = int(pupil_loc_w * w_factor)
        pupil_loc_h = int(pupil_loc_h * h_factor)

    if rectangle is not None:
        logger.debug('converting to precropped coordinate system')

        pupil_loc_w += rectangle.left()
        pupil_loc_h += rectangle.top()

    logger.debug('pupil detection successful')

    return pupil_loc_w, pupil_loc_h
