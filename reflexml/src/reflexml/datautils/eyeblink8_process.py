from glob import glob
import numpy as np

from skimage import exposure
import cv2

from reflexml.detection.face import track_face, get_eye_landmarks, ingest_image
from reflexml.utils.image import resize
from reflexml.utils.general import window_select

from joblib import Parallel, delayed


def job(_):
    # r = resize(_, (128, 128))
    # return get_eye_landmarks(r), r
    # im = (exposure.equalize_adapthist(img, clip_limit=0.04) * 255).astype('uint8')
    return get_eye_landmarks(_)


def load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    return (exposure.equalize_adapthist(img, clip_limit=0.04) * 255).astype('uint8')


def get_eyeblink_dir(dirname, raw=False):
    assert int(dirname) in range(1, 12)

    flist_orig = sorted(glob('./{}/frames/*'.format(dirname)))
    flist = np.array(3 * [flist_orig]).T.reshape(-1, 1).ravel().tolist()
    tag_file = glob('./{}/*.tag'.format(dirname))[0]

    image_nums = [(int(f.split('-')[-1].replace('.png', '')), f)
                  for f in flist]

    image_nums = [(i + 1, f) for i, f in enumerate(flist)]

    tags = dict(map(
        lambda line:
        (lambda tup: (int(tup[0]), tup[1]))(
            tuple(line.strip().split('\t'))
        ),
        open(tag_file).readlines()
    ))

    labels = np.array([tags.get(fn, 'open') for fn, _ in image_nums[::3]])

    labels[(labels == 'closeRight') | (labels == 'closeRight')] = 'close'
    labels[(labels == 'halfRight') | (labels == 'halfRight')] = 'half'

    output = Parallel(n_jobs=8, verbose=50)(
        # delayed(cv2.imread)(fp, cv2.IMREAD_GRAYSCALE) for fp in flist
        delayed(load_img)(fp) for fp in flist_orig
    )

    # tracked = track_face(output)

    landmarks = Parallel(n_jobs=8, verbose=50)(
        delayed(ingest_image)(im, False, True) for im in output
    )

    # landmarks = Parallel(n_jobs=8, verbose=50)(
    #     delayed(job)(_) for _ in tracked
    # )

    # landmarks, frames = zip(*Parallel(n_jobs=8, verbose=50)(
    #     # delayed(get_eye_landmarks)(_) for _ in cropped_output
    #     delayed(job)(_) for _ in tracked
    # ))

    # landmarks_failed = [_ is False for _ in landmarks]

    separations = np.array([
        (l.get('left_sep', -1), l.get('right_sep', -1))
        for l in landmarks
    ])

    # return windowed_features, y

    # feature = separations.mean(axis=-1)

    if raw:
        return separations, labels

    windowed_features = np.array(window_select(separations, 6))
    # windowed_frames = np.array(window_select(frames, 6))
    # the [1:] is because the track_face function drops the first frame
    # windowed_labels = window_select(labels[1:], 5)
    windowed_labels = window_select(labels, 6)

    # return windowed_features, y

    y = 1 * np.array([('close' in l[3:-3]) or ('half' in l[3:-3])
                      for l in windowed_labels])
    # y = []
    # for l in windowed_labels:
    #     if 'close' in l[3:-3]:
    #         y.append('close')
    #     elif 'half' in l[3:-3]:
    #         y.append('half')
    #     else:
    #         y.append('open')

    # y = 1 * np.array([('close' in l[3:-3]) or ('half' in l[3:-3])
    #                   for l in windowed_labels])

    return windowed_features, y


dirs = [1,  10,  11,  2,  3,  4,  8,  9]


def get_eyeblink_dir_features(dirname):
    assert int(dirname) in range(1, 12)

    flist = sorted(glob('./{}/frames/*'.format(dirname)))
    tag_file = glob('./{}/*.tag'.format(dirname))[0]

    image_nums = [(int(f.split('-')[-1].replace('.png', '')), f)
                  for f in flist]

    tags = dict(map(
        lambda line:
        (lambda tup: (int(tup[0]), tup[1]))(
            tuple(line.strip().split('\t'))
        ),
        open(tag_file).readlines()
    ))

    labels = np.array([tags.get(fn, 'open') for fn, _ in image_nums])

    labels[(labels == 'closeRight') | (labels == 'closeRight')] = 'close'
    labels[(labels == 'halfRight') | (labels == 'halfRight')] = 'half'

    output = Parallel(n_jobs=8, verbose=50)(
        delayed(cv2.imread)(fp, cv2.IMREAD_GRAYSCALE) for fp in flist
    )

    tracked = track_face(output)

    cropped_output = Parallel(n_jobs=8, verbose=50)(
        delayed(resize)(_, (128, 128)) for _ in tracked
    )

    landmarks = Parallel(n_jobs=8, verbose=50)(
        delayed(get_eye_landmarks)(_) for _ in cropped_output
    )

    # landmarks_failed = [_ is False for _ in landmarks]

    separations = np.array([(l['left_sep'], l['right_sep'])
                            for l in landmarks])

    feature = separations.mean(axis=-1)

    return cropped_output, feature, labels
    # the [1:] is because the track_face function drops the first frame
    # windowed_labels = window_select(labels[1:], 5)
    # windowed_labels = window_select(labels, 6)

    y = 1 * np.array([('close' in l[3:-3]) or ('half' in l[3:-3])
                      for l in windowed_labels])

    # return windowed_features, y


def get_eyeblink_raw(dirname):
    assert int(dirname) in range(1, 12)

    flist = sorted(glob('./{}/frames/*'.format(dirname)))
    tag_file = glob('./{}/*.tag'.format(dirname))[0]

    image_nums = [(int(f.split('-')[-1].replace('.png', '')), f)
                  for f in flist]

    tags = dict(map(
        lambda line:
        (lambda tup: (int(tup[0]), tup[1]))(
            tuple(line.strip().split('\t'))
        ),
        open(tag_file).readlines()
    ))

    labels = np.array([tags.get(fn, 'open') for fn, _ in image_nums])

    labels[(labels == 'closeRight') | (labels == 'closeRight')] = 'close'
    labels[(labels == 'halfRight') | (labels == 'halfRight')] = 'close'
    labels[(labels == 'half')] = 'close'

    labels = 1 * (labels == 'close')

    output = Parallel(n_jobs=8, verbose=50)(
        delayed(cv2.imread)(fp, cv2.IMREAD_GRAYSCALE) for fp in flist
    )

    tracked = track_face(output)

    # cropped_output = Parallel(n_jobs=8, verbose=50)(
    #     delayed(resize)(_, (128, 128)) for _ in tracked
    # )

    landmarks, frames = zip(*Parallel(n_jobs=8, verbose=50)(
        # delayed(get_eye_landmarks)(_) for _ in cropped_output
        delayed(job)(_) for _ in tracked
    ))

    # landmarks_failed = [_ is False for _ in landmarks]

    separations = np.array([(l['left_sep'], l['right_sep'])
                            for l in landmarks])

    feature = separations.mean(axis=-1)

    windowed_features = np.array(window_select(feature, 6))
    windowed_frames = np.array(window_select(frames, 6))
    # the [1:] is because the track_face function drops the first frame
    # windowed_labels = window_select(labels[1:], 5)
    windowed_labels = window_select(labels, 6)

    y = 1 * np.array([('close' in l[3:-3]) or ('half' in l[3:-3])
                      for l in windowed_labels])

    return windowed_frames, windowed_features, y


data = [get_eyeblink_dir(i) for i in dirs]

d = zip(*data)

X_windowed = np.concatenate(d[0])
y_windowed = np.concatenate(d[1])

keep = (X_windowed == -1).sum((1, 2)) == 0
X_windowed = X_windowed[keep]
y_windowed = y_windowed[keep]

ix = range(y_windowed.shape[0])
np.random.shuffle(ix)

X_windowed, y_windowed = X_windowed[ix], y_windowed[ix]

import h5py
fp = h5py.File('./eb8.h5', 'w')

fp['X'] = X_windowed
fp['y'] = y_windowed

fp.close()

# model = LogisticRegressionCV(Cs=10,
#                              fit_intercept=True,
#                              cv=None,
#                              dual=False,
#                              penalty='l2',
#                              scoring=None,
#                              solver='lbfgs',
#                              tol=0.0001,
#                              max_iter=100,
#                              class_weight='balanced',
#                              verbose=1,
#                              refit=True,
#                              intercept_scaling=1.0,
#                              multi_class='ovr',
#                              random_state=None)
