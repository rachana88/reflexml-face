

from glob import glob

flist = sorted(glob('./1/frames/*'))
tag_file = './1/26122013_223310_cam.tag'

image_nums = [(int(f.split('-')[-1].replace('.png', '')), f) for f in flist]

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


output = Parallel(n_jobs=20, verbose=50)(
    delayed(cv2.imread)(fp, cv2.IMREAD_GRAYSCALE) for fp in flist
)

tracked = track_face(output)

# cropped_output = Parallel(n_jobs=20, verbose=50)(
#     delayed(get_cropped_face)(_) for _ in output
# )


cropped_output = Parallel(n_jobs=20, verbose=50)(
    delayed(resize)(_, (128, 128)) for _ in tracked
)


landmarks = Parallel(n_jobs=20, verbose=50)(
    # delayed(get_eye_landmarks)(_) for _ in cropped_output
    delayed(get_eye_landmarks)(_) for _ in cropped_output
)


separations = np.array([(l['left_sep'], l['right_sep']) for l in landmarks])


feature = separations.mean(axis=-1)


windowed_features = np.array(window_select(feature, 5))
# the [1:] is because the track_face function drops the first frame
windowed_labels = window_select(labels[1:], 5)
windowed_labels = window_select(labels, 5)

y = 1 * np.array([('close' in l[3:-3]) or ('half' in l[3:-3])
                  for l in windowed_labels])


model = LogisticRegressionCV(Cs=10,
                             fit_intercept=True,
                             cv=None,
                             dual=False,
                             penalty='l2',
                             scoring=None,
                             solver='lbfgs',
                             tol=0.0001,
                             max_iter=100,
                             class_weight='balanced',
                             verbose=1,
                             refit=True,
                             intercept_scaling=1.0,
                             multi_class='ovr',
                             random_state=None)
