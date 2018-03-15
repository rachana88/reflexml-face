import logging
import os

if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
else:
    logger = logging.getLogger(__name__)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Train the columbia model')

    parser.add_argument('--root-dir', action="store", required=True,
                        help='root training dir', dest='root_dir')

    parser.add_argument('--save', action="store", required=True,
                        help='final model path')

    parser.add_argument('--chkpt', action="store", required=False, default='reflex-chkpt.h5',
                        help='checkpoint model path')

    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--rotate', '-r', action='store', type=int)

    results = parser.parse_args()

    if results.simple:
        logging.basicConfig(level=logging.INFO)
    else:
        import sys
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s(%(funcName)s)[pid=%(process)d]'
            ' - %(levelname)s: %(message)s'
        )
        hander = logging.StreamHandler(sys.stdout)
        hander.setFormatter(formatter)
        logger.addHandler(hander)

    logger.info('Training reflexnet Columbia model.')

    from keras.callbacks import EarlyStopping, ModelCheckpoint

    from ..deepmodels import columbia_net

    from ..datautils.columbia import ColumbiaIterator

    image_size = (64, 64)

    ci = ColumbiaIterator(results.root_dir, batch_size=32)

    train_generator = ci.flow('train', rotate=results.rotate)
    validation_generator = ci.flow('test', rotate=results.rotate)

    nb_targets = 3

    logger.info('found {} targets'.format(nb_targets))

    model = columbia_net(
        shape=image_size, nb_channels=1)

    callbacks = [
        EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
        ModelCheckpoint(results.chkpt, monitor='val_loss',
                        verbose=True, save_best_only=True)
    ]

    logger.info('Commencing fit'.format(nb_targets))

    try:
        model.fit_generator(
            train_generator,
            samples_per_epoch=ci.nb_train,
            nb_epoch=100,
            validation_data=validation_generator,
            callbacks=callbacks,
            nb_val_samples=ci.nb_test
        )
    except KeyboardInterrupt, k:
        logger.info('Stopped early.')

    model.load_weights(results.chkpt)

    logger.info('saving model to {}'.format(results.save))
    model.save_weights(results.save, overwrite=True)
