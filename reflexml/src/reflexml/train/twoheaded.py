import logging
import os

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
else:
    logger = logging.getLogger(__name__)

def log(msg):
    logger.info(LOGGER_PREFIX % msg)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Train the pose model')

    parser.add_argument('--train-dir', action="store", required=True,
                        help='root training dir', dest='train_dir')

    parser.add_argument('--test-dir', action="store", required=True,
                        help='root testing dir', dest='test_dir')

    parser.add_argument('--save', action="store", required=True,
                        help='final model path')

    parser.add_argument('--vgg-weights', action="store", required=False,
                        help='initial VGG weights', dest='vgg_weights', default='models/vgg16_weights.h5')

    parser.add_argument('--chkpt', action="store", required=False, default='vgg-chkpt.h5',
                        help='checkpoint model path')

    parser.add_argument('--save-classes', action="store", required=False, dest='save_classes',
                        help='whether or not to save the output class indices to a file')



    results = parser.parse_args()

    log('Training VGG pretrained (domain adaptation) model.')

    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    from ..deepmodels import pretrained_vgg
    
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            channel_shift_range=0.1)


    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
            results.train_dir,  # this is the target directory
            target_size=(64, 64),
            batch_size=32,
            # since we use categorical_crossentropy loss, we need categorical labels
            class_mode='categorical')  


    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            results.test_dir,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')


    nb_classes = len(train_generator.class_indices)

    log('found {} classes'.format(nb_classes))

    model = pretrained_vgg(shape=(64, 64), nb_classes=nb_classes, weights_path=results.vgg_weights)

    
    callbacks = [
        EarlyStopping(verbose=True, patience=10, monitor='val_acc'),
        ModelCheckpoint(results.chkpt, monitor='val_acc', 
                        verbose=True, save_best_only=True),
    ]

    log('Commencing fit'.format(nb_classes))

    try:
        model.fit_generator(
            train_generator,
            samples_per_epoch=train_generator.N,
            nb_epoch=50,
            validation_data=validation_generator,
            callbacks=callbacks,
            nb_val_samples=900
        )
    except KeyboardInterrupt, k:
        log('Stopped early.')

    model.load_weights(results.chkpt)

    log('saving model to {}'.format(results.save))
    model.save_weights(results.save, overwrite=True)

    if results.save_classes is not None:
        from cPickle import dump
        log('saving class indices to {}'.format(results.save_classes))
        with open(results.save_classes, 'w') as f:
            dump(train_generator.class_indices, f)






