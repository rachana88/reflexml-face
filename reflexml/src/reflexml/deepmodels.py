import logging
import os

import cv2
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, merge, Input, Highway

logger = logging.getLogger(__name__)


def columbia_net(shape=(64, 64), nb_channels=1):

    logger.info(
        'generating net with input shape ({})'.format(', '.join(str(s) for s in shape)))

    img_width, img_height = shape

    nb_poses = 5
    nb_vertical = 3
    nb_horiz = 7

    face = Input(shape=(nb_channels, img_width, img_height))
    left_eye = Input(shape=(nb_channels, img_width, img_height))
    right_eye = Input(shape=(nb_channels, img_width, img_height))

    face_model = Sequential()
    face_model.add(Flatten(input_shape=(nb_channels, img_width, img_height)))
    face_model.add(Dense(1024, activation='relu'))
    face_model.add(Dropout(0.25))
    face_model.add(Highway(activation='relu'))

    face_model.add(Dropout(0.25))
    face_model.add(Highway(activation='relu'))

    face_model.add(Dropout(0.25))
    face_model.add(Highway(activation='relu'))

    face_model.add(Dense(512, activation='relu'))

    face_h = face_model(face)

    eye_model = Sequential()
    eye_model.add(Flatten(input_shape=(nb_channels, img_width, img_height)))
    eye_model.add(Dense(1024, activation='relu'))
    eye_model.add(Dropout(0.25))
    eye_model.add(Highway(activation='relu'))

    eye_model.add(Dropout(0.25))
    eye_model.add(Highway(activation='relu'))

    eye_model.add(Dropout(0.25))
    eye_model.add(Highway(activation='relu'))

    eye_model.add(Dense(512, activation='relu'))

    # eye_model.add(Flatten())

    left_eye_h = eye_model(left_eye)
    right_eye_h = eye_model(right_eye)

    # combined = merge([face_h, left_eye_h, right_eye_h], mode='concat', concat_axis=1)
    eyes = merge([left_eye_h, right_eye_h], mode='sum')
    combined = merge([face_h, eyes], mode='concat', concat_axis=1)

    h = Dense(128)(combined)
    h = Activation('relu')(h)
    h = Dropout(0.2)(h)
    out_pose = Dense(nb_poses, activation='softmax', name='pose')(h)

    h = Dense(128)(combined)
    h = Activation('relu')(h)
    h = Dropout(0.2)(h)
    out_vertical = Dense(nb_vertical, activation='softmax', name='vertical')(h)

    h = Dense(128)(combined)
    h = Activation('relu')(h)
    h = Dropout(0.2)(h)
    out_horiz = Dense(nb_horiz, activation='softmax', name='horizontal')(h)

    model = Model(
        input=[face, left_eye, right_eye],
        output=[out_pose, out_vertical, out_horiz]
    )

    logger.info('compiling with Adam and mse')
    model.compile(
        'adam', 3 * ['sparse_categorical_crossentropy'], metrics=['acc'])

    return model


def blink_net(shape=(64, 64), nb_channels=1):

    logger.info(
        'generating net with input shape ({})'.format(', '.join(str(s) for s in shape)))

    img_width, img_height = shape

    eye = Input(shape=(nb_channels, img_width, img_height))

    eye_model = Sequential()
    eye_model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu',
                                input_shape=(nb_channels, img_width, img_height)))
    eye_model.add(Dropout(0.25))
    eye_model.add(Convolution2D(
        32, 3, 3, border_mode='valid', activation='relu'))
    eye_model.add(Dropout(0.25))
    eye_model.add(Flatten(input_shape=(nb_channels, img_width, img_height)))
    eye_model.add(Dense(1024, activation='relu'))
    eye_model.add(Dropout(0.25))
    eye_model.add(Highway(activation='relu'))

    eye_model.add(Dropout(0.25))
    eye_model.add(Highway(activation='relu'))

    eye_model.add(Dropout(0.25))
    eye_model.add(Highway(activation='relu'))

    eye_model.add(Dense(512, activation='relu'))
    eye_model.add(Dropout(0.2))

    eye_model.add(Dense(128, activation='relu'))
    eye_model.add(Dropout(0.2))

    eye_model.add(Dense(2, activation='softmax', name='pose'))

    logger.info('compiling with Adam and mse')
    eye_model.compile(
        'adam', 'categorical_crossentropy', metrics=['acc'])

    return eye_model


def reflexnet(shape=(64, 64), nb_classes=2, nb_channels=3):

    logger.info('generating reflexnet net with input shape'' ({}) and {} output '
                'classes'.format(', '.join(str(s) for s in shape), nb_classes))

    img_width, img_height = shape

    model = Sequential()
    model.add(Convolution2D(128, 3, 3, border_mode='same',
                            input_shape=(nb_channels, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    logger.info('compiling with Adam and categorical_crossentropy')
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    return model


def pretrained_vgg(shape=(64, 64), nb_classes=2,
                   weights_path='models/vgg16_weights.h5', last_conv_block=True, nb_channels=3):

    logger.info('generating VGG16 net with input shape'' ({}) and {} output '
                'classes'.format(', '.join(str(s) for s in shape), nb_classes))
    img_width, img_height = shape

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(
        nb_channels, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu',
                            name='conv1_1', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu',
                            name='conv1_2', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',
                            name='conv2_1', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu',
                            name='conv2_2', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',
                            name='conv3_1', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',
                            name='conv3_2', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu',
                            name='conv3_3', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',
                            name='conv4_1', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',
                            name='conv4_2', trainable=False))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu',
                            name='conv4_3', trainable=False))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    if last_conv_block:
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu',
                                name='conv5_1', trainable=False))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu',
                                name='conv5_2', trainable=False))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu',
                                name='conv5_3', trainable=False))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)

    if weights_path is not None:

        logger.info('loading weights from {}'.format(weights_path))

        assert os.path.exists(
            weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the
                # savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)]
                       for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        logger.info('Model loaded.')

        logger.info('Adding layers to fine tune')

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    logger.info('compiling with Adam and categorical_crossentropy')
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    return model


def two_headed(shape=(64, 20), nb_classes=2):

    raise NotImplementedError('two_headed model not implemented yet')
