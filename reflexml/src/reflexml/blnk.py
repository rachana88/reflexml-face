from keras.layers import Convolution2D, TimeDistributed, Flatten, Dropout, Dense, Input
from keras.models import Model

NB_ROWS = 64
NB_COLS = 64
WINDOW_SIZE = 6
NB_CHANNELS = 1

# image model only...
im = Input(shape=(NB_CHANNELS, NB_ROWS, NB_COLS))


x = Input(
    shape=(2 * WINDOW_SIZE + 1, NB_CHANNELS, NB_ROWS, NB_COLS),
    dtype='float32'
)

spatio_temp_proj = TimeDistributed(Convolution2D(
    64, 3, 3, border_mode='same', activation='relu'
))

o = spatio_temp_proj(x)
