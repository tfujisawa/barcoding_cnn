import sys
import numpy as np

# from Bio import AlignIO

import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Softmax
#from tensorflow.keras.layers import LayerNormalization
# from tensorflow.keras.layers import UnitNormalization
#from tensorflow.keras.layers import ActivityRegularization
from tensorflow.keras.layers import Concatenate

# import read_sq

#Generalized ODIN layers
from DeConf.layers2 import DeConf, _Linear, _CosineRepresentation


def initialize_dna_cnn_model(sqn_length, nclass, filt1=64, filt2=32, drconv=0.2, drfc=0.3, deconf_layer=False):
    input_shape = (sqn_length, 4)

    x_input = Input(shape=input_shape)
    y = Conv1D(filters=filt1, kernel_size=3, padding="same", strides=1, activation=None, input_shape=input_shape)(x_input)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Dropout(drconv)(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Conv1D(filters=filt2, kernel_size=3, padding="same", strides=1, activation=None)(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Dropout(drconv)(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Conv1D(filters=filt2, kernel_size=3, padding="same", strides=1, activation=None)(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Dropout(drconv)(y)
    y = MaxPooling1D(pool_size=2)(y)
    y_cnn = GlobalAveragePooling1D()(y)

    #Fully connected layers
    y = Dense(filt2, activation="relu")(y_cnn)
    y = Dropout(drfc)(y)
    y = Dense(filt2, activation="relu")(y)
    y = Dropout(drfc)(y)

    if deconf_layer:
        y = DeConf(h=_CosineRepresentation(nclass), use_batch_normalization=True, class_num=nclass)(y) #Cosine
    else:
        y_no_sm = Dense(nclass, activation=None)(y)
        y = Softmax()(y_no_sm)

    m = Model(x_input, y)
    return (m)

if __name__ == "__main__":

    m_d = initialize_dna_cnn_model(sqn_length=700, nclass=16, deconf_layer=True)
    print (m_d.summary())
