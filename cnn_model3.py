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
from tensorflow.keras.layers import LayerNormalization
# from tensorflow.keras.layers import UnitNormalization
from tensorflow.keras.layers import ActivityRegularization
from tensorflow.keras.layers import Concatenate

# import read_sq

#Generalized ODIN layers
from DeConf.layers2 import DeConf, _Linear, _CosineRepresentation

#Scale Normalization
from scale_norm import ScaledL2Normalization
from scale_norm import L2Normalization

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

def initialize_multiinput_model(sqn_length, input_length, nclass, deconf_layer=False):
    input_shape1 = (sqn_length, 4)

    x_input = Input(shape=input_shape1)
    y = Conv1D(filters=64, kernel_size=3, padding="same", strides=1, activation=None, input_shape=input_shape1)(x_input)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Dropout(0.2)(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Conv1D(filters=32, kernel_size=3, padding="same", strides=1, activation=None)(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    y = Dropout(0.2)(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Conv1D(filters=32, kernel_size=3, padding="same", strides=1, activation=None)(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Dropout(0.2)(y)
    y = MaxPooling1D(pool_size=2)(y)
    y_cnn = GlobalAveragePooling1D()(y)
    y_cnn = L2Normalization()(y_cnn)
    # y_cnn = UnitNormalization()(y)
    #y_cnn = LayerNormalization()(y_cnn)
   #  y_cnn = BatchNormalization()(y_cnn)

    input_shape2 = (input_length,)
    x2_input = Input(shape=input_shape2)
    #y2 = ActivityRegularization(l1=0., l2=1.)(x2_input)
    y2 = L2Normalization()(x2_input)
    #y2 = ScaledL2Normalization()(x2_input)
    #y2 = LayerNormalization()(x2_input)
    # y2 = Dense(4, activation="relu")(x2_input)
    #y2 = BatchNormalization()(x2_input)
    y2 = ActivityRegularization(l1=.0, l2=1.)(y2)

    conc = Concatenate(axis=1)([y_cnn,y2])
    #conc = Concatenate(axis=1)([y_cnn,x2_input])
    y = Dense(512, activation="relu")(conc)
    y = Dropout(0.3)(y)
    y = Dense(256, activation="relu")(y)
    y = Dropout(0.3)(y)

    if deconf_layer:
        y = DeConf(h=_CosineRepresentation(nclass), use_batch_normalization=True, class_num=nclass)(y) #Cosine
    else:
        y_no_sm = Dense(nclass, activation=None)(y)
        y = Softmax()(y_no_sm)

    m = Model([x_input, x2_input], y)
    return (m)

if __name__ == "__main__":
    # m = initialize_dna_cnn_model(sqn_length=700, nclass=16)
    # print (m.summary())
    #
    # m2 = initialize_multiinput_model(sqn_length=700, input_length=2, nclass=16)
    # print (m2.summary())

    m_d = initialize_dna_cnn_model(sqn_length=700, nclass=16, deconf_layer=True)
    print (m_d.summary())
