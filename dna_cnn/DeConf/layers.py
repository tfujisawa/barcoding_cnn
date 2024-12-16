# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 06:58:18 2021

@author: IMAI Takashi
"""

import tensorflow as tf
from tensorflow.keras import activations, initializers
from tensorflow.keras.layers import (
    Layer,  # Base layers
    BatchNormalization,  # Normalization layers
)


class _Linear(Layer):

    def __init__(
        self,
        units,
        activation = None,
        use_bias = False,
        kernel_initializer = "glorot_uniform",
        bias_initializer = "zeros",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError(
                f"The last dimension of the input to \"{self.__class__.__name__}\" should be defined."
            )
        self.w = self.add_weight(
                name = "kernel",
                shape = (input_shape[-1], self.units),
                initializer = self.kernel_initializer,
                trainable = True
            )
        if self.use_bias:
            self.b = self.add_weight(
                    name = "bias",
                    shape = (self.units,),
                    initializer = self.bias_initializer,
                    trainable = True
                )
        super().build(input_shape)

    def call(self, inputs):
        outputs = tf.tensordot(inputs, self.w, axes = 1)
        if self.use_bias:
            outputs = tf.math.add(outputs, self.b)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)


class _CosineRepresentation(Layer):

    def __init__(
        self,
        class_num,
        kernel_initializer = "glorot_uniform",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.class_num = class_num
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError(
                f"The last dimension of the input to \"{self.__class__.__name__}\" should be defined."
            )
        self.w = self.add_weight(
                name = "kernel",
                shape = (input_shape[-1], self.class_num),
                initializer = self.kernel_initializer,
                trainable = True
            )
        super().build(input_shape)

    def call(self, inputs):
        return (
            tf.math.divide_no_nan(
                tf.tensordot(inputs, self.w, axes = 1),
                tf.tensordot(
                    tf.norm(inputs, axis = -1),
                    tf.norm(self.w, axis = 0),
                    axes = 0
                )
            )
        )

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.class_num,)


class DeConf(Layer):
    """
    Ref. [Hsu et al., Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2020)]
    """

    def __init__(
        self,
        h, class_num,
        use_batch_normalization = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.class_num = class_num
        self.h = h
        self.linear = _Linear(units = 1, use_bias = True)
        self.bn = (
            BatchNormalization() if use_batch_normalization
            else tf.keras.activations.linear
        )

    def _class_dependent_component(self, x):
        return self.h(x)

    def _class_independent_component(self, x):
        return tf.math.sigmoid(self.bn(self.linear(x)))

    def call(self, inputs):
        return (
            tf.nn.softmax(
                tf.math.divide_no_nan(
                    self._class_dependent_component(inputs),
                    self._class_independent_component(inputs)
                )
            )
        )

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.class_num,)

    def score(self, x): #X: outputs from the last layer in Tensor. Convert to numpy.array to use....
        return (
            tf.math.reduce_max(
                self._class_dependent_component(x),
                axis = -1
            )
        )

def DeConfC(
    class_num,
    use_batch_normalization = False,
    **kwargs
):
    return (
        DeConf(
            h = _CosineRepresentation(class_num = class_num),
            class_num = class_num,
            use_batch_normalization = use_batch_normalization,
            **kwargs
        )
    )
