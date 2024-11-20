import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers

class ScaledL2Normalization(Layer):
    """
    Ref. [Nguyen and Salazar, International Workshop on Spoken Language Translation (2019)]
    """

    def __init__(self, axis = -1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.scale = (
            self.add_weight(
                name = "global_scale",
                initializer = (
                    initializers.Constant(
                        tf.math.sqrt(tf.cast(feature_dim, tf.float32))
                    )
                ),
                trainable = True
            )
        )
        super().build(input_shape)

    def call(self, inputs):
        return (
            tf.math.scalar_mul(
                self.scale,
                tf.math.l2_normalize(inputs, axis = self.axis)
            )
        )

    def compute_output_shape(self, input_shape):
        return input_shape

class L2Normalization(Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return (tf.math.l2_normalize(inputs, axis=self.axis))
