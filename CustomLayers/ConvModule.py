import tensorflow.keras as krs
from tensorflow.keras import layers


class ConvModule(layers.Layer):
    def __init__(self,kernel_num,kernel_size,strides,padding='same', trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        #conv layer
        self.conv=layers.Conv2D(
            kernel_num,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding
        )
        #batch norm layer
        self.bn = layers.BatchNormalization()


    def call(self, inputs,training=False, **kwargs):
        x=self.conv(inputs)
        x=self.bn(x,training=training)
        x=krs.activations.relu(x)
        return x

