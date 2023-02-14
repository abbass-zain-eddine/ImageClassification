import sys
sys.path.append('../')
import tensorflow.keras as krs
from tensorflow.keras import layers
from  CustomLayers.ConvModule import ConvModule


class DownsamplingModule(layers.Layer):
    def __init__(self,kernel_num, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv1= ConvModule(kernel_num,kernel_size=(3,3),strides=(2,2),padding='valid')
        self.pool=layers.MaxPool2D(pool_size=(3,3),strides=(2,2))
        self.cat=layers.Concatenate()

    def call(self, inputs, training=False,**kwargs):
        x_conv=self.conv1(inputs,training=training)
        x_pool=self.pool(inputs)
        x= self.cat([x_conv,x_pool])
        return x