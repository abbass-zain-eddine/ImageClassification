import sys 
sys.path.append('../')
import tensorflow.keras as krs
from tensorflow.keras import layers
from CustomLayers.ConvModule import ConvModule
class InceptionModule(layers.Layer):
    def __init__(self,kernel_num1x1,kernel_num3x3, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv1x1=ConvModule(kernel_num1x1,kernel_size=(1,1),strides=(1,1))
        self.conv3x3=ConvModule(kernel_num3x3,kernel_size=(3,3),strides=(1,1))
        self.cat=layers.Concatenate()

    def call(self, inputs,training=False, **kwargs):
        x_1x1=self.conv1x1(inputs,training)
        x_3x3=self.conv3x3(inputs,training)
        x=self.cat([x_1x1,x_3x3])
        return x
