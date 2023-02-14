import sys
sys.path.append('../')
from CustomLayers.InceptionModule import InceptionModule
from CustomLayers.ConvModule import ConvModule
from CustomLayers.DownsamplingModule import DownsamplingModule
import tensorflow.keras as krs
from tensorflow.keras.models import Model

class MiniInception(Model):
    def __init__(self,num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_bloc=ConvModule(96,(3,3),(1,1))

        self.inception_block1=InceptionModule(32,32)
        self.inception_block2=InceptionModule(32,48)
        self.downsample_block1=DownsamplingModule(80)

        self.inception_block3=InceptionModule(112,48)
        self.inception_block4=InceptionModule(96,64)
        self.inception_block5=InceptionModule(80,80)
        self.inception_block6=InceptionModule(48,96)
        self.downsample_block2=DownsamplingModule(96)

        self.inception_block7=InceptionModule(17,160)
        self.inception_block8=InceptionModule(176,160)

        self.avg_pool=krs.layers.AveragePooling2D((7,7))

        self.flat=krs.layers.Flatten()
        self.classifier=krs.layers.Dense(
            num_classes,activation='softmax'
        )

    def call(self, inputs, training=None, mask=None):
        x=self.conv_bloc(inputs)
        x=self.inception_block1(x)
        x=self.inception_block2(x)
        x=self.downsample_block1(x)
        x=self.inception_block3(x)
        x=self.inception_block4(x)
        x=self.inception_block5(x)
        x=self.inception_block6(x)
        x=self.downsample_block2(x)
        x=self.inception_block7(x)
        x=self.inception_block8(x)
        x=self.avg_pool(x)

        x=self.flat(x)
        return self.classifier(x)


    def build_graph(self,raw_shape):
        x=krs.layers.Input(shape=raw_shape)
        return Model(inputs=[x],outputs=self.call(x))




