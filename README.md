# ImageClassification
<h1>this project is built of some customized layers to be trained on image detection tasks based on the idea of Inception</h1>
<img src="/images/0_CAFo0A-z6w7xn8Le.jpeg">
<h1>Customized Layers</h1>
<p>This folder contains three custmized layers block and one miniInception model. The first block is the ConvModule. In this block there are two consecutive 
layers, the conv2D layer followed by batch normalization. The second block is the InceptionModule, which is composed of three parts layers. A Two ConvModule
block with 1x1 and 3x3 kernels respectively, followed by a concatination layer that concatinates the output of those two ConvModule blockss.
Finally the DownsampleModule that is also composed of three parts. A ConvModule of 3x3 kernel and 2x2 strides and a 3x3 maxpooling layer also 2x2 strided. 
the third component is a concatination layer that combines the output of the previous two components. 
There is also a MiniInceptionModule that is build up of the previous defined modules.
<h1>Train</h1>
The train folder contains a train class that is responsible of managing all the training process of the miniInceptionModule. all you have to do is to load 
your data and send it to this model with some additional attributes.
<h1>main</h1>
The main.py file tests all the previous code on cifar-10 data set.
