import sys 
sys.path.append('../')
sys.path.append('.')
from CustomLayers.MiniInception import MiniInception
from train.train import Train
import tensorflow as tf
import tensorflow.keras as krs

batch_size=64
(x_train,y_train),(x_test,y_test)=krs.datasets.cifar10.load_data()
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255

class_names=['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

y_train=krs.utils.to_categorical(y_train,num_classes=10)
y_test=krs.utils.to_categorical(y_test,num_classes=10)

train_dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset=train_dataset.shuffle(buffer_size=1024).batch(batch_size)


test_dataset=tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_dataset=test_dataset.batch(batch_size)

model=MiniInception(10)

lossFn=krs.losses.CategoricalCrossentropy()
optimizer=krs.optimizers.Adam()

train_acc_metric=krs.metrics.CategoricalAccuracy()
test_acc_metric=krs.metrics.CategoricalAccuracy()

train_writer=tf.summary.create_file_writer('logs/train/')
test_writer= tf.summary.create_file_writer('logs/test/')


train=Train(model,train_dataset,test_dataset,lossFn,optimizer,train_acc_metric,test_acc_metric,train_writer,test_writer)

train.train(epochs=2)


