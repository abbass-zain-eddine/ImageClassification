import sys
sys.path.append('../')
sys.path.append('.')

from CustomLayers.MiniInception import MiniInception
import tensorflow.keras as krs
import tensorflow as tf
import time


class Train:
    def __init__(self,model:krs.models.Model,trainDs:tf.data.Dataset,valDs:tf.data.Dataset,lossFn:krs.losses,optimizer:krs.optimizers,train_acc_metric:krs.metrics,test_acc_metric:krs.metrics,train_writer:tf.summary,test_writer:tf.summary) -> None:
        self.model=model
        self.trainDs=trainDs
        self.valDs=valDs
        self.lossFn=lossFn
        self.optimizer=optimizer
        self.train_acc_metric=train_acc_metric
        self.test_acc_metric=test_acc_metric
        self.train_writer=train_writer#.creat_file_writer('logs/train/')
        self.test_writer=test_writer#.creat_file_writer('logs/train/')
    
    @tf.function
    def train_step(self,step,x_batch_train,y_batch_train):
        with tf.GradientTape() as tape:
            logits=self.model(x_batch_train,training=True)
            train_loss_value=self.lossFn(y_batch_train,logits)
        
        grads=tape.gradient(train_loss_value,self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_weights))
        self.train_acc_metric.update_state(y_batch_train,logits)
        
        #write train loss and accuracy to the tensorboard
        with self.train_writer.as_default():
            tf.summary.scalar('loss',train_loss_value,step=step)
            tf.summary.scalar('accuracy',self.train_acc_metric.result(),step=step)

        return train_loss_value

    @tf.function
    def test_step(self,step,x_batch_test,y_batch_test):
        val_logits=self.model(x_batch_test,training=False)
        test_loss_value=self.lossFn(y_batch_test,val_logits)
        self.test_acc_metric.update_state(y_batch_test,val_logits)
        

        #write test loss and accuracy to the tensorboard
        with self.test_writer.as_default():
            tf.summary.scalar('loss',test_loss_value,step=step)
            tf.summary.scalar('accuracy',self.test_acc_metric.result(),step=step)

        return test_loss_value
    
    def train(self,epochs):
        template='ETA: {} - epoch: {} loss: {} acc: {} val loss:{} val acc:{}\n'
        for epoch in range(epochs):
            t=time.time()
            for step, (x_batch_train,y_batch_train) in enumerate(self.trainDs):
                step=tf.convert_to_tensor(step,dtype=tf.int64)
                train_loss_val=self.train_step(step,x_batch_train,y_batch_train)
            
            for step, (x_batch_test,y_batch_test) in enumerate(self.trainDs):
                step=tf.convert_to_tensor(step,dtype=tf.int64)
                test_loss_val=self.test_step(step,x_batch_test,y_batch_test)


            #verbose
            print(template.format(
                    round((time.time() - t )/60 ,2 ),epoch+1,
                    train_loss_val,float(self.train_acc_metric.result()),
                    test_loss_val,float(self.test_acc_metric.result())
            ))   
            #reset metrics at the end of each epoch        
            self.train_acc_metric.reset_states()
            self.test_acc_metric.reset_states()
                
            
if __name__ == "__main__":
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


