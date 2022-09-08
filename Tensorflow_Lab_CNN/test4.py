'''
    author:WBZhang
    contents:InceptionNet
    Date:08/17/2022
'''

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,\
                                    MaxPool2D,GlobalMaxPool2D,Dense
import tensorflow as tf
np.set_printoptions(threshold=np.inf)


cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class ConvBNRelu(Model):
    def __init__(self,channels,kernel_size=3,strides=1,padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(channels,kernel_size,strides=strides,padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self,x):
        x = self.model(x)
        return x




class InceptionBlock(Model):
    def __init__(self,channels,strides=1):
        super(InceptionBlock, self).__init__()
        self.channels = channels
        self.strides = strides

        ## 分支1
        self.c1 = ConvBNRelu(channels,kernel_size=1,strides=strides)

        ## 分支2
        self.c2_1 = ConvBNRelu(channels,kernel_size=1,strides=strides)
        self.c2_2 = ConvBNRelu(channels,kernel_size=3,strides=1)

        ## 分支3
        self.c3_1 = ConvBNRelu(channels,kernel_size=1,strides=strides)
        self.c3_2 = ConvBNRelu(channels,kernel_size=5,strides=1)

        ## 分支4
        self.p4_1 = MaxPool2D(3,strides=1,padding='same')
        self.c4_2 = ConvBNRelu(channels,kernel_size=1,strides=strides)

    def call(self,x):
        x1 = self.c1(x)

        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)

        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)

        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)

        # 将四个分支按照深度的顺序拼接起来
        x = tf.concat([x1,x2_2,x3_2,x4_2],axis=3)
        return x


class InceptionNet(Model):
    def __init__(self,num_blocks,num_classes,init_channels=16,**kwargs):
        super(InceptionNet, self).__init__(**kwargs)
        self.in_channels = init_channels
        self.out_channels = init_channels
        self.num_blocks = num_blocks
        self.init_channels = init_channels

        self.c1 = ConvBNRelu(init_channels)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlock(self.out_channels, strides=2)
                else:
                    block = InceptionBlock(self.out_channels, strides=1)
                self.blocks.add(block)
            # 经过一个Inception Block 输出通道数会翻4倍
            self.out_channels *= 2 # 出于计算效率每次乘2，即在二进制中向左移一位
        self.p1 = GlobalMaxPool2D()
        self.f1 = Dense(num_classes,activation='softmax')


    def call(self,x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y

model = InceptionNet(num_blocks=2,num_classes=10)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/cifar10InceptionNet.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()


file = open('./model/weights_InceptionNet.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()



acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()