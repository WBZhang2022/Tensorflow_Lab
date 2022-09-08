'''
    author:WBZhang
    contents:A simple CNN
    Date:08/16/2022
'''

import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten,Dense
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train, x_test = x_train/255., x_test/255.


class CNNCifar10(Model):
    def __init__(self):
        super(CNNCifar10, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')


    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y


model = CNNCifar10()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)


checkpoint_save_path = './checkpoint/cifar10CNN.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('--------------load the model ----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True
)


history = model.fit(x_train,y_train,
                    batch_size=32,epochs=5,
                    validation_data=(x_test,y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])

model.summary()

file = open('./model/weights.txt','w')

for v in model.trainable_variables:
    file.write(str(v.name)+'\n')
    file.write(str(v.shape)+'\n')
    file.write(str(v.numpy())+'\n')

file.close()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


plt.subplot(1,2,1)
plt.plot(acc,label='Training Accurancy')
plt.plot(val_acc,label='Validation Accurancy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label='Training Loss')
plt.plot(val_loss,label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

