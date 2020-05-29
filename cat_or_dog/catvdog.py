import numpy as np
import sys
import matplotlib as plt
import os
import tensorflow as tf
import h5py
import pickle
from PIL import Image
print("imported packages")
def define_model() :
    Conv2D1=tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
                                   padding='same', input_shape=(200, 200, 3))
    Conv2D2=tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',
                                   padding='same', input_shape=(200, 200, 3))
    Conv2D3=tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',
                                   padding='same', input_shape=(200, 200, 3))
    Dense1=tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')
    Dense2=tf.keras.layers.Dense(1, activation='sigmoid')
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((200, 200, 3), input_shape=(40000, 3)))
    model.add(Conv2D1)
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Conv2D2)
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Conv2D3)
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(Dense1)
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(Dense2)

    opt = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
DATADIR = '/Users/johnathanxie/Documents/Python/datasets/dogs-vs-cats'
CATEGORIES = ["Dog", "Cat"]

x_train_file=open('/Users/johnathanxie/Documents/Python/datasets/dogs-vs-cats/x_train', 'rb')
y_train_file=open('/Users/johnathanxie/Documents/Python/datasets/dogs-vs-cats/y_train', 'rb')
x_val_file=open('/Users/johnathanxie/Documents/Python/datasets/dogs-vs-cats/x_val', 'rb')
y_val_file=open('/Users/johnathanxie/Documents/Python/datasets/dogs-vs-cats/y_val', 'rb')

x_train=pickle.load(x_train_file)
y_train=pickle.load(y_train_file)
x_val=pickle.load(x_val_file)
y_val=pickle.load(y_val_file)
print("data recieved")
img0=Image.new("RGB", (200, 200))
pixels=img0.load()
for i in range(200):
    for j in range(200):
        pixels[i, j]=tuple(x_train[0, i, j])
img0.show()
x_train, x_val=x_train/255, x_val/255
x_train_file.close()
y_train_file.close()
x_val_file.close()
y_val_file.close()
model=define_model()
model.load_weights('./catvdog_weights')
#model.fit(x_train, y_train, epochs=2)
model.save('./catvdog')
model.evaluate(x_test, y_test, verbose=2)
#print("width_max: " + str(width_max))
#print("width_min: " + str(width_min))
#print("height_max: " + str(height_max))
#print("height_min: " + str(height_min))
