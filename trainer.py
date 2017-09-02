#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 23:37:06 2017

@author: rjac
"""
from tensorflow.contrib.keras.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.contrib.keras.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.contrib.keras.python.keras.preprocessing import image
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense #, GlobalAveragePooling2D
import tensorflow.contrib.keras as k

IM_WIDTH = 256
IM_HEIGHT = 256
batch_size = 10
train_dir = "./images/train/"
val_dir = "./images/validation/"

#Step 1 - Input images
def train_model(epochs,steps):
    train_datagen =  image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    test_datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    train_generator = train_datagen.flow_from_directory(
      train_dir,
      target_size=(IM_WIDTH, IM_HEIGHT),
      batch_size=batch_size
    )
    
    validation_generator = test_datagen.flow_from_directory(
      val_dir,
      target_size=(IM_WIDTH, IM_HEIGHT),
      batch_size=batch_size,
    )
    
    
    base_model = InceptionV3(weights='imagenet', include_top=False,pooling='avg')
    net = base_model.output
    net = Dense(1024,activation='relu')(net)
    net = Dense(2,activation='softmax')(net)
    outputs = net
    
    model = Model(inputs=base_model.input,outputs=outputs)
    
    for layer in base_model.layers:
        layer.trainable = False
        
    model.compile(optimizer=k.optimizers.Adam(),loss=k.losses.categorical_crossentropy, metrics=['accuracy'])
    model.fit_generator(train_generator,steps_per_epoch=steps,epochs=epochs,validation_data=validation_generator,validation_steps=steps,class_weight='auto')

    model.save("./models/model_v001.h5")
    
    
if __name__=="__main__":
    train_model(10,50)