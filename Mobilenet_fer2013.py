# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AptQ_AZTgyU1ksP0vGDwiP7RnnSgybn4
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd

import pandas as pd

#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam,SGD
from keras.regularizers import l1, l2
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sn
#import skimage.io
import keras.backend as K
import tensorflow as tf
from keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization ,Activation
from tensorflow.keras.models import Model, Sequential
from keras.applications.nasnet import NASNetLarge
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Import necessary libraries
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the input image size
img_size = (224, 224)

# Create an instance of the MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Add new output layers to the MobileNet model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

# Combine the base MobileNet model with the new output layers
model1 = Model(inputs=base_model.input, outputs=predictions)
model = tf.keras.models.load_model('checked_model.h5')

valid_datagen = ImageDataGenerator(rescale=1./255)
batch_size=64
test_dir='Affectnet_testing_dataset'
loss,accuracy=model.evaluate(valid_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True))
print('Affectnet')
print(accuracy)
print(loss)

test_dir='raf_final'
loss,accuracy=model.evaluate(valid_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True))
print('Raf')
print(accuracy)
print(loss)



test_dir='FER 2013 Datatset/test'
loss,accuracy=model.evaluate(valid_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True))
print('FER 2013')
print(accuracy)
print(loss)

test_dir='kdef Original'
loss,accuracy=model.evaluate(valid_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True))
print('kdef')
print(accuracy)
print(loss)

test_dir='Jafee Original/test'
loss,accuracy=model.evaluate(valid_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True))
print('Jafee')
print(accuracy)
print(loss)

test_dir='ckplus original'
loss,accuracy=model.evaluate(valid_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True))
print('ck+')
print(accuracy)
print(loss)

# Freeze the base MobileNet layers to prevent them from being trained
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with appropriate loss and optimizer functions
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define data augmentation and preprocessing for the training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Define data augmentation and preprocessing for the validation data
valid_datagen = ImageDataGenerator(rescale=1./255)

# Define the directories for the training and validation data
train_dir = 'FER 2013 Datatset/train'
valid_dir = 'FER 2013 Datatset/val'

# Define the batch size for the training data
batch_size = 64

# Define the number of training and validation steps
train_steps = int(np.ceil(22000 / batch_size))
valid_steps = int(np.ceil(3588 / batch_size))

# Train the model using the training data and validate using the validation data
history = model.fit(
    train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical',shuffle=True),
    steps_per_epoch=train_steps,
    validation_data=valid_datagen.flow_from_directory(valid_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True),
    validation_steps=valid_steps,
    epochs=10)


batch_size=64
train_steps = int(np.ceil(168 / batch_size))
valid_steps = int(np.ceil(20 / batch_size))
train_dir = 'Jafee Original/train'
valid_dir = 'Jafee Original/val'
history = model.fit(
    train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical',shuffle=True),
    steps_per_epoch=train_steps,
    validation_data=valid_datagen.flow_from_directory(valid_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True),
    validation_steps=valid_steps,
    epochs=5)



# Freeze the base MobileNet layers to prevent them from being trained
for layer in base_model.layers:
    layer.trainable = True

# Compile the model with appropriate loss and optimizer functions
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define data augmentation and preprocessing for the training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Define data augmentation and preprocessing for the validation data
valid_datagen = ImageDataGenerator(rescale=1./255)

# Define the directories for the training and validation data
train_dir = 'FER 2013 Datatset/train'
valid_dir = 'FER 2013 Datatset/val'

# Define the batch size for the training data
batch_size = 64

# Define the number of training and validation steps
train_steps = int(np.ceil(22000 / batch_size))
valid_steps = int(np.ceil(3588 / batch_size))

# Train the model using the training data and validate using the validation data
history = model.fit(
    train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical',shuffle=True),
    steps_per_epoch=train_steps,
    validation_data=valid_datagen.flow_from_directory(valid_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True),
    validation_steps=valid_steps,
    epochs=10)



batch_size=64
train_steps = int(np.ceil(168 / batch_size))
valid_steps = int(np.ceil(20 / batch_size))
train_dir = 'Jafee Original/train'
valid_dir = 'Jafee Original/val'
history = model.fit(
    train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical',shuffle=True),
    steps_per_epoch=train_steps,
    validation_data=valid_datagen.flow_from_directory(valid_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True),
    validation_steps=valid_steps,
    epochs=5)






test_dir='FER 2013 Datatset/test'
loss,accuracy=model.evaluate(valid_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True))
print('FER 2013')
print(accuracy)
print(loss)

test_dir='kdef Original'
loss,accuracy=model.evaluate(valid_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True))
print('kdef')
print(accuracy)
print(loss)

test_dir='Jafee Original/test'
loss,accuracy=model.evaluate(valid_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True))
print('Jafee')
print(accuracy)
print(loss)

test_dir='ckplus original'
loss,accuracy=model.evaluate(valid_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True))
print('ck+')
print(accuracy)
print(loss)


model.save('checked_model.h5')