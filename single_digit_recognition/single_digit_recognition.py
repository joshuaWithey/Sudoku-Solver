from pandas.core.arrays.integer import UInt8Dtype
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

# import written images
(train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()

# Import text images
data = pd.read_csv('TMNIST_Data.csv')

# Extract labels
train_labels = np.array(data.pop('labels'))

# Convert to numpy array
train_data = np.array(data.drop(['names'], axis=1), dtype=np.uint8)

# Reshape into 28 x 28 images
train_data = np.reshape(train_data, (-1, 28, 28))

# Combine datasets

train_data = np.concatenate((train_data, train_X), axis=0)
train_labels = np.concatenate((train_labels, train_y), axis=0)

# # Scale values to between 0 and 1
train_data = train_data / 255
# test_data = test_data / 255


model = keras.Sequential()

# Create neural network
# First layer
model.add(keras.layers.Conv2D(
    input_shape=(28, 28, 1),
    kernel_size=5,
    filters=8,
    strides=1,
    activation='relu',
    kernel_initializer='variance_scaling'
))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second layer
model.add(keras.layers.Conv2D(
    input_shape=(28, 28, 1),
    kernel_size=5,
    filters=16,
    strides=1,
    activation='relu',
    kernel_initializer='variance_scaling'
))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Flatten model
model.add(keras.layers.Flatten())

# Compress to output
model.add(keras.layers.Dense(
    units=10,
    kernel_initializer='variance_scaling',
    activation='softmax'
))

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(train_data, train_labels, epochs=10)

# Serialize the model to disk
model.save('digit_classifier.h5')
