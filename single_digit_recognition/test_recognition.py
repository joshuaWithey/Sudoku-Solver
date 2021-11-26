import imutils
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plot
import numpy as np
import argparse
import cv2

# # Import digit images
# (train_data, train_labels), (test_data,
#                              test_labels) = keras.datasets.mnist.load_data()

# # Scale values to between 0 and 1
# train_data = train_data / 255
# test_data = test_data / 255

# model = tf.keras.models.load_model('digit_classifier.h5')
# model.evaluate(test_data, test_labels)

# prediction = model.predict(test_data)
# print(test_labels[6])
# print(np.argmax(prediction[6]))

# Import image
image = cv2.imread('eight.png', 0)
resized = cv2.resize(image, (28, 28))
cv2.imshow('image', resized)
cv2.waitKey(0)


# Resize
# Invert color to match data set
# Convert into 28 x 28 int? array, matching pixel brightness
