import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.datasets import mnist

# import written images
(train_X, train_y), (test_X, test_y) = mnist.load_data()

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

# Remove 0s
train_filter = np.where((train_labels != 0))
train_data, train_labels = train_data[train_filter], train_labels[train_filter]

# Add channel
train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))

# Scale values to between 0 and 1
train_data = train_data / 255

# Binarize labels
le = LabelBinarizer()
train_labels = le.fit_transform(train_labels)

model = Sequential()
inputShape = (28, 28, 1)

model.add(Conv2D(32, (5, 5), padding="same",
                 input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# second set of CONV => RELU => POOL layers
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# first set of FC => RELU layers
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
# second set of FC => RELU layers
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
# softmax classifier
model.add(Dense(9))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer='adam',
              metrics=["accuracy"])

# Training
model.fit(train_data, train_labels, epochs=10)

# Serialize the model to disk
model.save('utilites/digit_classifier.h5')
