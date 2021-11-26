from sudoku_recognition.sudoku_recogntion import find_puzzle, identify_cell
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array

model = tf.keras.models.load_model('digit_classifier.h5')

# Import image
image = cv2.imread('sudoku2.jpg')

# Extract board from image
original_image, grayscale_image = find_puzzle(image)

# Initialize 9 x 9 sudoku board
board = np.zeros((9, 9), dtype='int')

# Calculate distance between each cell
# Since we transform into a square, only need one side
cell_size = original_image.shape[0] // 9

for y in range(0, 9):
    for x in range(0, 9):
        start_x = x * cell_size
        start_y = y * cell_size
        end_x = (x + 1) * cell_size
        end_y = (y + 1) * cell_size

        # Extract cell from board
        cell = grayscale_image[start_y:end_y, start_x:end_x]
        digit = identify_cell(cell)
        if digit is not None:
            digit = cv2.resize(digit, (28, 28))
            digit = digit.astype("float") / 255
            digit = img_to_array(digit)
            digit = np.expand_dims(digit, axis=0)

            pred = model.predict(digit)
            print(pred)
            board[y][x] = pred.argmax(axis=1)[0]

print(board)
