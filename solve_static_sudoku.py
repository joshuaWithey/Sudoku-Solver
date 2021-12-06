from utilities.utilities import extract_board, find_puzzle, identify_cell, crop_puzzle, overlay_puzzle, solve_sudoku
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path='utilities/model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Get input shape
input_shape = input_details[0]['shape']

# Import image
image = cv2.imread('test_images/sudoku2.jpg')

# Find corners of puzzle out of frame
corners, processed_image = find_puzzle(image)

# Crop and warp puzzle from frame
cropped_image = crop_puzzle(processed_image, corners)

# Calculate distance between each cell
# Since we transform into a square, only need one side
cell_size = cropped_image.shape[0] // 9

board = extract_board(cropped_image, interpreter,
                      input_details, output_details)

solve_sudoku(board)

final_image = overlay_puzzle(image, board, cropped_image.shape[0], corners)

cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
