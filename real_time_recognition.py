import cv2
import numpy as np
import math
from utilities.utilities import crop_puzzle, extract_board, find_puzzle, identify_cell, overlay_puzzle, solve_sudoku
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path='utilities/model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Get input shape
input_shape = input_details[0]['shape']

# Import webcam
capture = cv2.VideoCapture(0)

# Counter for how long a puzzle was not found
puzzle_not_found = 0
# Boolean for whether a puzzle is solved, saves on repeatedly running predictions
puzzle_solved = False

if not capture.isOpened():
    raise IOError

while True:
    ret, frame = capture.read()
    # Find corners of puzzle out of frame
    corners, processed_frame = find_puzzle(frame)

    c = cv2.waitKey(1)

    if c == 32:

        if corners is not None:
            print(puzzle_solved)
            puzzle_not_found = 0
            if not puzzle_solved:
                # Extract digits from image
                cropped_frame = crop_puzzle(processed_frame, corners)
                board = extract_board(cropped_frame, interpreter,
                                      input_details, output_details)

                if board is not None:
                    if solve_sudoku(board):
                        puzzle_solved = True
        else:
            puzzle_not_found += 1
            if puzzle_not_found > 10:
                puzzle_solved = False

    if puzzle_solved:
        frame = overlay_puzzle(frame, board, cropped_frame.shape[0], corners)

    # if processed_frame is not None:
    #     cv2.imshow('Input', processed_frame)
    # else:
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break


capture.release()
cv2.destroyAllWindows()
