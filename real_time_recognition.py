import cv2
import numpy as np
import math
from utilities.utilities import crop_puzzle, extract_board, find_puzzle, identify_cell, overlay_puzzle, solve_sudoku
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import time

model = tf.keras.models.load_model('utilities/digit_classifier.h5')

# Import webcam
capture = cv2.VideoCapture(0)

# Counter for how long a puzzle was 
puzzle_not_found_counter = 0
puzzle_solved = False

if not capture.isOpened():
    raise IOError

while True:
    ret, frame = capture.read()
    # Find corners of puzzle out of frame
    corners, processed_frame = find_puzzle(frame) 

    if corners is not None and not puzzle_solved:
        puzzle_not_found_counter = 0
        # Extract digits from image
        cropped_frame = crop_puzzle(processed_frame, corners)
        board = extract_board(cropped_frame, model)
        
        if board is not None and solve_sudoku(board):
            puzzle_solved = True
    else:
        puzzle_not_found_counter += 1
    # If not puzzle found for 100 loops, assumes previously found puzzle was lost
    if puzzle_not_found_counter > 100:
        puzzle_solved = False

    if puzzle_solved:      
        frame = overlay_puzzle(frame, board, cropped_frame.shape[0], corners)
        
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)

    if c == 27:
        break


capture.release()
cv2.destroyAllWindows()


