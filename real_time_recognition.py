import cv2
import numpy as np
import math
from sudoku_recognition.sudoku_recogntion import crop_puzzle, find_puzzle, identify_cell
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import time

# Helper function to determine if image is blurry
# Compute laplcian of the image and return the focus
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# model = tf.keras.models.load_model('digit_classifier.h5')


def distance_between(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p1[1]
    return int(np.sqrt(a ** 2 + b ** 2))


capture = cv2.VideoCapture(0)

# Sudoku board
board = np.zeros((9, 9), dtype='int')

# Sudoku truth table board
solved_board = np.zeros((9, 9), dtype='bool')

# Counter for how long a puzzle has been found
puzzle_found = 0
puzzle_solved = False
cell_solved = 0

if not capture.isOpened():
    raise IOError


while True:
    ret, frame = capture.read()

    # Apply grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find corners of puzzle out of frame
    corners = find_puzzle(gray_frame)    

    if corners is not None:
        
        # Draw square around puzzle
        cv2.polylines(frame, [corners], True, (0,255,255))

        # Crop and warp puzzle from frame
        cropped_frame = crop_puzzle(gray_frame, corners)
        
        cell_size = cropped_frame.shape[0] // 9

        for y in range(0, 9):
            for x in range(0, 9):
                if solved_board[y][x] == False:
                    start_x = x * cell_size
                    start_y = y * cell_size
                    end_x = (x + 1) * cell_size
                    end_y = (y + 1) * cell_size

                    # Extract cell from board
                    cell = cropped_frame[start_y:end_y, start_x:end_x]
                    digit = identify_cell(cell)
                
        #             if digit is not None:
        #                 digit = cv2.resize(digit, (28, 28))
        #                 digit = digit.astype("float") / 255
        #                 digit = img_to_array(digit)
        #                 digit = np.expand_dims(digit, axis=0)

        #                 pred = model.predict(digit)
        #                 if max(pred[0]) > 0.98:
        #                     solved_board[y][x] = True
        #                     board[y][x] = pred.argmax(axis=1)[0]
        #             else:
        #                 solved_board[y][x] = True
        # if solved_board.sum() == 9 * 9:
        #     puzzle_solved = True
        #     print(board)
        #     break
    

    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)

    if c == 27:
        break


capture.release()
cv2.destroyAllWindows()
