import cv2
import numpy as np
import math
from sudoku_recognition.sudoku_recogntion import crop_puzzle, find_puzzle, identify_cell
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import time

model = tf.keras.models.load_model('digit_classifier.h5')

# Import webcam
capture = cv2.VideoCapture(0)

# Sudoku board
board = np.zeros((9, 9), dtype='int')

# Sudoku truth table board
solved_board = np.zeros((9, 9), dtype='bool')

# Counter for how long a puzzle has been found
puzzle_found = False
puzzle_solved = False
cell_solved = 0

if not capture.isOpened():
    raise IOError

while True:
    ret, frame = capture.read()

    # Find corners of puzzle out of frame
    corners, processed_frame = find_puzzle(frame)    

    if corners is not None:
        cropped_frame = crop_puzzle(processed_frame, corners)        
        # Draw square around puzzle
        cv2.polylines(frame, [corners], True, (0,255,255), 3)
        if not puzzle_found:
            # Crop and warp puzzle from frame        
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
                    
                        if digit is not None:
                            digit = cv2.resize(digit, (28, 28))
                            digit = digit.astype("float") / 255
                            digit = img_to_array(digit)
                            digit = np.expand_dims(digit, axis=0)

                            pred = model.predict(digit)
                            if max(pred[0]) > 0.95:
                                board[y][x] = pred.argmax(axis=1)[0] + 1
                                solved_board[y][x] = True
                        else:
                            solved_board[y][x] = True
    
        if solved_board.sum() == 81:
            new_image = cv2.putText(
                img = frame,
                text = "Good Morning",
                org = (200, 200),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 3.0,
                color = (125, 246, 55),
                thickness = 3
            )
            puzzle_found = True

    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)

    if c == 27:
        break


capture.release()
cv2.destroyAllWindows()


