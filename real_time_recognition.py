import cv2
import numpy as np
import math
from sudoku_recognition.sudoku_recogntion import find_puzzle, identify_cell
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array
import time

# model = tf.keras.models.load_model('digit_classifier.h5')


def distance_between(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p1[1]
    return int(np.sqrt(a ** 2 + b ** 2))


capture = cv2.VideoCapture(0)

# Sudoku board
board = np.zeros((9, 9), dtype='int')

# Counter for how long a puzzle has been found
puzzle_found = 0
puzzle_solved = False

if not capture.isOpened():
    raise IOError

time.sleep(5)


while True:
    ret, frame = capture.read()
    # frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

    # Apply grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Guassian filter
    processed_frame = cv2.GaussianBlur(gray_frame, (9, 9), 2)
    processed_frame = cv2.adaptiveThreshold(
        processed_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert image
    processed_frame = cv2.bitwise_not(processed_frame)

    # Find contours to identify edges
    contours, _ = cv2.findContours(
        processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Largest contour should be the outside of the grid
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find corners of largest contour representing the sudoku grid

    # Bottom right corner of puzzle will have largest (x + y) value
    # Top left should have the smallest (x + y) value
    # Top right has largest (x - y) value
    # Bottom left has smallest (x - y) value
    if len(contours) > 0:
        # Check if contour is square
        peri = cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], 0.02 * peri, True)

        if len(approx == 4):
            (x, y, w, h) = cv2.boundingRect(approx)
            ratio = w / float(h)
            if ratio > 0.95 and ratio < 1.05:

                puzzle_found += 1
                bottom_right = contours[0][np.argmax(
                    [point[0][0] + point[0][1] for point in contours[0]])][0]
                top_left = contours[0][np.argmin(
                    [point[0][0] + point[0][1] for point in contours[0]])][0]
                top_right = contours[0][np.argmax(
                    [point[0][0] - point[0][1] for point in contours[0]])][0]
                bottom_left = contours[0][np.argmin(
                    [point[0][0] - point[0][1] for point in contours[0]])][0]
                if (puzzle_found > 100):

                    cv2.line(frame, bottom_left, bottom_right, (0, 255, 0), 1)
                    cv2.line(frame, bottom_left, top_left, (0, 255, 0), 1)
                    cv2.line(frame, bottom_right, top_right, (0, 255, 0), 1)
                    cv2.line(frame, top_left, top_right, (0, 255, 0), 1)

                    # if (not puzzle_solved):
                    #     # # Crop and warp image
                    #     # Store points of corners
                    #     pt1 = np.float32(
                    #         [top_left, top_right, bottom_left, bottom_right])

                    #     # Calculate side length of new picture.
                    #     # Use maximum of all sides
                    #     side = max(distance_between(top_left, top_right),
                    #                distance_between(top_right, bottom_right),
                    #                distance_between(bottom_right, bottom_left),
                    #                distance_between(bottom_left, top_left)
                    #                )

                    #     # Define new image size
                    #     pt2 = np.float32(
                    #         [[0, 0], [side, 0], [0, side], [side, side]])

                    #     # Crop and warp original and grayscale images
                    #     gray_frame = cv2.warpPerspective(
                    #         gray_frame, cv2.getPerspectiveTransform(pt1, pt2), (side, side))

                    #     cell_size = gray_frame.shape[0] // 9

                    #     for y in range(0, 9):
                    #         for x in range(0, 9):
                    #             start_x = x * cell_size
                    #             start_y = y * cell_size
                    #             end_x = (x + 1) * cell_size
                    #             end_y = (y + 1) * cell_size

                    #             # Extract cell from board
                    #             cell = gray_frame[start_y:end_y, start_x:end_x]
                    #             digit = identify_cell(cell)
                    #             if digit is not None:
                    #                 digit = cv2.resize(digit, (28, 28))
                    #                 digit = digit.astype("float") / 255
                    #                 digit = img_to_array(digit)
                    #                 digit = np.expand_dims(digit, axis=0)

                    #                 pred = model.predict(digit)
                    #                 board[y][x] = pred.argmax(axis=1)[0]

                    #     puzzle_solved = True
                    #     print(board)

        else:
            puzzle_found = 0
            puzzle_solved = False

    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)

    if c == 27:
        break


capture.release()
cv2.destroyAllWindows()
