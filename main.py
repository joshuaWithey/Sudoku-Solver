from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.layout import Layout
from kivy.graphics.texture import Texture
from kivy.graphics import *
import cv2
import numpy as np
from utilities.utilities import find_puzzle, identify_cell, crop_puzzle, overlay_puzzle, solve_sudoku
import tensorflow as tf
from keras.preprocessing.image import img_to_array

model = tf.keras.models.load_model('utilities/digit_classifier.h5')


class Test(App):
    def build(self):
        layout = BoxLayout(orientation="vertical")
        image = Image()
        layout.add_widget(image)

        # Load input
        input_image = cv2.imread('test_images/sudoku.jpg')

        # # Find corners of puzzle out of frame
        corners, processed_image = find_puzzle(input_image)

        # # Crop and warp puzzle from frame
        cropped_image = crop_puzzle(processed_image, corners)

        # Initialize 9 x 9 sudoku board
        # Use 3d array to store the number in a cell, as well as whether
        # or not that number is given.
        board = np.zeros((9, 9, 2), dtype='int')

        # Calculate distance between each cell
        # Since we transform into a square, only need one side
        cell_size = cropped_image.shape[0] // 9
        start_x = 0
        start_y = 0
        end_x = cell_size
        end_y = cell_size

        for y in range(0, 9):
            for x in range(0, 9):
                start_x = x * cell_size
                start_y = y * cell_size
                end_x = (x + 1) * cell_size
                end_y = (y + 1) * cell_size

                # Extract cell from board
                cell = cropped_image[start_y:end_y, start_x:end_x]
                digit = identify_cell(cell)
                if digit is not None:
                    digit = cv2.resize(digit, (28, 28))
                    digit = digit.astype("float") / 255
                    digit = img_to_array(digit)
                    digit = np.expand_dims(digit, axis=0)

                    pred = model.predict(digit)
                    board[y][x][0] = pred.argmax(axis=1) + 1
                    board[y][x][1] = 1

        solve_sudoku(board)

        final_image = overlay_puzzle(
            input_image, board, cropped_image.shape[0], corners)

        buffer = cv2.flip(final_image, 0).tobytes()
        texture = Texture.create(
            size=(final_image.shape[1], final_image.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        image.texture = texture

        return layout


test = Test()
test.run()
