from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
from utilities.utilities import crop_puzzle, extract_board, find_puzzle, identify_cell, overlay_puzzle, solve_sudoku
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('utilities/digit_classifier.h5')


class MainApp(App):
    def build(self):
        layout = BoxLayout(orientation="vertical")
        self.image = Image()
        self.puzzle_solved = False
        self.counter = 0
        self.buffer = 10
        self.temp = 0
        self.board = np.zeros((9, 9, 3), dtype='int')

        layout.add_widget(self.image)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.load_video, 1.0/30.0)
        return layout

    def load_video(self, *args):
        print(self.counter)
        ret, frame = self.capture.read()
        corners, processed_frame = find_puzzle(frame)

        if corners is not None:
            self.counter += 1
            if not self.puzzle_solved and self.counter > self.buffer:
                # Extract digits from image
                self.cropped_frame = crop_puzzle(processed_frame, corners)
                # cv2.imwrite(
                #     f'output/image_{self.temp}.jpg', self.cropped_frame)
                # self.temp += 1
                self.board = extract_board(
                    self.cropped_frame, self.board, model)
                sum = 0
                if self.board is not None:
                    for x in range(0, 9):
                        for y in range(0, 9):
                            sum += self.board[y][x][2]
                    if sum == 81:
                        solve_sudoku(self.board)
                        print(self.board)
                        self.puzzle_solved = True
        else:
            self.counter = min(self.counter, self.buffer)
            self.counter = max(0, self.counter - 1)
            if self.counter == 0:
                self.puzzle_solved = False
                self.board = np.zeros((9, 9, 3), dtype='int')

        if self.puzzle_solved:
            frame = overlay_puzzle(
                frame, self.board, self.cropped_frame.shape[0], corners)

        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture


test = MainApp()
test.run()
