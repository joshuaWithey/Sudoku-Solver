import cv2
# import numpy as np
from django.conf import settings
from solver.utilities import crop_puzzle, extract_board, find_puzzle, overlay_puzzle, solve_sudoku
import tensorflow as tf
import numpy as np

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
		# Sudoku variables
		self.puzzle_not_found = 0
		self.puzzle_solved = False
		self.board = np.zeros((9, 9, 2), dtype='int')
		
		# Load model
		self.interpreter = tf.lite.Interpreter(model_path='solver/model.tflite')
		self.interpreter.allocate_tensors()

		# Get input and output tensors.
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		# Get input shape
		self.input_shape = self.input_details[0]['shape']


	def __del__(self):
		self.video.release()

	def get_frame(self):
		ret, frame = self.video.read()
		corners, processed_frame = find_puzzle(frame)

		if corners is not None:
			self.puzzle_not_found = 0
			if not self.puzzle_solved:
				# Extract digits from image
				self.cropped_frame = crop_puzzle(processed_frame, corners)
				self.board = extract_board(self.cropped_frame, self.interpreter,
										self.input_details, self.output_details)            
				
				if self.board is not None:
					if solve_sudoku(self.board):
						self.puzzle_solved = True
		else:
			self.puzzle_not_found += 1
			if self.puzzle_not_found > 10:
				self.puzzle_solved = False

		if self.puzzle_solved:
			frame = overlay_puzzle(frame, self.board, self.cropped_frame.shape[0], corners)

		resize = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR) 
		ret, jpeg = cv2.imencode('.jpg', resize)
		return jpeg.tobytes()