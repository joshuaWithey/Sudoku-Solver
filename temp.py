import cv2
import numpy as np
from utilities.utilities import extract_board
import tensorflow as tf
import math

# Load model
interpreter = tf.lite.Interpreter(model_path='utilities/model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Get input shape
input_shape = input_details[0]['shape']

# Import image
image = cv2.imread('output/sample.png')

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cell_size = image.shape[0] // 9
cell_area = cell_size * cell_size
limit = int(cell_size * 0.1) 
print(cell_area)


# Close horizontal and vertical lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

invert = cv2.bitwise_not(closed)

contours, _ = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Helper functions for sorting
def smallest_y(elem):
    x, y, w, h = cv2.boundingRect(elem)
    return y

def smallest_x(elem):
    x, y, w, h = cv2.boundingRect(elem)
    return x

# Sort contours by area and only use first 81
contours = sorted(contours, key=cv2.contourArea, reverse=True)
contours = contours[:81]
print(len(contours))

# Sort contours into top to bottom
contours = sorted(contours, key=smallest_y)
# Sort contours into left to right
start = 0
while start < 81:
    contours[start:start+9] = sorted(contours[start:start+9], key=smallest_x)
    start += 9

# print(len(contours))
# for c in contours:
#     area = cv2.contourArea(c)
#     print(area)



for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    temp = image[y:y+h, x:x+w]
    

cv2.imshow('image', image)
cv2.imshow('closed', closed)
cv2.imshow('invert', invert)
cv2.imshow('mask', temp)
cv2.waitKey()
