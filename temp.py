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
image = cv2.imread('output/sample.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# # Close horizontal and vertical lines
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

# cv2.imshow('test', gray)
# cv2.waitKey(0)

contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Create mask to surround digit
mask = np.zeros(image.shape, dtype="uint8")

for i in range(0, len(contours)):
    area = cv2.contourArea(contours[i])
    if area < 1000:
        cv2.drawContours(mask, contours[i], -1, 255, -1)

cv2.imshow('test', image)
cv2.imshow('test', mask)
cv2.waitKey()


# # Close horizontal and vertical lines
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
# gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

# cv2.imshow('test', gray)
# cv2.waitKey(0)

# invert = cv2.bitwise_not(gray)

# cv2.imshow('test', invert)
# cv2.waitKey(0)

# dst = cv2.Canny(gray, 50, 200, None, 3)

# # Copy edges to the images that will display the results in BGR
# cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
# cdstP = np.copy(cdst)

# lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#         cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)


# linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

# if linesP is not None:
#     for i in range(0, len(linesP)):
#         l = linesP[i][0]
#         cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]),
#                  (0, 0, 255), 3, cv2.LINE_AA)


# cv2.imshow("Source", image)
# cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
# cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

# cv2.waitKey()


# cv2.imshow('test', gray)
# cv2.waitKey(0)

# # Find contours
# contours, hierarchy = cv2.findContours(
#     gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# index = -1
# area = -1
# for i in range(0, len(contours)):
#     temp_area = cv2.contourArea(contours[i])
#     if temp_area > area:
#         area = temp_area
#         index = i

# # # Find largest contour representing outside grid
# # largest_contour = max(contours, key=cv2.contourArea)

# # Create mask to surround digit
# # mask = np.zeros(image.shape, dtype="uint8")

# # cv2.imshow('test', mask)
# # cv2.waitKey(0)

# # Draw contour on image
# cv2.drawContours(image, contours[index], -1, (255, 0, 0), -1)

# cv2.imshow('test', image)
# cv2.waitKey(0)

# for i in range(0, len(contours)):
#     if hierarchy[0][i][3] == index:
#         cv2.drawContours(image, contours[i], -1, (255, 0, 0), -1)


# cv2.imshow('test', image)
# cv2.waitKey(0)

# board = extract_board(image, interpreter,
#                       input_details, output_details)
