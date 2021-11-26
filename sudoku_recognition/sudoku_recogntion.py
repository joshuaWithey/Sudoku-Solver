import cv2
import numpy as np
from skimage.segmentation import clear_border
import tensorflow as tf
from tensorflow import keras


# Helper function to compute distance between two points using pythagorus
def distance_between(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p1[1]
    return int(np.sqrt(a ** 2 + b ** 2))

# Given an image of a sudoku board, return it cropped and straightened


def find_puzzle(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Guassian filter
    processed_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
    processed_image = cv2.adaptiveThreshold(
        processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert image
    processed_image = cv2.bitwise_not(processed_image)

    # Find contours to identify edges
    contours, _ = cv2.findContours(
        processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Largest contour should be the outside of the grid
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find corners of largest contour representing the sudoku grid

    # Bottom right corner of puzzle will have largest (x + y) value
    # Top left should have the smallest (x + y) value
    # Top right has largest (x - y) value
    # Bottom left has smallest (x - y) value
    bottom_right = contours[0][np.argmax(
        [point[0][0] + point[0][1] for point in contours[0]])][0]
    top_left = contours[0][np.argmin(
        [point[0][0] + point[0][1] for point in contours[0]])][0]
    top_right = contours[0][np.argmax(
        [point[0][0] - point[0][1] for point in contours[0]])][0]
    bottom_left = contours[0][np.argmin(
        [point[0][0] - point[0][1] for point in contours[0]])][0]

    # Crop and warp image
    # Store points of corners
    pt1 = np.float32([top_left, top_right, bottom_left, bottom_right])

    # Calculate side length of new picture.
    # Use maximum of all sides
    side = max(distance_between(top_left, top_right),
               distance_between(top_right, bottom_right),
               distance_between(bottom_right, bottom_left),
               distance_between(bottom_left, top_left)
               )

    # Define new image size
    pt2 = np.float32([[0, 0], [side, 0], [0, side], [side, side]])

    # Crop and warp original and grayscale images
    gray_image = cv2.warpPerspective(
        gray_image, cv2.getPerspectiveTransform(pt1, pt2), (side, side))

    image = cv2.warpPerspective(
        image, cv2.getPerspectiveTransform(pt1, pt2), (side, side))

    # Return images
    return(image, gray_image)

# Given a cell from a sudoku, return none if empty or the number in it


def identify_cell(cell):
    cell = cv2.threshold(cell, 0, 255,
                         cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cell = clear_border(cell)

    # Find contours to find black square
    contours, _ = cv2.findContours(
        cell.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, empty cell
    if len(contours) == 0:
        return None

    # Find largest contour to determine if a number
    largest_contour = max(contours, key=cv2.contourArea)

    # Create mask to surround digit
    mask = np.zeros(cell.shape, dtype="uint8")

    # Draw contour on mask
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    # Calculate perecentage of mask that is filled
    (height, width) = cell.shape
    if (cv2.countNonZero(mask) / float(height * width)) < 0.02:
        # cv2.imshow('none', cell)
        # cv2.waitKey(0)
        return None

    # Apply mask to initial cell
    digit = cv2.bitwise_and(cell, cell, mask=mask)
    # cv2.imshow('found', cell)
    # cv2.waitKey(0)

    # cv2.imshow('Digit', digit)
    # cv2.waitKey(0)

    return digit
