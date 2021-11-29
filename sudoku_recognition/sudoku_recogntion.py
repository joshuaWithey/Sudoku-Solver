import cv2
import numpy as np
from skimage.segmentation import clear_border
import tensorflow as tf
from tensorflow import keras


# Helper function to determine if image is blurry
# Compute laplcian of the image and return the focus
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Helper function to compute distance between two points


def distance_between(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p1[1]
    return int(np.sqrt(a ** 2 + b ** 2))

# Given an image of a sudoku board, return it cropped and straightened, as well as array of corners


def find_puzzle(image):
    # If image too blurry
    if variance_of_laplacian(image) < 300:
        return None

    # Apply grayscale
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

    # Check if contour is square
    peri = cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], 0.02 * peri, True)
    (x, y, w, h) = cv2.boundingRect(approx)
    ratio = w / float(h)
    if len(approx) == 4:
        # and ratio > 0.95 and ratio < 1.05:
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

        return np.array([top_left, top_right, bottom_right, bottom_left]), processed_image

    else:
        return None


def crop_puzzle(image, corners):
    # Crop and warp image
    # Store points of corners
    pt1 = np.float32(corners)

    # Calculate side length of new picture.
    # Use maximum of all sides
    side = max(distance_between(corners[0], corners[1]),
               distance_between(corners[1], corners[2]),
               distance_between(corners[2], corners[3]),
               distance_between(corners[3], corners[0])
               )

    # Define new image size
    pt2 = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]], dtype="float32")

    # Crop and warp original and grayscale images

    image = cv2.warpPerspective(
        image, cv2.getPerspectiveTransform(pt1, pt2), (side, side))

    # Return image
    return(image)


def identify_cell(cell):
    limit = cell.shape[0] // 10

    processed_cell = cv2.threshold(cell, 128, 255, cv2.THRESH_BINARY)[1]

    processed_cell = clear_border(cell)

    # Find contours to find black square
    contours, _ = cv2.findContours(
        processed_cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, empty cell
    if len(contours) == 0:
        return None

    # Find largest contour to determine if a number
    largest_contour = max(contours, key=cv2.contourArea)

    # Find if largest contour is close to edges, indicating noise
    (x, y, w, h) = cv2.boundingRect(largest_contour)

    if x < limit or y < limit or w > cell.shape[0] - limit or h > cell.shape[0] - limit:
        return None

    # processed_cell = processed_cell[x:x+w, y:y+h]

    # Create mask to surround digit
    mask = np.zeros(cell.shape, dtype="uint8")

    # Draw contour on mask
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    # mask = mask[y:y+h, x:x+w]

    # Calculate perecentage of mask that is filled
    (height, width) = cell.shape
    if (cv2.countNonZero(mask) / float(height * width)) < 0.05:
        return None

    # Apply mask to initial cell
    digit = cv2.bitwise_and(cell, cell, mask=mask)

    # Final crop
    side = max(w, h)
    # Edge size of returned square
    # Add margin
    side = side + side // 5
    start_y = y-((side-h)//2)
    start_x = x-((side-w)//2)

    # Calculatee cropped digit
    digit = digit[start_y:start_y+side, start_x:start_x+side]

    return digit
