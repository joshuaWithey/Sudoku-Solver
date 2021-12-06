import cv2
import numpy as np
from skimage.segmentation import clear_border
from keras.preprocessing.image import img_to_array
from sudoku import Sudoku


# Helper function to determine if image is blurry
# Compute laplcian of the image and return the focus
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Helper function to compute distance between two points


def distance_between(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p1[1]
    return int(np.sqrt(a ** 2 + b ** 2))

# Given an image of a sudoku board, return it processed, as well as array of corners


def find_puzzle(image):
    # If image too blurry
    if variance_of_laplacian(image) < 200:
        return None, None

    # Apply grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Guassian filter
    processed_image = cv2.GaussianBlur(
        gray_image, (7, 7), 0)

    # Threshold
    processed_image = cv2.adaptiveThreshold(
        processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)

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
        # and ratio > 0.85 and ratio < 1.15:
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
        return None, None


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
    side = cell.shape[0]
    border = int(side * 0.1)

    # remove outer 10% of cell
    cell = cell[border:side-border, border:side-border]

    # Find contours to find black square
    contours, _ = cv2.findContours(
        cell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, empty cell
    if len(contours) == 0:
        return None

    # Find largest contour to determine if a number
    largest_contour = max(contours, key=cv2.contourArea)

    # # Find if largest contour is close to edges, indicating noise
    # (x, y, w, h) = cv2.boundingRect(largest_contour)

    # if x < limit or y < limit or w > cell.shape[0] - limit or h > cell.shape[0] - limit:
    #     # cv2.imshow('cell', cell)
    #     # cv2.waitKey(0)
    #     # print('Limit')
    #     return None

    # Create mask to surround digit
    mask = np.zeros(cell.shape, dtype="uint8")

    # Draw contour on mask
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    # Calculate perecentage of mask that is filled
    (height, width) = cell.shape
    if (cv2.countNonZero(mask) / float(height * width)) < 0.05:
        return None

    # Apply mask to initial cell
    digit = cv2.bitwise_and(cell, cell, mask=mask)

    # # Final crop
    # side = max(w, h)
    # # Edge size of returned square
    # # Add margin
    # side = side + side // 5
    # start_y = y-((side-h)//2)
    # start_x = x-((side-w)//2)

    # # Calculate cropped digit
    # digit = digit[start_y:start_y+side, start_x:start_x+side]

    cv2.imshow('cell', digit)
    cv2.waitKey(0)

    return digit


def overlay_puzzle(image, board, grid_size, corners):
    try:
        colour = (0, 255, 0)

        # Draw overlay
        overlay = np.zeros((grid_size, grid_size, 3), np.uint8)
        border_pts = np.array([[0, 0], [grid_size, 0], [grid_size,
                                                        grid_size], [0, grid_size]], np.int32)

        # Draw polygon for border
        cv2.polylines(overlay, [border_pts], True, colour, 3)

        # Draw gridlines
        cell_size = overlay.shape[0] // 9
        for x in range(1, 9):
            cv2.line(overlay, (0, x * cell_size),
                     (grid_size, x * cell_size), colour, 1)
            cv2.line(overlay, (x * cell_size, 0),
                     (x * cell_size, grid_size), colour, 1)

        # Draw digits onto overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = float(cell_size / 50)
        for x in range(0, 9):
            for y in range(0, 9):
                if board[y][x][1] == 0:
                    text = str(board[y][x][0])
                    text_size = cv2.getTextSize(text, font, scale, 4)[0]
                    corner_x = x * cell_size + \
                        ((cell_size - text_size[0]) // 2)
                    corner_y = y * cell_size + \
                        ((cell_size + text_size[1]) // 2)
                    cv2.putText(overlay, text, (corner_x, corner_y),
                                font, scale, colour, 4)

        # Warp overlay onto original frame
        pt1 = np.float32(border_pts)
        pt2 = np.float32(corners)

        overlay = cv2.warpPerspective(
            overlay, cv2.getPerspectiveTransform(pt1, pt2), (image.shape[1], image.shape[0]))

        # Add overlay to original image
        added_image = cv2.addWeighted(image, 1, overlay, 1, 0)

        return added_image
    except:
        return image


def extract_board(image, interpreter, input_details, output_details):
    try:
        board = np.zeros((9, 9, 2), dtype='int')
        cell_size = image.shape[0] // 9
        for y in range(0, 9):
            for x in range(0, 9):
                start_x = x * cell_size
                start_y = y * cell_size
                end_x = (x + 1) * cell_size
                end_y = (y + 1) * cell_size

                # Extract cell from board
                cell = image[start_y:end_y, start_x:end_x]
                digit = identify_cell(cell)

                if digit is not None:
                    # cv2.imshow('digit', digit)
                    # cv2.waitKey(0)
                    digit = cv2.resize(digit, (28, 28))
                    digit = digit.astype("float") / 255
                    digit = img_to_array(digit)
                    digit = np.expand_dims(digit, axis=0)

                    interpreter.set_tensor(input_details[0]['index'], digit)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(
                        output_details[0]['index'])

                    board[y][x][0] = np.argmax(output_data) + 1
                    board[y][x][1] = 1
        return board
    except:
        return None


def is_valid(board, number, position):
    # Check row
    for i in range(0, 9):
        if board[position[0]][i][0] == number and position[1] != i:
            return False

    # Check column
    for i in range(0, 9):
        if board[i][position[1]][0] == number and position[0] != i:
            return False

    # Check square
    x_start = position[1] // 3 * 3
    y_start = position[0] // 3 * 3
    for i in range(y_start, y_start + 3):
        for j in range(x_start, x_start + 3):
            if board[i][j][0] == number and (i, j) != position:
                return False
    return True


def find_empty(board):
    for i in range(0, 9):
        for j in range(0, 9):
            if board[i][j][0] == 0:
                return [i, j]
    return None


def solve_sudoku(board):
    empty = find_empty(board)
    if not empty:
        return True
    else:
        row, col = empty
    for i in range(1, 10):
        if is_valid(board, i, (row, col)):
            board[row][col][0] = i
            if solve_sudoku(board):
                return True
            board[row][col][0] = 0
    return False

# board = [[8, 0, 0, 0, 1, 0, 0, 0, 9],
#          [0, 5, 0, 8, 0, 7, 0, 1, 0],
#          [0, 0, 4, 0, 9, 0, 7, 0, 0],
#          [0, 6, 0, 7, 0, 1, 0, 2, 0],
#          [5, 0, 8, 0, 6, 0, 1, 0, 7],
#          [0, 1, 0, 5, 0, 2, 0, 9, 0],
#          [0, 0, 7, 0, 4, 0, 6, 0, 0],
#          [0, 8, 0, 3, 0, 9, 0, 4, 0],
#          [3, 0, 0, 0, 5, 0, 0, 0, 8]]

# puzzle = Sudoku(3, 3, board=board)

# puzzle.solve().show_full()

# print(puzzle)
