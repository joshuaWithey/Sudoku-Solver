import cv2
import numpy as np
import math

def distance_between(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p1[1]
    return int(np.sqrt(a ** 2 + b ** 2))

capture = cv2.VideoCapture(0)

# Sudoku board
board = np.zeros((9, 9), dtype='int')

# Counter for how long a puzzle has been found
puzzle_found = 0

if not capture.isOpened():
    raise IOError

while True:
    ret, frame = capture.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    # Apply grayscale
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Guassian filter
    processed_frame = cv2.GaussianBlur(processed_frame, (9, 9), 2)
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
        bottom_right = contours[0][np.argmax([point[0][0] + point[0][1] for point in contours[0]])][0]
        top_left = contours[0][np.argmin(
            [point[0][0] + point[0][1] for point in contours[0]])][0]
        top_right = contours[0][np.argmax(
            [point[0][0] - point[0][1] for point in contours[0]])][0]
        bottom_left = contours[0][np.argmin(
            [point[0][0] - point[0][1] for point in contours[0]])][0]
        if (math.isclose(distance_between(top_left, bottom_right), distance_between(top_right, bottom_left), rel_tol=0.05)):
            puzzle_found += 1
            if (puzzle_found > 5):
                cv2.line(frame, bottom_left, bottom_right, (0,255,0), 3)
                cv2.line(frame, bottom_left, top_left, (0,255,0), 3)
                cv2.line(frame, bottom_right, top_right, (0,255,0), 3)
                cv2.line(frame, top_left, top_right, (0,255,0), 3)

            # Draw corners on grid
            # cv2.circle(frame, bottom_right, 5, (255,0,0), 3)
            # cv2.circle(frame, bottom_left, 5, (255,0,0), 3)
            # cv2.circle(frame, top_right, 5, (255,0,0), 3)
            # cv2.circle(frame, top_left, 5, (255,0,0), 3)
        else:
            puzzle_found = 0

    
    
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    
    

    

    

    

    # # Crop and warp image
    # # Store points of corners
    # pt1 = np.float32([top_left, top_right, bottom_left, bottom_right])

    # # Calculate side length of new picture.
    # # Use maximum of all sides
    # side = max(distance_between(top_left, top_right),
    #            distance_between(top_right, bottom_right),
    #            distance_between(bottom_right, bottom_left),
    #            distance_between(bottom_left, top_left)
    #            )

    # # Define new image size
    # pt2 = np.float32([[0, 0], [side, 0], [0, side], [side, side]])

    # # Crop and warp original and grayscale images
    # gray_image = cv2.warpPerspective(
    #     gray_image, cv2.getPerspectiveTransform(pt1, pt2), (side, side))

    # image = cv2.warpPerspective(
    #     image, cv2.getPerspectiveTransform(pt1, pt2), (side, side))

    # # Return images
    # return(image, gray_image)
    
    
    
    if c == 27:
        break


capture.release()
cv2.destroyAllWindows()