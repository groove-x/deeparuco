import numpy as np
import cv2
from random import random, randint

def get_marker(id, size=512, border_width=1.0):
    # Create the dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

    # Generate the marker
    marker = aruco_dict.generateImageMarker(id, size)

    # Convert to RGBA
    marker_rgba = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGRA)

    # Create canvas with transparent background
    canvas = np.ones((size, size, 4), dtype=np.uint8) * 255
    canvas[:,:,3] = 0

    # Calculate border sizes
    center = size // 2
    bg_size = int(size - 2 * (size / 10) * (1 - border_width))
    wo_border = int(size - 2 * (size / 10))

    # Add white background with border
    canvas[center - bg_size//2:center + bg_size//2,
           center - bg_size//2:center + bg_size//2, 3] = 255

    # Add marker
    canvas[(size - wo_border) // 2:(size + wo_border) // 2,
           (size - wo_border) // 2:(size + wo_border) // 2] = \
           cv2.resize(marker_rgba, (wo_border, wo_border), interpolation=cv2.INTER_NEAREST)

    # Calculate corner positions
    corners = [[(size - wo_border) // 2, (size - wo_border) // 2],
               [(size - wo_border) // 2, (size + wo_border) // 2 - 1],
               [(size + wo_border) // 2 - 1, (size + wo_border) // 2 - 1],
               [(size + wo_border) // 2 - 1, (size - wo_border) // 2]]

    return canvas, corners

if __name__ == '__main__':
    # Generate a random marker from DICT_5X5_50 (0-49)
    marker, corners = get_marker(randint(0, 49), border_width=random())
    for c in corners:
        cv2.circle(marker, (c[0], c[1]), 5, (0, 255, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.imwrite('test_aruco.png', marker)