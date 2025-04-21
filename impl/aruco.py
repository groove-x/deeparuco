import numpy as np
import cv2
from random import random, randint

ARUCO_MARKER_SIZE = 5
ARUCO_DICTIONARY_SIZE = 50

_DICT_NAME = f"DICT_{ARUCO_MARKER_SIZE}X{ARUCO_MARKER_SIZE}_{ARUCO_DICTIONARY_SIZE}"
_DICT = getattr(cv2.aruco, _DICT_NAME)
_CV_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(_DICT)

def get_marker(id, size=512, border_width=1.0):
    # Create base marker in marker_size*marker_size
    base_size = ARUCO_MARKER_SIZE + 2
    marker = np.ones((base_size, base_size, 4), dtype=np.uint8) * 255
    marker[:,:,3] = 0  # Transparent background

    # Special cases for ID 50 and 51
    if id == ARUCO_DICTIONARY_SIZE:
        # Create a completely black marker
        marker = np.zeros((base_size, base_size, 4), dtype=np.uint8)
        marker[:,:,3] = 255  # Set alpha channel to opaque
    elif id == ARUCO_DICTIONARY_SIZE + 1:
        # Create a border-only marker
        marker = np.zeros((base_size, base_size, 4), dtype=np.uint8)
        marker[:,:,3] = 255  # Set alpha channel to opaque
        # Fill the center with white
        marker[1:base_size-1, 1:base_size-1] = (255, 255, 255, 255)
    else:
        # Normal ArUco marker generation
        marker = _CV_ARUCO_DICT.generateImageMarker(id, base_size)
        marker_rgba = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGRA)
        marker = marker_rgba

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

    # Resize the base marker to the target size
    resized_marker = cv2.resize(marker, (wo_border, wo_border), interpolation=cv2.INTER_NEAREST)
    canvas[(size - wo_border) // 2:(size + wo_border) // 2,
           (size - wo_border) // 2:(size + wo_border) // 2] = resized_marker

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