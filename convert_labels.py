import os
import json
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def convert_corners_to_yolo(corners, img_width, img_height):
    """Convert ArUco marker corners to YOLO format (x_center, y_center, width, height)"""
    corners = np.array(corners)
    x_min = np.min(corners[:, 0])
    y_min = np.min(corners[:, 1])
    x_max = np.max(corners[:, 0])
    y_max = np.max(corners[:, 1])
    
    # Convert to relative coordinates
    x_center = (x_min + x_max) / (2 * img_width)
    y_center = (y_min + y_max) / (2 * img_height)
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return [x_center, y_center, width, height]

def main():
    # Create output directories
    os.makedirs('data/train/labels', exist_ok=True)
    
    # Process each JSON file
    json_files = glob('data/train/*.json')
    for json_file in tqdm(json_files):
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Get corresponding image
        img_file = os.path.join('data/train/images', 
                               os.path.basename(json_file).replace('.json', '.jpg'))
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: Could not read image {img_file}")
            continue
            
        img_height, img_width = img.shape[:2]
        
        # Create YOLO label file
        label_file = os.path.join('data/train/labels', 
                                 os.path.basename(json_file).replace('.json', '.txt'))
        
        with open(label_file, 'w') as f:
            for marker in data['markers']:
                # Convert corners to YOLO format
                bbox = convert_corners_to_yolo(marker['corners'], img_width, img_height)
                
                # Write label (class_id x_center y_center width height)
                # We use 0 as class_id since we only have one class (marker)
                f.write(f"0 {' '.join([str(x) for x in bbox])}\n")

if __name__ == '__main__':
    main() 