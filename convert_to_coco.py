import json
import os
import random
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def get_bbox_from_corners(corners: List[List[float]]) -> List[float]:
    """Convert corner points to bounding box [x, y, width, height]."""
    corners = np.array(corners)
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def convert_to_coco_format(
    input_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    seed: int = 42
) -> None:
    """
    Convert ArUco JSON annotations to MS COCO format with train/val split.

    Args:
        input_dir: Directory containing input images and JSON annotations
        output_dir: Directory to save COCO format JSON files
        val_split: Fraction of data to use for validation (default: 0.2)
        seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize COCO format data structures
    train_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "aruco_marker"}]
    }
    val_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "aruco_marker"}]
    }

    # Get all JSON files
    json_files = glob(os.path.join(input_dir, "*.json"))
    random.shuffle(json_files)

    # Split into train and validation sets
    split_idx = int(len(json_files) * (1 - val_split))
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    # Process train files
    image_id = 1
    annotation_id = 1

    for json_file in train_files:
        # Read image to get dimensions
        image_path = json_file.replace(".json", ".jpg")
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Add image info
        image_info = {
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        }
        train_data["images"].append(image_info)

        # Read and process annotations
        with open(json_file, "r") as f:
            data = json.load(f)

        for marker in data["markers"]:
            bbox = get_bbox_from_corners(marker["corners"])
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "aruco_id": marker["id"],
                "rotation": marker["rot"]
            }
            train_data["annotations"].append(annotation)
            annotation_id += 1

        image_id += 1

    # Process validation files
    for json_file in val_files:
        # Read image to get dimensions
        image_path = json_file.replace(".json", ".jpg")
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Add image info
        image_info = {
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        }
        val_data["images"].append(image_info)

        # Read and process annotations
        with open(json_file, "r") as f:
            data = json.load(f)

        for marker in data["markers"]:
            bbox = get_bbox_from_corners(marker["corners"])
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "aruco_id": marker["id"],
                "rotation": marker["rot"]
            }
            val_data["annotations"].append(annotation)
            annotation_id += 1

        image_id += 1

    # Save COCO format JSON files
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(output_dir, "val.json"), "w") as f:
        json.dump(val_data, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ArUco JSON annotations to MS COCO format")
    parser.add_argument("input_dir", help="Directory containing input images and JSON annotations")
    parser.add_argument("output_dir", help="Directory to save COCO format JSON files")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    convert_to_coco_format(args.input_dir, args.output_dir, args.val_split, args.seed)