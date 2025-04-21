import json
import os
import random
import shutil
from glob import glob
from typing import List, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm


def get_bbox_from_corners(corners: List[List[float]]) -> List[float]:
    """Convert corner points to bounding box [x, y, width, height]."""
    corners = np.array(corners)
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def process_image(image_path: str, output_dir: str, split: str) -> tuple[str, int, int]:
    """Process an image and copy it to the appropriate directory.

    Args:
        image_path: Path to the input image
        output_dir: Base output directory
        split: Either 'train' or 'valid'

    Returns:
        Tuple of (image_filename, width, height)
    """
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Copy image to appropriate directory
    image_filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_dir, "images", split, image_filename)
    shutil.copy2(image_path, output_image_path)

    return image_filename, width, height


def process_annotations(json_file: str, image_id: int, annotation_id: int) -> tuple[List[Dict[str, Any]], int]:
    """Process annotations from a JSON file.

    Args:
        json_file: Path to the JSON annotation file
        image_id: Current image ID
        annotation_id: Starting annotation ID

    Returns:
        Tuple of (annotations, next_annotation_id)
    """
    annotations = []
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
        annotations.append(annotation)
        annotation_id += 1

    return annotations, annotation_id


def process_files(
    json_files: List[str],
    output_dir: str,
    split: str,
    data: Dict[str, Any],
    start_image_id: int,
    start_annotation_id: int
) -> tuple[int, int]:
    """Process a set of JSON files and update the COCO data structure.

    Args:
        json_files: List of JSON files to process
        output_dir: Base output directory
        split: Either 'train' or 'valid'
        data: COCO data structure to update
        start_image_id: Starting image ID
        start_annotation_id: Starting annotation ID

    Returns:
        Tuple of (next_image_id, next_annotation_id)
    """
    image_id = start_image_id
    annotation_id = start_annotation_id

    for json_file in tqdm(json_files, desc=f"Processing {split} files"):
        # Read image to get dimensions
        image_path = json_file.replace(".json", ".jpg")
        if not os.path.exists(image_path):
            continue

        image_filename, width, height = process_image(image_path, output_dir, split)

        # Add image info
        image_info = {
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        }
        data["images"].append(image_info)

        # Process annotations
        annotations, annotation_id = process_annotations(json_file, image_id, annotation_id)
        data["annotations"].extend(annotations)

        image_id += 1

    return image_id, annotation_id


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
        output_dir: Directory to save COCO format JSON files and images
        val_split: Fraction of data to use for validation (default: 0.2)
        seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)

    # Create output directory structure
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "valid"), exist_ok=True)

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

    print(f"Processing {len(train_files)} training files and {len(val_files)} validation files")

    # Process train files
    image_id, annotation_id = process_files(
        train_files, output_dir, "train", train_data, 1, 1
    )

    # Process validation files
    process_files(
        val_files, output_dir, "valid", val_data, image_id, annotation_id
    )

    print("Saving annotation files...")
    # Save COCO format JSON files
    with open(os.path.join(output_dir, "annotations", "train_annotations.json"), "w") as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(output_dir, "annotations", "valid_annotations.json"), "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"Done! Processed {len(train_data['images'])} training images and {len(val_data['images'])} validation images")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ArUco JSON annotations to MS COCO format")
    parser.add_argument("input_dir", help="Directory containing input images and JSON annotations")
    parser.add_argument("output_dir", help="Directory to save COCO format JSON files and images")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    convert_to_coco_format(args.input_dir, args.output_dir, args.val_split, args.seed)