import os
import random
import shutil
from glob import glob

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Get all image files
    image_files = glob('data/flyingaruco/*.jpg')
    
    # Shuffle the files
    random.shuffle(image_files)
    
    # Split into train (80%) and validation (20%) sets
    split_idx = int(len(image_files) * 0.8)
    train_images = image_files[:split_idx]
    valid_images = image_files[split_idx:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(valid_images)}")
    
    # Move files to their respective directories
    for img_path in train_images:
        # Get corresponding label file
        label_path = img_path.replace('.jpg', '.txt')
        
        # Get base names
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        
        # Move files
        shutil.copy2(img_path, os.path.join('data/flyingaruco/train/images', img_name))
        shutil.copy2(label_path, os.path.join('data/flyingaruco/train/labels', label_name))
    
    for img_path in valid_images:
        # Get corresponding label file
        label_path = img_path.replace('.jpg', '.txt')
        
        # Get base names
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        
        # Move files
        shutil.copy2(img_path, os.path.join('data/flyingaruco/valid/images', img_name))
        shutil.copy2(label_path, os.path.join('data/flyingaruco/valid/labels', label_name))
    
    print("\nDataset split complete!")

if __name__ == '__main__':
    main() 