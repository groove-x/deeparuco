from argparse import ArgumentParser
from glob import glob
import cv2
import numpy as np
import pandas as pd
from os import makedirs
from os.path import exists, basename, join
from shutil import rmtree
from tqdm import tqdm

def calculate_median_luma(image):
    """Calculate median luma (brightness) of an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.median(gray) / 255.0

def process_image(image):
    """Rotate if needed and crop to 640x360."""
    height, width = image.shape[:2]
    
    # Rotate if height > width
    if height > width:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        height, width = width, height
    
    # Check if image is large enough
    if width < 640 or height < 360:
        return None
        
    # Center crop
    start_x = (width - 640) // 2
    start_y = (height - 360) // 2
    cropped = image[start_y:start_y+360, start_x:start_x+640]
    
    return cropped

def main():
    parser = ArgumentParser(description='Create source dataset from MSCOCO')
    parser.add_argument('coco_dir', help='Path to MSCOCO train2017 directory')
    parser.add_argument('output_dir', help='Output directory for processed images')
    parser.add_argument('--num_images', type=int, default=2500, 
                       help='Number of images to sample (default: 2500)')
    parser.add_argument('--num_bins', type=int, default=10,
                       help='Number of brightness bins (default: 10)')
    args = parser.parse_args()

    # Create/clear output directory
    if exists(args.output_dir):
        rmtree(args.output_dir)
    makedirs(args.output_dir)

    # Process all images and calculate brightness
    print("Processing images and calculating brightness...")
    image_data = []
    
    for img_path in tqdm(glob(join(args.coco_dir, '*.jpg'))):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        processed = process_image(img)
        if processed is not None:
            brightness = calculate_median_luma(processed)
            image_data.append({
                'path': img_path,
                'brightness': brightness
            })

    # Create DataFrame and sort by brightness
    df = pd.DataFrame(image_data)
    df = df.sort_values('brightness')

    # Create bins and sample equally from each
    images_per_bin = args.num_images // args.num_bins
    sampled_images = []
    
    for i in range(args.num_bins):
        bin_start = i * len(df) // args.num_bins
        bin_end = (i + 1) * len(df) // args.num_bins
        bin_df = df.iloc[bin_start:bin_end]
        sampled = bin_df.sample(n=images_per_bin, random_state=42)
        sampled_images.append(sampled)

    # Combine all sampled images
    final_df = pd.concat(sampled_images)

    # Save selected images and create brightness.csv
    print("Saving selected images...")
    for _, row in tqdm(final_df.iterrows()):
        img = cv2.imread(row['path'])
        processed = process_image(img)
        output_path = join(args.output_dir, basename(row['path']))
        cv2.imwrite(output_path, processed)

    # Save brightness.csv with relative paths
    final_df['path'] = final_df['path'].apply(lambda x: basename(x))
    final_df.to_csv(join(args.output_dir, 'brightness.csv'), index=False)
    
    print(f"Dataset created with {len(final_df)} images")
    print(f"Brightness range: {final_df['brightness'].min():.4f} to {final_df['brightness'].max():.4f}")

if __name__ == '__main__':
    main()
