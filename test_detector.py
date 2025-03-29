import argparse
import os
import cv2
import numpy as np
from impl.yolox_detector import YOLOXDetector

def main():
    parser = argparse.ArgumentParser(description='Test YOLOX detector on images')
    parser.add_argument('model_dir', help='directory containing the trained model')
    parser.add_argument('image_dir', help='directory containing test images')
    parser.add_argument('--output_dir', '-o', help='directory to save results', default='results')
    parser.add_argument('--conf_threshold', '-c', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--device', '-d', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = YOLOXDetector(model_size='s', num_classes=1)
    model.model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pt')))
    model.model.to(args.device)
    model.model.eval()

    # Process each image
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for image_file in image_files:
        # Load image
        image_path = os.path.join(args.image_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get predictions
        boxes = model.predict(image, conf_threshold=args.conf_threshold)
        
        # Draw boxes
        h, w = image.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = box
            # Convert normalized coordinates to pixel coordinates
            x1, y1, x2, y2 = map(int, [x1 * w, y1 * h, x2 * w, y2 * h])
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save result
        output_path = os.path.join(args.output_dir, f'pred_{image_file}')
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f'Processed {image_file}: Found {len(boxes)} markers')

if __name__ == '__main__':
    main() 