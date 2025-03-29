import argparse
import os
import torch
from torch.utils.data import DataLoader
from impl.yolox_detector import YOLOXDetector
from impl.yolox_dataset import YOLOXDataset

def main():
    parser = argparse.ArgumentParser(description='DeepArUco++ detector trainer with YOLOX')
    parser.add_argument('source_dir', help='where to find source images')
    parser.add_argument('run_name', help='directory of the resulting model')
    parser.add_argument('--model', '-m', help='model size (s, m, l, x)', default='s')
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--device', '-d', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.run_name, exist_ok=True)

    # Create dataset and dataloader
    train_dataset = YOLOXDataset(
        image_dir=os.path.join(args.source_dir, 'train/images'),
        label_dir=os.path.join(args.source_dir, 'train/labels')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = YOLOXDetector(model_size=args.model, num_classes=1)
    
    # Train model
    model.train(
        train_loader=train_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )

    # Save model
    torch.save(model.model.state_dict(), os.path.join(args.run_name, 'model.pt'))

if __name__ == '__main__':
    main()