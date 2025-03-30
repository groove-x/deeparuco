import os
import argparse
import torch
from torch.utils.data import DataLoader
from impl.yolox_detector import YOLOXDetector
from impl.yolox_dataset import YOLOXDataset, collate_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir', help='Directory containing images and labels')
    parser.add_argument('run_name', help='Name of the training run')
    parser.add_argument('--model', default='s', choices=['s', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train on')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.run_name, exist_ok=True)

    # Initialize datasets
    train_dataset = YOLOXDataset(
        image_dir=os.path.join(args.source_dir, 'train', 'images'),
        label_dir=os.path.join(args.source_dir, 'train', 'labels')
    )
    valid_dataset = YOLOXDataset(
        image_dir=os.path.join(args.source_dir, 'valid', 'images'),
        label_dir=os.path.join(args.source_dir, 'valid', 'labels')
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")

    # Initialize model
    model = YOLOXDetector(model_size=args.model)

    # Train model
    model.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        save_dir=args.run_name
    )

if __name__ == '__main__':
    main()