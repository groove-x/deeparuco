import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from typing import List, Tuple, Optional

def collate_fn(batch):
    """
    Custom collate function to handle batches with different numbers of targets
    Args:
        batch: List of tuples (image, target)
    Returns:
        Tuple of (batched_images, batched_targets)
    """
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    
    # Stack images (they are all the same size)
    images = torch.stack(images, dim=0)  # [batch_size, channels, height, width]
    print(f"Batch image shape: {images.shape}")  # Debug print
    
    # Return as is (don't stack targets as they have different sizes)
    return images, targets

class YOLOXDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str, transform: Optional[bool] = True, 
                 img_size: int = 640):
        """
        Initialize YOLOX dataset
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing labels
            transform: Whether to apply data augmentation
            img_size: Target image size (both width and height)
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample
        Args:
            idx: Index of the sample
        Returns:
            Tuple of (image, target)
            - image: [3, H, W] normalized image tensor
            - target: [num_objects, 5] tensor where each row is [class_id, x_center, y_center, width, height]
        """
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        h0, w0 = image.shape[:2]
        
        # Resize image
        scale = min(self.img_size / w0, self.img_size / h0)
        new_w = int(w0 * scale)
        new_h = int(h0 * scale)
        image = cv2.resize(image, (new_w, new_h))
        
        # Create a square image with padding
        new_image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        offset_x = (self.img_size - new_w) // 2
        offset_y = (self.img_size - new_h) // 2
        new_image[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = image
        
        # Load labels
        label_path = os.path.join(self.label_dir, 
                                 os.path.splitext(self.image_files[idx])[0] + '.txt')
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # Each line format: class_id x_center y_center width height
                    label = [float(x) for x in line.strip().split()]
                    
                    # Adjust coordinates for resizing and padding
                    x_center = (label[1] * w0 * scale + offset_x) / self.img_size
                    y_center = (label[2] * h0 * scale + offset_y) / self.img_size
                    width = label[3] * w0 * scale / self.img_size
                    height = label[4] * h0 * scale / self.img_size
                    
                    # Clip coordinates to [0, 1]
                    x_center = np.clip(x_center, 0, 1)
                    y_center = np.clip(y_center, 0, 1)
                    width = np.clip(width, 0, 1)
                    height = np.clip(height, 0, 1)
                    
                    labels.append([label[0], x_center, y_center, width, height])
        
        # Convert to tensor
        image = torch.from_numpy(new_image).float().permute(2, 0, 1) / 255.0
        print(f"Single image shape: {image.shape}")  # Debug print
        
        if not labels:
            # If no labels, create a dummy target with no objects
            target = torch.zeros((0, 5), dtype=torch.float32)
        else:
            target = torch.tensor(labels, dtype=torch.float32)
        
        print(f"Target shape: {target.shape}")  # Debug print
        
        # Apply data augmentation if enabled
        if self.transform:
            image, target = self._apply_augmentation(image, target)
            
        return image, target
    
    def _apply_augmentation(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation
        Args:
            image: Input image tensor [3, H, W]
            target: Target tensor [num_objects, 5]
        Returns:
            Augmented image and target
        """
        # Random horizontal flip
        if np.random.random() < 0.5:
            image = torch.flip(image, [2])
            if len(target) > 0:
                # Flip x coordinates
                target[:, 1] = 1 - target[:, 1]
        
        # Random brightness and contrast
        if np.random.random() < 0.5:
            image = image * (0.8 + 0.4 * np.random.random())
            image = torch.clamp(image, 0, 1)
        
        return image, target 