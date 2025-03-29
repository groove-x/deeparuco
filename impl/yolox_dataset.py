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
    images = torch.stack(images, dim=0)
    
    # Return as is (don't stack targets as they have different sizes)
    return images, targets

class YOLOXDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str, transform: Optional[bool] = True):
        """
        Initialize YOLOX dataset
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing labels
            transform: Whether to apply data augmentation
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
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
        """
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = os.path.join(self.label_dir, 
                                 os.path.splitext(self.image_files[idx])[0] + '.txt')
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    label = [float(x) for x in line.strip().split()]
                    labels.append(label)
        
        if not labels:
            labels = [[0, 0, 0, 0, 0]]  # Default to no object
            
        # Convert to tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        target = torch.tensor(labels)
        
        # Apply data augmentation if enabled
        if self.transform:
            image, target = self._apply_augmentation(image, target)
            
        return image, target
    
    def _apply_augmentation(self, image: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation
        Args:
            image: Input image tensor
            target: Target tensor containing multiple labels
        Returns:
            Augmented image and target
        """
        # Random horizontal flip
        if np.random.random() < 0.5:
            image = torch.flip(image, [2])
            if len(target) > 0:
                target[:, 1] = 1 - target[:, 1]  # Flip x coordinate for all boxes
        
        # Random brightness and contrast
        if np.random.random() < 0.5:
            image = image * (0.8 + 0.4 * np.random.random())
            
        # Random rotation
        if np.random.random() < 0.5:
            angle = np.random.uniform(-10, 10)
            # Implement rotation logic here
            # This would require more complex handling of bounding boxes
            
        return image, target 