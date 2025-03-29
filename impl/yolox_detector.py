import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from typing import List, Tuple, Optional

class YOLOXDetector:
    def __init__(self, model_size: str = 's', num_classes: int = 1):
        """
        Initialize YOLOX detector
        Args:
            model_size: Size of the model ('s', 'm', 'l', 'x')
            num_classes: Number of classes (default: 1 for marker detection)
        """
        self.model_size = model_size
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self) -> nn.Module:
        """Build YOLOX model architecture"""
        # Basic YOLOX architecture components
        class YOLOXBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
                else:
                    self.shortcut = nn.Identity()
                    
            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                return F.relu(out)
        
        # Define model architecture based on size
        if self.model_size == 's':
            channels = [32, 64, 128, 256, 512]
            num_blocks = [3, 4, 6, 3]
        elif self.model_size == 'm':
            channels = [32, 64, 128, 256, 512]
            num_blocks = [6, 8, 12, 6]
        elif self.model_size == 'l':
            channels = [32, 64, 128, 256, 512]
            num_blocks = [8, 12, 16, 8]
        else:  # x
            channels = [32, 64, 128, 256, 512]
            num_blocks = [10, 16, 20, 10]
            
        # Build model
        model = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Add backbone blocks
        for i in range(4):
            model.add_module(f'layer{i+1}', nn.Sequential(
                *[YOLOXBlock(channels[i], channels[i+1], 2 if j == 0 else 1)
                  for j in range(num_blocks[i])]
            ))
            
        # Add detection head
        model.add_module('det_head', nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1] * 2, 1),
            nn.BatchNorm2d(channels[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[-1] * 2, (5 + self.num_classes) * 3, 1)
        ))
        
        return model
    
    def train(self, train_loader: DataLoader, num_epochs: int = 100, 
              learning_rate: float = 0.01, device: str = 'cuda'):
        """
        Train the model
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model.to(device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = self._build_loss()
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = images.to(device)
                targets = [t.to(device) for t in targets]  # Move each target tensor to device
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
    
    def _build_loss(self) -> nn.Module:
        """Build YOLOX loss function"""
        return YOLOXLoss(self.num_classes)
    
    def predict(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple[float, float, float, float]]:
        """
        Predict bounding boxes for an image
        Args:
            image: Input image (numpy array)
            conf_threshold: Confidence threshold for detections
        Returns:
            List of bounding boxes (x1, y1, x2, y2)
        """
        self.model.eval()
        with torch.no_grad():
            # Preprocess image
            img = cv2.resize(image, (640, 640))
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
            
            # Get predictions
            outputs = self.model(img)
            
            # Post-process predictions
            boxes = self._post_process(outputs, conf_threshold)
            
            # Scale boxes to original image size
            h, w = image.shape[:2]
            boxes = [(box[0] * w/640, box[1] * h/640, 
                     box[2] * w/640, box[3] * h/640) for box in boxes]
            
            return boxes
    
    def _post_process(self, outputs: torch.Tensor, conf_threshold: float) -> List[Tuple[float, float, float, float]]:
        """Post-process model outputs to get bounding boxes"""
        # Implementation of YOLOX post-processing
        # This is a simplified version - you might want to add more sophisticated NMS
        boxes = []
        for output in outputs:
            # Reshape output to (num_anchors, 5 + num_classes)
            output = output.view(-1, 5 + self.num_classes)
            
            # Get confidence scores
            conf = output[:, 4]
            mask = conf > conf_threshold
            
            if mask.any():
                # Get boxes with high confidence
                box_outputs = output[mask]
                boxes.extend(box_outputs[:, :4].tolist())
        
        return boxes

class YOLOXLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, predictions: torch.Tensor, targets: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute YOLOX loss
        Args:
            predictions: Model predictions
            targets: List of ground truth target tensors
        Returns:
            Total loss
        """
        # Implementation of YOLOX loss
        # This is a simplified version - you might want to add more sophisticated loss components
        loss = 0
        
        # Process each target in the batch
        for pred, target in zip(predictions, targets):
            # Classification loss
            cls_loss = F.binary_cross_entropy_with_logits(
                pred[..., 5:], target[..., 5:]
            )
            loss += cls_loss
            
            # Box regression loss
            box_loss = F.smooth_l1_loss(
                pred[..., :4], target[..., :4]
            )
            loss += box_loss
            
            # Objectness loss
            obj_loss = F.binary_cross_entropy_with_logits(
                pred[..., 4], target[..., 4]
            )
            loss += obj_loss
        
        return loss / len(targets)  # Average loss across batch 