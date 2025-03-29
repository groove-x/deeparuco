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
            channels = [64, 128, 256, 512, 1024]
            num_blocks = [6, 8, 12, 6]
        elif self.model_size == 'l':
            channels = [64, 128, 256, 512, 1024]
            num_blocks = [8, 12, 16, 8]
        else:  # x
            channels = [64, 128, 256, 512, 1024]
            num_blocks = [10, 16, 20, 10]
            
        # Build model
        layers = []
        
        # Initial convolution
        layers.extend([
            nn.Conv2d(3, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        ])
        
        # Add backbone blocks
        in_channels = channels[0]
        for i in range(4):
            out_channels = channels[i + 1]
            for j in range(num_blocks[i]):
                stride = 2 if j == 0 else 1
                layers.append(YOLOXBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        
        # Add detection head
        layers.extend([
            nn.Conv2d(channels[-1], channels[-1] * 2, 1),
            nn.BatchNorm2d(channels[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[-1] * 2, (5 + self.num_classes) * 3, 1)
        ])
        
        return nn.Sequential(*layers)
    
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
        criterion.to(device)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (images, targets) in enumerate(train_loader):
                print(f"Input image shape: {images.shape}")  # Debug print
                images = images.to(device)
                targets = [t.to(device) for t in targets]  # Move each target tensor to device
                
                optimizer.zero_grad()
                outputs = self.model(images)
                # Reshape outputs to match target format
                B, _, H, W = outputs.shape
                outputs = outputs.view(B, 3, -1, H, W).permute(0, 1, 3, 4, 2)
                print(f"Output shape after reshape: {outputs.shape}")  # Debug print
                
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
            predictions: Model predictions [batch_size, 3, H, W, 5+num_classes]
            targets: List of ground truth target tensors [num_objects, 5]
        Returns:
            Total loss
        """
        total_loss = 0
        batch_size = predictions.shape[0]
        
        for i in range(batch_size):
            pred = predictions[i]  # [3, H, W, 5+num_classes]
            target = targets[i]    # [num_objects, 5]
            
            # Convert target to grid format
            grid_h, grid_w = pred.shape[1:3]
            target_grid = torch.zeros_like(pred)  # [3, H, W, 5+num_classes]
            
            if len(target) > 0:
                # Scale target coordinates to grid size
                target_xy = target[:, 1:3] * torch.tensor([grid_w, grid_h], device=pred.device)
                grid_xy = target_xy.long()  # Get grid cell indices
                
                # Assign targets to best matching anchor
                for t_idx in range(len(target)):
                    grid_x, grid_y = grid_xy[t_idx]
                    if grid_x < grid_w and grid_y < grid_h:
                        # Find best anchor based on aspect ratio
                        target_wh = target[t_idx, 3:5]
                        best_anchor = 0  # Simplified: using first anchor
                        
                        # Assign target to grid cell
                        target_grid[best_anchor, grid_y, grid_x, :5] = target[t_idx]
                        target_grid[best_anchor, grid_y, grid_x, 5:] = 1.0  # One-hot class
            
            # Compute losses
            # Box regression loss
            box_loss = F.smooth_l1_loss(
                pred[..., :4],
                target_grid[..., :4],
                reduction='sum'
            )
            
            # Objectness loss
            obj_loss = F.binary_cross_entropy_with_logits(
                pred[..., 4],
                target_grid[..., 4],
                reduction='sum'
            )
            
            # Classification loss
            cls_loss = F.binary_cross_entropy_with_logits(
                pred[..., 5:],
                target_grid[..., 5:],
                reduction='sum'
            )
            
            # Combine losses
            total_loss += (box_loss + obj_loss + cls_loss) / (grid_h * grid_w * 3)
        
        return total_loss / batch_size 