import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class PCBAugmentedDataset(Dataset):
    """
    Enhanced PCB Dataset with full augmentation using torchvision
    Equivalent to albumentations but more robust
    """
    
    def __init__(self, images_dir, labels_dir, img_size=600, is_train=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.is_train = is_train
        
        # Get all image files
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.image_files.sort()
        
        # Class names mapping
        self.class_names = {
            0: 'mouse_bite',
            1: 'spur', 
            2: 'missing_hole',
            3: 'short',
            4: 'open_circuit',
            5: 'spurious_copper'
        }
        
        # Base transforms (always applied)
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Augmentation parameters
        self.aug_params = {
            'horizontal_flip_p': 0.5,
            'vertical_flip_p': 0.3,
            'rotation_degrees': 90,
            'rotation_p': 0.4,
            'brightness': 0.2,
            'contrast': 0.2,
            'colorjitter_p': 0.3,
            'gaussian_blur_p': 0.2,
            'gaussian_blur_sigma': (0.1, 2.0),
            'noise_p': 0.1,
            'noise_factor': 0.05
        }
        
    def __len__(self):
        return len(self.image_files)
    
    def apply_augmentation(self, image, boxes):
        """Apply augmentation to image and corresponding bounding boxes"""
        if not self.is_train:
            return image, boxes
            
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Keep track of applied transforms for bbox adjustment
        applied_transforms = []
        
        # 1. Horizontal Flip
        if random.random() < self.aug_params['horizontal_flip_p']:
            image = TF.hflip(image)
            applied_transforms.append('hflip')
            # Adjust bounding boxes
            for box in boxes:
                box[1] = 1.0 - box[1]  # x_center = 1 - x_center
                
        # 2. Vertical Flip  
        if random.random() < self.aug_params['vertical_flip_p']:
            image = TF.vflip(image)
            applied_transforms.append('vflip')
            # Adjust bounding boxes
            for box in boxes:
                box[2] = 1.0 - box[2]  # y_center = 1 - y_center
                
        # 3. Random Rotation (90 degree multiples for PCB)
        if random.random() < self.aug_params['rotation_p']:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            applied_transforms.append(f'rotate_{angle}')
            
            # Adjust bounding boxes for rotation
            for box in boxes:
                x, y = box[1], box[2]
                if angle == 90:
                    box[1], box[2] = 1.0 - y, x
                elif angle == 180:
                    box[1], box[2] = 1.0 - x, 1.0 - y
                elif angle == 270:
                    box[1], box[2] = y, 1.0 - x
                    
        # 4. Color Jitter (brightness/contrast for grayscale)
        if random.random() < self.aug_params['colorjitter_p']:
            brightness_factor = 1.0 + random.uniform(-self.aug_params['brightness'], 
                                                    self.aug_params['brightness'])
            contrast_factor = 1.0 + random.uniform(-self.aug_params['contrast'], 
                                                  self.aug_params['contrast'])
            
            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)
            applied_transforms.append('colorjitter')
            
        # 5. Gaussian Blur
        if random.random() < self.aug_params['gaussian_blur_p']:
            sigma = random.uniform(*self.aug_params['gaussian_blur_sigma'])
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
            applied_transforms.append('blur')
            
        return image, boxes
    
    def add_noise(self, tensor):
        """Add gaussian noise to tensor"""
        if random.random() < self.aug_params['noise_p']:
            noise = torch.randn_like(tensor) * self.aug_params['noise_factor']
            tensor = tensor + noise
            tensor = torch.clamp(tensor, -1, 1)  # Keep in normalized range
        return tensor
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load as grayscale
        image = Image.open(img_path).convert('L')
        
        # Load corresponding label
        label_name = img_name.replace('_600.jpg', '_256.txt')
        label_path = os.path.join(self.labels_dir, label_name)
        
        # Parse bounding boxes
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            x, y, w, h = map(float, parts[1:5])
                            boxes.append([cls, x, y, w, h])
        
        # Apply augmentation (modifies boxes in-place if training)
        image, boxes = self.apply_augmentation(image, boxes)
        
        # Apply base transforms
        image = self.base_transform(image)
        
        # Add noise (after normalization)
        if self.is_train:
            image = self.add_noise(image)
        
        # Convert boxes to tensor format
        max_objects = 50
        targets = torch.zeros(max_objects, 6)  # [class, x, y, w, h, mask]
        
        if boxes:
            for i, box in enumerate(boxes[:max_objects]):
                targets[i] = torch.tensor([box[0], box[1], box[2], box[3], box[4], 1.0])
        
        return image, targets, img_name


def create_enhanced_dataloaders(data_dir, batch_size=32, num_workers=12, img_size=600, pin_memory=True):
    """Create enhanced data loaders with full augmentation"""
    
    # Training dataset with augmentation
    train_dataset = PCBAugmentedDataset(
        images_dir=os.path.join(data_dir, 'train', 'images'),
        labels_dir=os.path.join(data_dir, 'train', 'labels'),
        img_size=img_size,
        is_train=True  # Enable augmentation
    )
    
    # Validation dataset without augmentation
    val_dataset = PCBAugmentedDataset(
        images_dir=os.path.join(data_dir, 'val', 'images'),
        labels_dir=os.path.join(data_dir, 'val', 'labels'),
        img_size=img_size,
        is_train=False  # No augmentation
    )
    
    # Test dataset without augmentation
    test_dataset = PCBAugmentedDataset(
        images_dir=os.path.join(data_dir, 'test', 'images'),
        labels_dir=os.path.join(data_dir, 'test', 'labels'),
        img_size=img_size,
        is_train=False  # No augmentation
    )
    
    # Create data loaders - Optimized for RTX A4000/3090
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader


# Quick test function
def test_augmentation():
    """Test the augmentation pipeline"""
    dataset = PCBAugmentedDataset(
        'pcb-defect-dataset/train/images',
        'pcb-defect-dataset/train/labels',
        is_train=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a few samples
    for i in range(3):
        image, targets, filename = dataset[i]
        valid_targets = targets[targets[:, 5] > 0]
        
        print(f"\nSample {i+1}: {filename}")
        print(f"Image shape: {image.shape}")
        print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"Valid targets: {len(valid_targets)}")
        
        if len(valid_targets) > 0:
            for j, target in enumerate(valid_targets[:2]):
                cls, x, y, w, h, mask = target
                print(f"  Target {j}: class={cls:.0f}, bbox=({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})")


if __name__ == "__main__":
    test_augmentation()