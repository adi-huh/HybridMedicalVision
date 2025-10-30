"""
Dataset loader optimized for TorchXRayVision models

Key differences from standard loaders:
- Can handle grayscale or RGB
- Proper normalization for medical images
- TorchXRayVision-compatible preprocessing
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MedicalChestXrayDataset(Dataset):
    """
    Dataset for chest X-rays compatible with TorchXRayVision
    """
    
    def __init__(self, data_dir, split='train', use_augmentation=True, grayscale=False):
        self.data_dir = data_dir
        self.split = split
        self.use_augmentation = use_augmentation and (split == 'train')
        self.grayscale = grayscale
        self.image_paths = []
        self.labels = []
        
        # Load data
        split_dir = os.path.join(data_dir, 'chest_xray', 'chest_xray', split)
        
        for label_idx, label_name in enumerate(['NORMAL', 'PNEUMONIA']):
            label_dir = os.path.join(split_dir, label_name)
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(label_dir, img_name))
                        self.labels.append(label_idx)
        
        self._setup_transforms()
        
        print(f"{split.upper()} Dataset:")
        print(f"  Total: {len(self.image_paths)}")
        print(f"  Normal: {self.labels.count(0)}")
        print(f"  Pneumonia: {self.labels.count(1)}")
        print(f"  Grayscale: {self.grayscale}")
    
    def _setup_transforms(self):
        """Setup transforms compatible with TorchXRayVision"""
        
        if self.use_augmentation:
            # Training augmentation
            self.transform = A.Compose([
                # Resize
                A.Resize(224, 224),
                
                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
                
                # Contrast enhancement (critical for X-rays)
                A.CLAHE(clip_limit=2.0, p=0.8),
                
                # Pixel-level transforms
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                ], p=0.5),
                
                # Noise (simulate different X-ray machines)
                A.OneOf([
                    A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.3),
                
                # Normalize to [-1024, 1024] range (TorchXRayVision standard)
                A.Normalize(mean=[0.5], std=[0.5]),  # Normalizes to [-1, 1]
                ToTensorV2()
            ])
        else:
            # Validation/Test - minimal preprocessing
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.CLAHE(clip_limit=2.0, p=1.0),
                # Normalize to [-1024, 1024] range
                A.Normalize(mean=[0.5], std=[0.5]),  # Normalizes to [-1, 1]
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        
        # Convert to grayscale if needed
        if self.grayscale:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.expand_dims(image, axis=-1)  # Add channel dimension
        else:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image = transformed['image']
        
        # Ensure correct number of channels
        if self.grayscale:
            if image.shape[0] != 1:
                image = image.mean(dim=0, keepdim=True)  # Convert to single channel
        else:
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)  # Convert to 3 channels
        
        return image, label


def get_balanced_sampler(dataset):
    """Create weighted sampler for class imbalance"""
    labels = dataset.labels
    class_counts = [labels.count(0), labels.count(1)]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def get_medical_dataloaders(data_dir, batch_size=32, num_workers=4, grayscale=False):
    """
    Create dataloaders for medical pretrained models
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        grayscale: Whether to use grayscale images (True for some medical models)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Detect device and adjust num_workers
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MPS (Apple Silicon) has issues with num_workers > 0, so disable it
    if device == 'mps':
        num_workers = 0
        pin_memory = False
        print("MPS detected - disabling num_workers and pin_memory for stability")
    else:
        pin_memory = True
    
    # Create datasets
    train_dataset = MedicalChestXrayDataset(
        data_dir, split='train', use_augmentation=True, grayscale=grayscale
    )
    val_dataset = MedicalChestXrayDataset(
        data_dir, split='val', use_augmentation=False, grayscale=grayscale
    )
    test_dataset = MedicalChestXrayDataset(
        data_dir, split='test', use_augmentation=False, grayscale=grayscale
    )
    
    # Create balanced sampler for training
    train_sampler = get_balanced_sampler(train_dataset)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    
    return train_loader, val_loader, test_loader


def visualize_batch(dataloader, num_images=8):
    """Visualize a batch from dataloader"""
    import matplotlib.pyplot as plt
    
    images, labels = next(iter(dataloader))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx in range(min(num_images, len(images))):
        img = images[idx]
        
        # Handle grayscale vs RGB
        if img.shape[0] == 1:
            img = img.squeeze(0).numpy()
            axes[idx].imshow(img, cmap='gray')
        else:
            img = img.permute(1, 2, 0).numpy()
            axes[idx].imshow(img)
        
        label_name = 'PNEUMONIA' if labels[idx] == 1 else 'NORMAL'
        axes[idx].set_title(label_name, fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_sample.png', dpi=150, bbox_inches='tight')
    print("✓ Sample batch saved to 'dataset_sample.png'")
    plt.close()


if __name__ == "__main__":
    # Test the dataset
    print("Testing Medical Dataset Loader...")
    print("="*60)
    
    train_loader, val_loader, test_loader = get_medical_dataloaders(
        data_dir='data',
        batch_size=16,
        grayscale=False  # Set to True for grayscale models
    )
    
    print("\n" + "="*60)
    print("Visualizing sample batch...")
    visualize_batch(train_loader)
    
    print("\n" + "="*60)
    print("Dataset loading test complete!")