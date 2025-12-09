"""
Custom image transformations for TIL Analyzer.
Provides train, validation, and UMAP transformation pipelines.
"""

from torchvision import transforms


class ImageTransformations:
    """
    Image transformation pipelines for the TIL survival analysis model.
    """
    
    def __init__(self, image_size=224):
        """
        Initialize transformation pipelines.
        
        Args:
            image_size: Target image size (default 224 for ResNet)
        """
        self.image_size = image_size
        
        # Training transformations with data augmentation
        self.train_transformations = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Validation transformations (no augmentation)
        self.validation_transformations = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # UMAP transformations (same as validation, for feature extraction)
        self.umap_transformations = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
