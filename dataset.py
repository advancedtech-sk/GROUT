"""
Dataset module for mosaic grout segmentation.
Handles loading, splitting, augmentation, and patch extraction.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import random

from config import Config


class GroutDataset(Dataset):
    """Dataset for mosaic grout line segmentation."""

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[A.Compose] = None,
        patch_size: int = 512,
        extract_patches: bool = True,
        dilate_masks: bool = True,
    ):
        """
        Args:
            image_paths: List of paths to input images
            mask_paths: List of paths to corresponding masks
            transform: Albumentations transforms
            patch_size: Size of patches to extract
            extract_patches: If True, extract multiple patches from large images
            dilate_masks: If True, dilate masks to make grout lines thicker
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.patch_size = patch_size
        self.extract_patches = extract_patches
        self.dilate_masks = dilate_masks

        # Validate paths
        assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"

        # If extracting patches, create patch information
        self.patches = []
        if extract_patches:
            self._create_patch_info()
        else:
            # Use full images
            self.patches = [(i, None) for i in range(len(image_paths))]

        print(f"Dataset created with {len(self.patches)} samples")

    def _create_patch_info(self):
        """Create information about patches to extract from each image."""
        print("Creating patch information...")
        for idx, (img_path, mask_path) in enumerate(zip(self.image_paths, self.mask_paths)):
            # Read image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Cannot read image {img_path}, skipping...")
                continue

            h, w = img.shape[:2]

            # If image is smaller than patch size, use the full image
            if h <= self.patch_size and w <= self.patch_size:
                self.patches.append((idx, None))
                continue

            # Extract multiple patches
            stride = self.patch_size - Config.PATCH_OVERLAP

            for y in range(0, h - self.patch_size + 1, stride):
                for x in range(0, w - self.patch_size + 1, stride):
                    self.patches.append((idx, (y, x, y + self.patch_size, x + self.patch_size)))

            # Handle edges if image doesn't divide evenly
            if (h - self.patch_size) % stride != 0:
                y = h - self.patch_size
                for x in range(0, w - self.patch_size + 1, stride):
                    self.patches.append((idx, (y, x, y + self.patch_size, x + self.patch_size)))

            if (w - self.patch_size) % stride != 0:
                x = w - self.patch_size
                for y in range(0, h - self.patch_size + 1, stride):
                    self.patches.append((idx, (y, x, y + self.patch_size, x + self.patch_size)))

            # Bottom-right corner
            if (h - self.patch_size) % stride != 0 and (w - self.patch_size) % stride != 0:
                self.patches.append((idx, (h - self.patch_size, w - self.patch_size, h, w)))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        img_idx, patch_coords = self.patches[idx]

        # Load image and mask
        image = cv2.imread(self.image_paths[img_idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[img_idx], cv2.IMREAD_GRAYSCALE)

        # Check for loading errors
        if image is None or mask is None:
            raise ValueError(f"Failed to load image or mask at index {idx}")

        # Extract patch if specified
        if patch_coords is not None:
            y1, x1, y2, x2 = patch_coords
            image = image[y1:y2, x1:x2]
            mask = mask[y1:y2, x1:x2]

        # Resize if needed
        if image.shape[0] != self.patch_size or image.shape[1] != self.patch_size:
            image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)

        # Dilate mask to make grout lines thicker
        if self.dilate_masks and Config.MASK_DILATION_KERNEL > 0:
            kernel = np.ones((Config.MASK_DILATION_KERNEL, Config.MASK_DILATION_KERNEL), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Normalize mask to [0, 1]
        mask = (mask > 127).astype(np.float32)

        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return {
            'image': image,
            'mask': mask.unsqueeze(0) if len(mask.shape) == 2 else mask,
        }


def get_train_transform() -> A.Compose:
    """Get training augmentation pipeline."""
    return A.Compose([
        # Geometric transforms
        A.RandomRotate90(p=Config.AUGMENT_PROB),
        A.HorizontalFlip(p=Config.AUGMENT_PROB),
        A.VerticalFlip(p=Config.AUGMENT_PROB),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=Config.SCALE_LIMIT,
            rotate_limit=Config.ROTATE_LIMIT,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=Config.AUGMENT_PROB,
        ),

        # Elastic deformations
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3,
        ),

        # Color transforms (only affect image, not mask)
        A.RandomBrightnessContrast(
            brightness_limit=Config.BRIGHTNESS_CONTRAST_LIMIT,
            contrast_limit=Config.BRIGHTNESS_CONTRAST_LIMIT,
            p=Config.AUGMENT_PROB,
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.3,
        ),
        A.GaussNoise(var_limit=Config.NOISE_VAR_LIMIT, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Normalization
        A.Normalize(mean=Config.MEAN, std=Config.STD),
        ToTensorV2(),
    ])


def get_val_transform() -> A.Compose:
    """Get validation/test augmentation pipeline (no augmentation, only normalization)."""
    return A.Compose([
        A.Normalize(mean=Config.MEAN, std=Config.STD),
        ToTensorV2(),
    ])


def split_dataset(
    image_paths: List[str],
    mask_paths: List[str],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    force_train_files: List[str] = None,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        image_paths: List of image file paths
        mask_paths: List of mask file paths
        train_split: Fraction for training set
        val_split: Fraction for validation set
        test_split: Fraction for test set
        random_seed: Random seed for reproducibility
        force_train_files: List of filenames that must be in training set

    Returns:
        Tuple of (train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Separate forced training files from the rest
    force_train_files = force_train_files or []
    force_train_set = set(force_train_files)

    forced_train_imgs = []
    forced_train_masks = []
    remaining_imgs = []
    remaining_masks = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        filename = Path(img_path).name
        if filename in force_train_set:
            forced_train_imgs.append(img_path)
            forced_train_masks.append(mask_path)
        else:
            remaining_imgs.append(img_path)
            remaining_masks.append(mask_path)

    if forced_train_imgs:
        print(f"\nForced training files: {len(forced_train_imgs)}")
        for f in forced_train_imgs:
            print(f"  - {Path(f).name}")

    # If all files are forced to train, no splitting needed for remaining
    if not remaining_imgs:
        print(f"\nDataset split:")
        print(f"  Training:   {len(forced_train_imgs)} images (100.0%) - all forced")
        print(f"  Validation: 0 images (0.0%)")
        print(f"  Test:       0 images (0.0%)")
        return forced_train_imgs, forced_train_masks, [], [], [], []

    # First split: separate test set from remaining
    train_val_imgs, test_imgs, train_val_masks, test_masks = train_test_split(
        remaining_imgs,
        remaining_masks,
        test_size=test_split,
        random_state=random_seed,
    )

    # Second split: separate train and validation
    val_ratio = val_split / (train_split + val_split)
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        train_val_imgs,
        train_val_masks,
        test_size=val_ratio,
        random_state=random_seed,
    )

    # Add forced training files to training set
    train_imgs = forced_train_imgs + train_imgs
    train_masks = forced_train_masks + train_masks

    print(f"\nDataset split:")
    print(f"  Training:   {len(train_imgs)} images ({len(train_imgs)/len(image_paths)*100:.1f}%) [{len(forced_train_imgs)} forced]")
    print(f"  Validation: {len(val_imgs)} images ({len(val_imgs)/len(image_paths)*100:.1f}%)")
    print(f"  Test:       {len(test_imgs)} images ({len(test_imgs)/len(image_paths)*100:.1f}%)")

    return train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks


def load_data_paths(images_dir: str, masks_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load image and mask file paths from directories.

    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks

    Returns:
        Tuple of (image_paths, mask_paths)
    """
    # Supported image extensions
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    # Get all image files
    image_files = []
    for ext in image_exts:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))

    image_files = sorted(image_files)

    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")

    # Match masks to images
    image_paths = []
    mask_paths = []

    for img_path in image_files:
        # Try to find corresponding mask
        mask_name = img_path.stem + '.png'
        mask_path = Path(masks_dir) / mask_name

        if not mask_path.exists():
            # Try with original extension
            mask_name = img_path.name
            mask_path = Path(masks_dir) / mask_name

        if mask_path.exists():
            image_paths.append(str(img_path))
            mask_paths.append(str(mask_path))
        else:
            print(f"Warning: No mask found for {img_path.name}, skipping...")

    if len(image_paths) == 0:
        raise ValueError(f"No matching image-mask pairs found")

    print(f"\nFound {len(image_paths)} image-mask pairs")

    return image_paths, mask_paths


def create_dataloaders(
    train_imgs: List[str],
    train_masks: List[str],
    val_imgs: List[str],
    val_masks: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        train_imgs: Training image paths
        train_masks: Training mask paths
        val_imgs: Validation image paths
        val_masks: Validation mask paths
        batch_size: Batch size
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = GroutDataset(
        train_imgs,
        train_masks,
        transform=get_train_transform(),
        patch_size=Config.PATCH_SIZE,
        extract_patches=True,
        dilate_masks=True,
    )

    val_dataset = GroutDataset(
        val_imgs,
        val_masks,
        transform=get_val_transform(),
        patch_size=Config.PATCH_SIZE,
        extract_patches=False,  # Don't extract patches for validation
        dilate_masks=True,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"\nDataloaders created:")
    print(f"  Training batches:   {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")

    # Load paths
    img_paths, mask_paths = load_data_paths(Config.IMAGES_DIR, Config.MASKS_DIR)

    # Split dataset
    train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks = split_dataset(
        img_paths, mask_paths, force_train_files=Config.FORCE_TRAIN_FILES
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_imgs, train_masks, val_imgs, val_masks
    )

    # Test loading a batch
    print("\nTesting batch loading...")
    batch = next(iter(train_loader))
    print(f"Image batch shape: {batch['image'].shape}")
    print(f"Mask batch shape: {batch['mask'].shape}")
    print(f"Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    print(f"Mask range: [{batch['mask'].min():.3f}, {batch['mask'].max():.3f}]")
    print("\nDataset test completed successfully!")
