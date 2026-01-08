"""
Fine-tuning script for transfer learning on small high-quality datasets.

Two-stage approach:
  Stage 1 (Frozen Encoder): Lock encoder, train decoder only with normal LR
  Stage 2 (Gentle Thaw): Unlock all, train with very low LR

Usage:
  # Stage 1: Frozen encoder fine-tuning
  python finetune.py --stage 1 --checkpoint checkpoints/best_model.pth --data_dir data_finetune

  # Stage 2: Full fine-tuning with low LR (after Stage 1)
  python finetune.py --stage 2 --checkpoint checkpoints/finetune_stage1.pth --data_dir data_finetune
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from finetune_config import FinetuneConfig
from dataset import load_data_paths, split_dataset, GroutDataset
from model import build_model
from losses import CombinedLoss
from utils import calculate_metrics


def get_heavy_augmentation_transform(img_size: int):
    """
    Heavy augmentation for small datasets.
    Turns few images into many unique training samples.
    Uses settings from FinetuneConfig.
    """
    cfg = FinetuneConfig

    return A.Compose([
        A.Resize(img_size, img_size),

        # GEOMETRY - See image from all angles
        A.HorizontalFlip(p=cfg.HORIZONTAL_FLIP_PROB),
        A.VerticalFlip(p=cfg.VERTICAL_FLIP_PROB),
        A.RandomRotate90(p=cfg.ROTATE90_PROB),
        A.ShiftScaleRotate(
            shift_limit=cfg.SHIFT_LIMIT,
            scale_limit=cfg.SCALE_LIMIT,
            rotate_limit=cfg.ROTATE_LIMIT,
            border_mode=0,
            p=cfg.SHIFT_SCALE_ROTATE_PROB
        ),
        A.Perspective(scale=(0.02, 0.05), p=cfg.PERSPECTIVE_PROB),
        A.ElasticTransform(alpha=50, sigma=10, p=cfg.ELASTIC_PROB),

        # TEXTURE - See image in different conditions
        A.RandomBrightnessContrast(
            brightness_limit=cfg.BRIGHTNESS_LIMIT,
            contrast_limit=cfg.CONTRAST_LIMIT,
            p=cfg.BRIGHTNESS_CONTRAST_PROB
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=cfg.HUE_SAT_PROB
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=cfg.GAUSS_NOISE_PROB),
        A.GaussianBlur(blur_limit=(3, 5), p=cfg.GAUSS_BLUR_PROB),
        A.CLAHE(clip_limit=2.0, p=cfg.CLAHE_PROB),

        # COLOR variations
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
            p=cfg.COLOR_JITTER_PROB
        ),
        A.ToGray(p=cfg.TO_GRAY_PROB),

        # Normalize and convert
        A.Normalize(mean=Config.MEAN, std=Config.STD),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int):
    """Validation transform (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=Config.MEAN, std=Config.STD),
        ToTensorV2(),
    ])


def freeze_encoder(model):
    """Freeze encoder weights (Stage 1)."""
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
            frozen_count += 1
    print(f"Frozen {frozen_count} encoder parameters")

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} parameters ({100*trainable/total:.1f}%)")


def unfreeze_encoder(model):
    """Unfreeze all weights (Stage 2)."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen all {trainable:,} parameters")


def load_pretrained_model(checkpoint_path: str, device: str):
    """Load pre-trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    # Build model with same architecture
    model = build_model(
        encoder=Config.ENCODER,
        encoder_weights=None,  # Don't load ImageNet weights, we have our own
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_dice' in checkpoint:
            print(f"Previous best Dice: {checkpoint['best_dice']:.4f}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    return model


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_dice = 0
    total_iou = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Calculate metrics
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            metrics = calculate_metrics(preds, masks)

        total_loss += loss.item()
        total_dice += metrics['dice']
        total_iou += metrics['iou']

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dice': f"{metrics['dice']:.4f}",
        })

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()

    total_loss = 0
    total_dice = 0
    total_iou = 0

    for batch in tqdm(loader, desc="Validating"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, masks)

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        metrics = calculate_metrics(preds, masks)

        total_loss += loss.item()
        total_dice += metrics['dice']
        total_iou += metrics['iou']

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def finetune(args):
    """Main fine-tuning function."""
    device = Config.DEVICE
    print(f"Device: {device}")

    # Determine stage settings
    if args.stage == 1:
        epochs = args.epochs or FinetuneConfig.STAGE1_EPOCHS
        lr = args.lr or FinetuneConfig.STAGE1_LR
        batch_size = args.batch_size or FinetuneConfig.STAGE1_BATCH_SIZE
        stage_name = "stage1_frozen"
        print("\n" + "="*60)
        print("STAGE 1: FROZEN ENCODER FINE-TUNING")
        print("="*60)
    else:
        epochs = args.epochs or FinetuneConfig.STAGE2_EPOCHS
        lr = args.lr or FinetuneConfig.STAGE2_LR
        batch_size = args.batch_size or FinetuneConfig.STAGE2_BATCH_SIZE
        stage_name = "stage2_unfrozen"
        print("\n" + "="*60)
        print("STAGE 2: UNFROZEN FINE-TUNING (LOW LR)")
        print("="*60)

    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")

    # Setup data directories
    if args.data_dir:
        images_dir = os.path.join(args.data_dir, 'images')
        masks_dir = os.path.join(args.data_dir, 'masks')
    else:
        images_dir = FinetuneConfig.IMAGES_DIR
        masks_dir = FinetuneConfig.MASKS_DIR

    print(f"\nData directory: {args.data_dir or FinetuneConfig.DATA_DIR}")

    # Load data
    print("\nLoading fine-tuning data...")
    img_paths, mask_paths = load_data_paths(images_dir, masks_dir)
    print(f"Found {len(img_paths)} image-mask pairs")

    if len(img_paths) == 0:
        print(f"ERROR: No images found in {images_dir}")
        print("Please create the directory with your high-quality images and masks")
        return

    # Split data (no test set for fine-tuning)
    train_imgs, train_masks, val_imgs, val_masks, _, _ = split_dataset(
        img_paths, mask_paths,
        train_split=FinetuneConfig.TRAIN_SPLIT,
        val_split=FinetuneConfig.VAL_SPLIT,
        test_split=FinetuneConfig.TEST_SPLIT,
        random_seed=Config.RANDOM_SEED,
    )

    # Create datasets with heavy augmentation for training
    train_dataset = GroutDataset(
        train_imgs, train_masks,
        transform=get_heavy_augmentation_transform(Config.IMG_SIZE)
    )
    val_dataset = GroutDataset(
        val_imgs, val_masks,
        transform=get_val_transform(Config.IMG_SIZE)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Load pre-trained model
    model = load_pretrained_model(args.checkpoint, device)

    # Freeze/unfreeze based on stage
    if args.stage == 1:
        freeze_encoder(model)
    else:
        unfreeze_encoder(model)

    # Setup training
    criterion = CombinedLoss(
        dice_weight=Config.DICE_WEIGHT,
        bce_weight=Config.BCE_WEIGHT,
        connectivity_weight=0.0,  # Keep disabled for speed
    )

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=Config.WEIGHT_DECAY)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    scaler = GradScaler(enabled=Config.USE_AMP)

    # Training loop
    best_dice = 0.0
    patience_counter = 0

    print("\nStarting fine-tuning...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )

        # Validate
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_dice)

        # Print metrics
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = os.path.join(
                FinetuneConfig.OUTPUT_DIR,
                f"finetune_{stage_name}_{Config.get_run_name()}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'stage': args.stage,
                'config': {
                    'encoder': Config.ENCODER,
                    'img_size': Config.IMG_SIZE,
                }
            }, checkpoint_path)
            print(f"  Saved best model: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{FinetuneConfig.PATIENCE})")

        # Early stopping
        if patience_counter >= FinetuneConfig.PATIENCE:
            print(f"\nEarly stopping after {epoch} epochs")
            break

    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE")
    print("="*60)
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Model saved to: {checkpoint_path}")

    if args.stage == 1:
        print(f"\nNext step: Run Stage 2 with:")
        print(f"  python finetune.py --stage 2 --checkpoint {checkpoint_path} --data_dir {args.data_dir or FinetuneConfig.DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune grout segmentation model")

    parser.add_argument('--stage', type=int, choices=[1, 2], required=True,
                        help='Fine-tuning stage: 1=frozen encoder, 2=unfrozen low LR')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pre-trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory with images/ and masks/ subdirectories')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: 20 for stage 1, 10 for stage 2)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 1e-4 for stage 1, 1e-6 for stage 2)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: 16)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(FinetuneConfig.OUTPUT_DIR, exist_ok=True)

    finetune(args)


if __name__ == "__main__":
    main()
