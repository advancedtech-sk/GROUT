"""
Main training script for mosaic grout segmentation.
"""
import os
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
try:
    # PyTorch 2.0+
    from torch.amp import autocast, GradScaler
except ImportError:
    # PyTorch 1.x
    from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random

from config import Config
from dataset import load_data_paths, split_dataset, create_dataloaders
from model import build_model, save_checkpoint, load_checkpoint, print_model_summary
from losses import CombinedLoss
from utils import calculate_metrics, visualize_batch, plot_training_history


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """Training manager for grout segmentation."""

    def __init__(self, config: Config):
        """
        Initialize trainer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.device = config.DEVICE

        # Create directories
        config.create_directories()

        # Set random seed
        set_seed(config.RANDOM_SEED)

        # Setup training log file
        self.run_name = config.get_run_name()
        self.log_file = os.path.join(config.LOGS_DIR, f'training_log_{self.run_name}.txt')
        self.log_to_file(f"Training started at {config.get_timestamp()}")
        self.log_to_file(f"Run name: {self.run_name}")
        self.log_to_file("=" * 60)

        # Load data
        print("\n" + "=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        self.image_paths, self.mask_paths = load_data_paths(
            config.IMAGES_DIR, config.MASKS_DIR
        )

        # Split dataset
        (
            self.train_imgs,
            self.train_masks,
            self.val_imgs,
            self.val_masks,
            self.test_imgs,
            self.test_masks,
        ) = split_dataset(
            self.image_paths,
            self.mask_paths,
            train_split=config.TRAIN_SPLIT,
            val_split=config.VAL_SPLIT,
            test_split=config.TEST_SPLIT,
            random_seed=config.RANDOM_SEED,
            force_train_files=config.FORCE_TRAIN_FILES,
        )

        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(
            self.train_imgs,
            self.train_masks,
            self.val_imgs,
            self.val_masks,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )

        # Build model
        print("\n" + "=" * 60)
        print("BUILDING MODEL")
        print("=" * 60)
        self.model = build_model(
            encoder=config.ENCODER,
            encoder_weights=config.ENCODER_WEIGHTS,
            in_channels=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
        )
        self.model = self.model.to(self.device)
        print_model_summary(self.model)

        # Loss function
        self.criterion = CombinedLoss(
            dice_weight=config.DICE_WEIGHT,
            bce_weight=config.BCE_WEIGHT,
            connectivity_weight=config.CONNECTIVITY_WEIGHT,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.LR_SCHEDULER_MIN_LR,
        )

        # Mixed precision training
        if config.USE_AMP:
            try:
                self.scaler = GradScaler('cuda' if config.DEVICE == 'cuda' else 'cpu')
            except TypeError:
                # Older PyTorch version
                self.scaler = GradScaler()
        else:
            self.scaler = None

        # TensorBoard
        self.writer = None
        if config.USE_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=config.LOGS_DIR)

        # Training state
        self.start_epoch = 0
        self.best_metric = 0.0
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'lr': [],
        }

    def log_to_file(self, message: str):
        """Log message to file and optionally print."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def log_config(self):
        """Log configuration to file."""
        self.log_to_file("\nCONFIGURATION:")
        self.log_to_file(f"  Device: {self.config.DEVICE}")
        self.log_to_file(f"  Image Size: {self.config.IMG_SIZE}x{self.config.IMG_SIZE}")
        self.log_to_file(f"  Batch Size: {self.config.BATCH_SIZE}")
        self.log_to_file(f"  Epochs: {self.config.NUM_EPOCHS}")
        self.log_to_file(f"  Learning Rate: {self.config.LEARNING_RATE}")
        self.log_to_file(f"  Weight Decay: {self.config.WEIGHT_DECAY}")
        self.log_to_file(f"  Encoder: {self.config.ENCODER}")
        self.log_to_file(f"  Loss Weights:")
        self.log_to_file(f"    - Dice: {self.config.DICE_WEIGHT}")
        self.log_to_file(f"    - BCE: {self.config.BCE_WEIGHT}")
        self.log_to_file(f"    - Connectivity: {self.config.CONNECTIVITY_WEIGHT}")
        self.log_to_file(f"  Data Split:")
        self.log_to_file(f"    - Train: {self.config.TRAIN_SPLIT}")
        self.log_to_file(f"    - Val: {self.config.VAL_SPLIT}")
        self.log_to_file(f"    - Test: {self.config.TEST_SPLIT}")
        self.log_to_file(f"  Augmentation:")
        self.log_to_file(f"    - Probability: {self.config.AUGMENT_PROB}")
        self.log_to_file(f"    - Rotate Limit: {self.config.ROTATE_LIMIT}")
        self.log_to_file(f"    - Scale Limit: {self.config.SCALE_LIMIT}")
        self.log_to_file(f"    - Mask Dilation: {self.config.MASK_DILATION_KERNEL}")
        self.log_to_file(f"  Mixed Precision: {self.config.USE_AMP}")
        self.log_to_file(f"  Early Stopping Patience: {self.config.EARLY_STOPPING_PATIENCE}")
        self.log_to_file("=" * 60 + "\n")

    def train_epoch(self, epoch: int) -> dict:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        all_metrics = []

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.NUM_EPOCHS} [Train]")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Forward pass with mixed precision
            device_type = 'cuda' if self.device == 'cuda' else 'cpu'
            # For CPU, disable AMP as it's not beneficial
            use_amp = self.config.USE_AMP and device_type == 'cuda'
            with autocast(device_type=device_type, enabled=use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP_VALUE
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP_VALUE
                )
                self.optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                metrics = calculate_metrics(probs, masks)
                all_metrics.append(metrics)

            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'dice': f'{metrics["dice"]:.4f}',
                'iou': f'{metrics["iou"]:.4f}',
            })

            # Log to TensorBoard
            if self.writer and batch_idx % self.config.LOG_FREQUENCY == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

        # Aggregate metrics
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        avg_metrics['loss'] = total_loss / len(self.train_loader)

        return avg_metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """
        Validate model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_metrics = []

        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config.NUM_EPOCHS} [Val]  ")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Forward pass
            device_type = 'cuda' if self.device == 'cuda' else 'cpu'
            # For CPU, disable AMP as it's not beneficial
            use_amp = self.config.USE_AMP and device_type == 'cuda'
            with autocast(device_type=device_type, enabled=use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # Calculate metrics
            probs = torch.sigmoid(outputs)
            metrics = calculate_metrics(probs, masks)
            all_metrics.append(metrics)

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'dice': f'{metrics["dice"]:.4f}',
                'iou': f'{metrics["iou"]:.4f}',
            })

            # Save sample visualizations
            if batch_idx == 0 and epoch % self.config.SAVE_FREQUENCY == 0:
                save_path = os.path.join(
                    self.config.RESULTS_DIR,
                    f'val_predictions_epoch_{epoch:03d}.png'
                )
                visualize_batch(images, masks, outputs, save_path)

        # Aggregate metrics
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        avg_metrics['loss'] = total_loss / len(self.val_loader)

        return avg_metrics

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        self.config.print_config()

        # Log configuration
        self.log_config()

        start_time = time.time()
        self.log_to_file(f"\nTraining started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        for epoch in range(self.start_epoch + 1, self.config.NUM_EPOCHS + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['lr'].append(current_lr)

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Metrics/Train_Dice', train_metrics['dice'], epoch)
                self.writer.add_scalar('Metrics/Val_Dice', val_metrics['dice'], epoch)
                self.writer.add_scalar('Metrics/Train_IoU', train_metrics['iou'], epoch)
                self.writer.add_scalar('Metrics/Val_IoU', val_metrics['iou'], epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            summary = f"\nEpoch {epoch}/{self.config.NUM_EPOCHS} Summary:"
            print(summary)
            self.log_to_file(summary)

            metrics_str = f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}"
            print(metrics_str)
            self.log_to_file(metrics_str)

            dice_str = f"  Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}"
            print(dice_str)
            self.log_to_file(dice_str)

            iou_str = f"  Train IoU:  {train_metrics['iou']:.4f} | Val IoU:  {val_metrics['iou']:.4f}"
            print(iou_str)
            self.log_to_file(iou_str)

            time_str = f"  LR: {current_lr:.2e} | Time: {epoch_time:.2f}s"
            print(time_str)
            self.log_to_file(time_str)

            # Save checkpoint
            is_best = val_metrics['dice'] > self.best_metric

            if is_best:
                self.best_metric = val_metrics['dice']
                self.epochs_without_improvement = 0

                # Save best model with timestamped name
                best_model_filename = self.config.get_model_filename('best_model')
                checkpoint_path = os.path.join(
                    self.config.CHECKPOINT_DIR,
                    best_model_filename
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.best_metric,
                    checkpoint_path,
                    self.scheduler,
                )
                best_msg = f"  *** New best model! Dice: {self.best_metric:.4f} ***"
                print(best_msg)
                self.log_to_file(best_msg)
                self.log_to_file(f"  Saved to: {best_model_filename}")
            else:
                self.epochs_without_improvement += 1

            # Save latest model with timestamped name
            latest_model_filename = self.config.get_model_filename('latest_model')
            checkpoint_path = os.path.join(
                self.config.CHECKPOINT_DIR,
                latest_model_filename
            )
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                self.best_metric,
                checkpoint_path,
                self.scheduler,
            )

            # Learning rate scheduler step
            self.scheduler.step(val_metrics['dice'])

            # Early stopping
            if self.epochs_without_improvement >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best Dice score: {self.best_metric:.4f}")
                break

            print("-" * 60)

        # Training complete
        total_time = time.time() - start_time
        completion_msg = "\n" + "=" * 60 + "\nTRAINING COMPLETE\n" + "=" * 60
        print(completion_msg)
        self.log_to_file(completion_msg)

        time_msg = f"Total training time: {total_time / 3600:.2f} hours ({total_time:.0f}s)"
        print(time_msg)
        self.log_to_file(time_msg)

        dice_msg = f"Best validation Dice: {self.best_metric:.4f}"
        print(dice_msg)
        self.log_to_file(dice_msg)

        best_model_filename = self.config.get_model_filename('best_model')
        model_path_msg = f"Best model saved to: {os.path.join(self.config.CHECKPOINT_DIR, best_model_filename)}"
        print(model_path_msg)
        self.log_to_file(model_path_msg)

        # Save training history plot with timestamped name
        history_filename = f'training_history_{self.run_name}.png'
        history_plot_path = os.path.join(self.config.RESULTS_DIR, history_filename)
        plot_training_history(self.history, history_plot_path)
        self.log_to_file(f"Training history saved to: {history_filename}")

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()

        return self.best_metric


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train U-Net for mosaic grout segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=None, help='Image size')

    # Model parameters
    parser.add_argument('--encoder', type=str, default=None,
                       choices=['resnet34', 'resnet50', 'efficientnet-b0', 'mobilenet_v2'],
                       help='Encoder architecture')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Data paths
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')

    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Update config from arguments
    Config.update_from_args(args)

    # Update data directory if specified
    if args.data_dir:
        Config.DATA_DIR = args.data_dir
        Config.IMAGES_DIR = os.path.join(args.data_dir, 'images')
        Config.MASKS_DIR = os.path.join(args.data_dir, 'masks')

    # Create trainer
    trainer = Trainer(Config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.model = load_checkpoint(trainer.model, args.resume, trainer.device)

        # Try to load optimizer and scheduler state
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'epoch' in checkpoint:
            trainer.start_epoch = checkpoint['epoch']
        if 'best_metric' in checkpoint:
            trainer.best_metric = checkpoint['best_metric']

    # Train
    best_dice = trainer.train()

    print(f"\nTraining finished! Best Dice score: {best_dice:.4f}")


if __name__ == "__main__":
    main()
