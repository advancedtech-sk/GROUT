"""
Configuration module for mosaic grout segmentation training pipeline.
Centralizes all hyperparameters and paths.
"""
import os
import torch
from pathlib import Path
from datetime import datetime


class Config:
    """Central configuration for the grout segmentation project."""

    # ==================== PATHS ====================
    DATA_DIR = 'data_local'
    IMAGES_DIR = os.path.join(DATA_DIR, 'images')
    MASKS_DIR = os.path.join(DATA_DIR, 'masks')
    CHECKPOINT_DIR = 'checkpoints'
    EXPORT_DIR = 'exports'
    LOGS_DIR = 'logs'
    RESULTS_DIR = 'results'

    # ==================== DATA SPLIT ====================
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    RANDOM_SEED = 42

    # Files that must always be in training set (by filename, without path)
    # Example: ['1.png', '2.png', 'important_sample.jpg']
    FORCE_TRAIN_FILES = [] #['7.png']

    # ==================== TRAINING HYPERPARAMETERS ====================
    IMG_SIZE = 512
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Early stopping
    EARLY_STOPPING_PATIENCE = 15

    # Learning rate scheduler
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_MIN_LR = 1e-7

    # Gradient clipping
    GRADIENT_CLIP_VALUE = 1.0

    # ==================== MODEL ARCHITECTURE ====================
    ENCODER = 'efficientnet-b3'  # Options: resnet34, resnet50, efficientnet-b0/b3, mobilenet_v2
    ENCODER_WEIGHTS = 'imagenet'
    IN_CHANNELS = 3  # RGB
    NUM_CLASSES = 1  # Binary segmentation

    # ==================== LOSS WEIGHTS ====================
    DICE_WEIGHT = 0.5
    BCE_WEIGHT = 0.3
    CONNECTIVITY_WEIGHT = 0.0  # Disabled - CPU-based, very slow with large batches

    # ==================== AUGMENTATION ====================
    AUGMENT_PROB = 0.5
    ROTATE_LIMIT = 90
    SCALE_LIMIT = 0.2
    BRIGHTNESS_CONTRAST_LIMIT = 0.2
    NOISE_VAR_LIMIT = (10.0, 50.0)

    # Mask dilation for thicker grout lines
    MASK_DILATION_KERNEL = 0  # Set to 0 to disable dilation (use original mask thickness)

    # ==================== PATCH EXTRACTION ====================
    PATCH_SIZE = 512
    PATCH_OVERLAP = 64  # Overlap between patches for large images
    MIN_GROUT_RATIO = 0.01  # Minimum ratio of grout pixels in patch to include

    # ==================== INFERENCE & POST-PROCESSING ====================
    PREDICTION_THRESHOLD = 0.5
    MORPHOLOGICAL_KERNEL_SIZE = 3
    MIN_COMPONENT_SIZE = 50  # Remove small disconnected components

    # ==================== DEVICE & PERFORMANCE ====================
    # Auto-detect GPU availability
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 0  # DataLoader workers (0 = main process, faster on Windows)
    PIN_MEMORY = True  # Faster data transfer to GPU

    # Mixed precision training
    USE_AMP = True  # Automatic Mixed Precision

    # ==================== LOGGING & CHECKPOINTING ====================
    SAVE_FREQUENCY = 5  # Save sample predictions every N epochs
    LOG_FREQUENCY = 10  # Log metrics every N batches
    SAVE_BEST_ONLY = True

    # TensorBoard
    USE_TENSORBOARD = True

    # ==================== NORMALIZATION ====================
    # ImageNet statistics for pretrained encoders
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.IMAGES_DIR,
            cls.MASKS_DIR,
            cls.CHECKPOINT_DIR,
            cls.EXPORT_DIR,
            cls.LOGS_DIR,
            cls.RESULTS_DIR,
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"Device: {cls.DEVICE}")
        print(f"Image Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Encoder: {cls.ENCODER}")
        print(f"Loss Weights - Dice: {cls.DICE_WEIGHT}, BCE: {cls.BCE_WEIGHT}, Connectivity: {cls.CONNECTIVITY_WEIGHT}")
        print(f"Data Split - Train: {cls.TRAIN_SPLIT}, Val: {cls.VAL_SPLIT}, Test: {cls.TEST_SPLIT}")
        print(f"Mixed Precision: {cls.USE_AMP}")
        print("=" * 60)

    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if hasattr(args, 'img_size') and args.img_size:
            cls.IMG_SIZE = args.img_size
            cls.PATCH_SIZE = args.img_size
        if hasattr(args, 'batch_size') and args.batch_size:
            cls.BATCH_SIZE = args.batch_size
        if hasattr(args, 'epochs') and args.epochs:
            cls.NUM_EPOCHS = args.epochs
        if hasattr(args, 'lr') and args.lr:
            cls.LEARNING_RATE = args.lr
        if hasattr(args, 'encoder') and args.encoder:
            cls.ENCODER = args.encoder

    @classmethod
    def get_encoder_short_name(cls):
        """Get short encoder name for filenames."""
        encoder_map = {
            'resnet34': 'res34',
            'resnet50': 'res50',
            'efficientnet-b0': 'effb0',
            'mobilenet_v2': 'mobv2',
        }
        return encoder_map.get(cls.ENCODER, cls.ENCODER.replace('-', ''))

    @classmethod
    def get_timestamp(cls):
        """Get current timestamp in YYYY_MM_DD format."""
        return datetime.now().strftime('%Y_%m_%d')

    @classmethod
    def get_run_name(cls):
        """Get run name: encoder_date (e.g., res34_2025_10_20)."""
        return f"{cls.get_encoder_short_name()}_{cls.get_timestamp()}"

    @classmethod
    def get_model_filename(cls, prefix='best_model'):
        """Get timestamped model filename."""
        return f"{prefix}_{cls.get_run_name()}.pth"

    @classmethod
    def get_result_filename(cls, base_name, suffix, extension='png'):
        """
        Get timestamped result filename.

        Args:
            base_name: Base name (e.g., 'image2')
            suffix: Suffix (e.g., 'overlay', 'mask')
            extension: File extension (default: 'png')

        Returns:
            Filename like 'image2_overlay_res34_2025_10_20.png'
        """
        return f"{base_name}_{suffix}_{cls.get_run_name()}.{extension}"
