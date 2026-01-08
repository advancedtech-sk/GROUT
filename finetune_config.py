"""
Configuration for fine-tuning on small high-quality datasets.

Two-stage approach:
  Stage 1: Frozen encoder - train decoder only
  Stage 2: Unfrozen - fine-tune everything with very low LR
"""
import os


class FinetuneConfig:
    """Fine-tuning configuration - edit these values as needed."""

    # ==================== DATA PATHS ====================
    # Directory containing your high-quality fine-tuning data
    DATA_DIR = 'data_finetune'
    IMAGES_DIR = os.path.join(DATA_DIR, 'images')
    MASKS_DIR = os.path.join(DATA_DIR, 'masks')

    # ==================== DATA SPLIT ====================
    # For small datasets: more training, less validation, no test
    TRAIN_SPLIT = 0.85
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.0

    # ==================== STAGE 1: FROZEN ENCODER ====================
    # Goal: Train decoder while keeping encoder (EfficientNet) frozen
    # - Encoder already knows edges/textures from pre-training
    # - Decoder learns new mapping for your specific data
    STAGE1_EPOCHS = 20
    STAGE1_LR = 1e-4          # Normal LR is safe since encoder is frozen
    STAGE1_BATCH_SIZE = 16

    # ==================== STAGE 2: UNFROZEN (GENTLE THAW) ====================
    # Goal: Slightly adjust encoder to understand your specific features
    # - All weights trainable
    # - Very low LR to avoid catastrophic forgetting
    STAGE2_EPOCHS = 10
    STAGE2_LR = 1e-6          # 100x lower than Stage 1!
    STAGE2_BATCH_SIZE = 16

    # ==================== EARLY STOPPING ====================
    PATIENCE = 10

    # ==================== HEAVY AUGMENTATION ====================
    # For small datasets, heavy augmentation is critical
    # These settings turn 10 images into thousands of unique samples

    # Geometry transforms
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB = 0.5
    ROTATE90_PROB = 0.5
    SHIFT_SCALE_ROTATE_PROB = 0.8
    SHIFT_LIMIT = 0.1
    SCALE_LIMIT = 0.2
    ROTATE_LIMIT = 45
    PERSPECTIVE_PROB = 0.3
    ELASTIC_PROB = 0.3

    # Texture/color transforms
    BRIGHTNESS_CONTRAST_PROB = 0.8
    BRIGHTNESS_LIMIT = 0.3
    CONTRAST_LIMIT = 0.3
    HUE_SAT_PROB = 0.5
    GAUSS_NOISE_PROB = 0.3
    GAUSS_BLUR_PROB = 0.3
    CLAHE_PROB = 0.3
    COLOR_JITTER_PROB = 0.5
    TO_GRAY_PROB = 0.1         # Occasionally train on grayscale

    # ==================== OUTPUT ====================
    OUTPUT_DIR = 'checkpoints'

    @classmethod
    def print_config(cls, stage: int):
        """Print configuration for specified stage."""
        print("=" * 60)
        print(f"FINE-TUNING CONFIGURATION - STAGE {stage}")
        print("=" * 60)
        print(f"Data directory: {cls.DATA_DIR}")
        print(f"Train/Val split: {cls.TRAIN_SPLIT}/{cls.VAL_SPLIT}")

        if stage == 1:
            print(f"\nStage 1 (Frozen Encoder):")
            print(f"  Epochs: {cls.STAGE1_EPOCHS}")
            print(f"  Learning Rate: {cls.STAGE1_LR}")
            print(f"  Batch Size: {cls.STAGE1_BATCH_SIZE}")
            print(f"  Encoder: FROZEN (decoder only)")
        else:
            print(f"\nStage 2 (Unfrozen):")
            print(f"  Epochs: {cls.STAGE2_EPOCHS}")
            print(f"  Learning Rate: {cls.STAGE2_LR}")
            print(f"  Batch Size: {cls.STAGE2_BATCH_SIZE}")
            print(f"  All weights: TRAINABLE")

        print(f"\nEarly Stopping Patience: {cls.PATIENCE}")
        print("=" * 60)
