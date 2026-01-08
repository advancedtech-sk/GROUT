"""
Model definition for mosaic grout segmentation.
Uses U-Net architecture from segmentation_models_pytorch.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional

from config import Config


def build_model(
    encoder: str = 'resnet34',
    encoder_weights: str = 'imagenet',
    in_channels: int = 3,
    num_classes: int = 1,
) -> nn.Module:
    """
    Build U-Net model for grout segmentation.

    Args:
        encoder: Encoder backbone (resnet34, resnet50, efficientnet-b0, mobilenet_v2, etc.)
        encoder_weights: Pretrained weights ('imagenet' or None)
        in_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes (1 for binary segmentation)

    Returns:
        U-Net model
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # We'll use sigmoid in the loss function
    )

    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cuda',
    strict: bool = True,
) -> nn.Module:
    """
    Load model weights from checkpoint.

    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        strict: Whether to strictly enforce state dict matching

    Returns:
        Model with loaded weights
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict)

    # Print additional info if available
    if 'epoch' in checkpoint:
        print(f"  Loaded from epoch {checkpoint['epoch']}")
    if 'best_metric' in checkpoint:
        print(f"  Best metric: {checkpoint['best_metric']:.4f}")

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    checkpoint_path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    additional_info: Optional[dict] = None,
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        best_metric: Best validation metric achieved
        checkpoint_path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        additional_info: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if additional_info is not None:
        checkpoint.update(additional_info)

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    """Print model architecture summary."""
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")

    # Print encoder info
    if hasattr(model, 'encoder'):
        encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"Encoder parameters: {encoder_params:,}")
        decoder_params = total_params - encoder_params
        print(f"Decoder parameters: {decoder_params:,}")

    print("=" * 60 + "\n")


class ModelEMA:
    """
    Exponential Moving Average (EMA) for model weights.
    Can improve model generalization.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: Model to track
            decay: EMA decay rate
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    # Create model
    model = build_model(
        encoder=Config.ENCODER,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )

    print_model_summary(model)

    # Test forward pass
    print("Testing forward pass...")
    device = Config.DEVICE
    model = model.to(device)

    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    print("\nModel test completed successfully!")
