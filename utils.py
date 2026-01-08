"""
Utility functions for visualization, metrics, and post-processing.
"""
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
from scipy.ndimage import label as scipy_label
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from pathlib import Path


def calculate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate segmentation metrics.

    Args:
        pred: Predicted probabilities, shape (B, 1, H, W) or (1, H, W) or (H, W)
        target: Ground truth masks, shape (B, 1, H, W) or (1, H, W) or (H, W)
        threshold: Threshold for binarizing predictions

    Returns:
        Dictionary of metrics
    """
    # Handle different input shapes
    if len(pred.shape) == 4:
        pred = pred.squeeze(1)  # (B, H, W)
        target = target.squeeze(1)
    elif len(pred.shape) == 3:
        pred = pred.squeeze(0)  # (H, W)
        target = target.squeeze(0)

    # Ensure tensors are on same device
    if isinstance(pred, torch.Tensor):
        pred = pred.detach()
    if isinstance(target, torch.Tensor):
        target = target.detach()

    # Binarize predictions
    pred_binary = (pred > threshold).float()

    # Flatten for calculations
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    # Calculate components
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()

    # Avoid division by zero
    epsilon = 1e-7

    # Calculate metrics
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
    }


def evaluate_connectivity(
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, int]:
    """
    Evaluate connectivity by counting connected components.

    Args:
        pred: Predicted binary mask (H, W)
        target: Ground truth binary mask (H, W)

    Returns:
        Dictionary with component counts
    """
    # Use 8-connectivity
    structure = np.ones((3, 3), dtype=np.int32)

    # Count components
    _, pred_components = scipy_label(pred, structure=structure)
    _, target_components = scipy_label(target, structure=structure)

    return {
        'pred_components': pred_components,
        'target_components': target_components,
        'component_diff': abs(pred_components - target_components),
    }


def post_process_grout(
    mask: np.ndarray,
    tile_size_est: int = 20,
    fill_gaps: bool = True,
    remove_small: bool = True,
    skeletonize_output: bool = False,
) -> np.ndarray:
    """
    Post-process grout segmentation mask.

    Args:
        mask: Binary mask (H, W), values in [0, 255] or [0, 1]
        tile_size_est: Estimated tile size in pixels (for gap filling)
        fill_gaps: Fill small gaps using morphological closing
        remove_small: Remove small disconnected components
        skeletonize_output: Thin grout lines to skeleton

    Returns:
        Processed binary mask (H, W), values in [0, 255]
    """
    # Ensure binary mask
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    processed = mask.copy()

    # Fill small gaps using morphological closing
    if fill_gaps:
        kernel_size = max(3, tile_size_est // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    # Remove small disconnected components
    if remove_small:
        # Convert to binary
        binary_mask = (processed > 127).astype(np.uint8)

        # Label connected components
        labeled = label(binary_mask)

        # Remove small objects
        min_size = max(50, tile_size_est * 2)
        cleaned = remove_small_objects(labeled, min_size=min_size)

        processed = (cleaned > 0).astype(np.uint8) * 255

    # Skeletonize to thin lines
    if skeletonize_output:
        binary_mask = (processed > 127).astype(bool)
        skeleton = skeletonize(binary_mask)
        processed = (skeleton * 255).astype(np.uint8)

    return processed


def overlay_grout_on_image(
    image: np.ndarray,
    grout_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Overlay grout mask on original image.

    Args:
        image: RGB image (H, W, 3)
        grout_mask: Binary mask (H, W), values in [0, 255] or [0, 1]
        color: RGB color for grout overlay
        alpha: Transparency (0 = transparent, 1 = opaque)

    Returns:
        Image with grout overlay (H, W, 3)
    """
    # Ensure image is RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Ensure mask is binary
    if grout_mask.max() <= 1.0:
        grout_mask = (grout_mask * 255).astype(np.uint8)

    # Create colored overlay
    overlay = image.copy()
    mask_3d = grout_mask > 127

    # Apply color
    overlay[mask_3d] = color

    # Blend with original image
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return result


def visualize_prediction(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Visualize image, ground truth, and prediction side-by-side.

    Args:
        image: Original image (H, W, 3) or (3, H, W)
        mask: Ground truth mask (H, W) or (1, H, W)
        prediction: Predicted mask (H, W) or (1, H, W)
        save_path: Path to save visualization
        show: Whether to display the plot
    """
    # Handle different input formats
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()

    # Handle channel-first format
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if len(prediction.shape) == 3 and prediction.shape[0] == 1:
        prediction = prediction.squeeze(0)

    # Denormalize image if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    # Ground truth
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=12)
    axes[1].axis('off')

    # Prediction
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction', fontsize=12)
    axes[2].axis('off')

    # Overlay
    overlay = overlay_grout_on_image(image, (prediction * 255).astype(np.uint8))
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay', fontsize=12)
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    save_path: str,
    max_samples: int = 4,
):
    """
    Visualize a batch of predictions.

    Args:
        images: Batch of images (B, 3, H, W)
        masks: Batch of ground truth masks (B, 1, H, W)
        predictions: Batch of predictions (B, 1, H, W)
        save_path: Path to save visualization
        max_samples: Maximum number of samples to visualize
    """
    batch_size = min(images.shape[0], max_samples)

    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))

    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size):
        # Get sample
        img = images[i].detach().cpu().numpy()
        mask = masks[i, 0].detach().cpu().numpy()
        pred = torch.sigmoid(predictions[i, 0]).detach().cpu().numpy()

        # Denormalize image
        img = np.transpose(img, (1, 2, 0))
        from config import Config
        img = img * np.array(Config.STD) + np.array(Config.MEAN)
        img = np.clip(img, 0, 1)

        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Image' if i == 0 else '')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth' if i == 0 else '')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction' if i == 0 else '')
        axes[i, 2].axis('off')

        # Overlay
        overlay = overlay_grout_on_image(
            (img * 255).astype(np.uint8),
            (pred * 255).astype(np.uint8)
        )
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay' if i == 0 else '')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Batch visualization saved to {save_path}")


def plot_training_history(
    history: Dict,
    save_path: str,
):
    """
    Plot training history (losses and metrics).

    Args:
        history: Dictionary with training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Plot Dice score
    if 'train_dice' in history and 'val_dice' in history:
        axes[0, 1].plot(history['train_dice'], label='Train Dice', linewidth=2)
        axes[0, 1].plot(history['val_dice'], label='Val Dice', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_title('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Plot IoU
    if 'train_iou' in history and 'val_iou' in history:
        axes[1, 0].plot(history['train_iou'], label='Train IoU', linewidth=2)
        axes[1, 0].plot(history['val_iou'], label='Val IoU', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].set_title('Intersection over Union')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Plot learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Denormalize image from ImageNet statistics.

    Args:
        image: Normalized image (H, W, 3) or (3, H, W)
        mean: Mean values used for normalization
        std: Std values used for normalization

    Returns:
        Denormalized image in [0, 255] range
    """
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    mean = np.array(mean)
    std = np.array(std)

    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Create dummy data
    pred = torch.rand(4, 1, 256, 256)
    target = torch.randint(0, 2, (4, 1, 256, 256)).float()

    # Test metrics
    print("\nTesting metrics calculation...")
    metrics = calculate_metrics(pred, target)
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Test post-processing
    print("\nTesting post-processing...")
    mask = (torch.rand(256, 256) > 0.5).numpy().astype(np.uint8) * 255
    processed = post_process_grout(mask)
    print(f"Original mask shape: {mask.shape}")
    print(f"Processed mask shape: {processed.shape}")

    print("\nUtility functions test completed successfully!")
