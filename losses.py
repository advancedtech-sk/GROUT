"""
Loss functions for grout segmentation.
Includes Dice Loss, BCE Loss, and Connectivity Loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label as scipy_label
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice coefficient loss for segmentation.
    Measures overlap between prediction and ground truth.
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (logits), shape (B, 1, H, W)
            targets: Ground truth masks, shape (B, 1, H, W)

        Returns:
            Dice loss value
        """
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        # Return Dice loss
        return 1.0 - dice


class BCELoss(nn.Module):
    """Binary Cross Entropy Loss with logits."""

    def __init__(self, pos_weight: float = 1.0):
        """
        Args:
            pos_weight: Weight for positive class (useful for imbalanced data)
        """
        super(BCELoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (logits), shape (B, 1, H, W)
            targets: Ground truth masks, shape (B, 1, H, W)

        Returns:
            BCE loss value
        """
        if self.pos_weight.device != predictions.device:
            self.pos_weight = self.pos_weight.to(predictions.device)

        return F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            pos_weight=self.pos_weight,
        )


class ConnectivityLoss(nn.Module):
    """
    Connectivity loss to penalize disconnected grout lines.
    Encourages continuous, connected segmentations.
    """

    def __init__(self, weight: float = 1.0, threshold: float = 0.5):
        """
        Args:
            weight: Loss weight
            threshold: Threshold for binarizing predictions
        """
        super(ConnectivityLoss, self).__init__()
        self.weight = weight
        self.threshold = threshold

    def count_connected_components(self, mask: np.ndarray) -> int:
        """
        Count connected components in a binary mask.

        Args:
            mask: Binary mask (H, W)

        Returns:
            Number of connected components
        """
        # Use 8-connectivity (diagonal neighbors count)
        structure = np.ones((3, 3), dtype=np.int32)
        labeled, num_features = scipy_label(mask, structure=structure)
        return num_features

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (logits), shape (B, 1, H, W)
            targets: Ground truth masks, shape (B, 1, H, W)

        Returns:
            Connectivity loss value
        """
        # Apply sigmoid and threshold
        pred_probs = torch.sigmoid(predictions)

        # Calculate connectivity loss for each sample in batch
        total_loss = 0.0
        batch_size = predictions.shape[0]

        for i in range(batch_size):
            # Convert to numpy for connected component analysis
            pred_mask = (pred_probs[i, 0] > self.threshold).cpu().numpy().astype(np.uint8)
            target_mask = targets[i, 0].cpu().numpy().astype(np.uint8)

            # Skip if target mask is empty
            if target_mask.sum() == 0:
                continue

            # Count components
            pred_components = self.count_connected_components(pred_mask)
            target_components = self.count_connected_components(target_mask)

            # Penalize if prediction has more components (more fragmented)
            if pred_components > target_components:
                component_penalty = (pred_components - target_components) / max(target_components, 1)
                total_loss += component_penalty

        # Average over batch
        return torch.tensor(total_loss / batch_size, device=predictions.device)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor in range (0,1) to balance positive/negative examples
            gamma: Exponent of modulating factor (1 - p_t)^gamma
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (logits), shape (B, 1, H, W)
            targets: Ground truth masks, shape (B, 1, H, W)

        Returns:
            Focal loss value
        """
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')

        # Calculate probabilities
        probs = torch.sigmoid(predictions)

        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss

        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky loss - generalization of Dice loss.
    Useful for dealing with class imbalance.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        """
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (logits), shape (B, 1, H, W)
            targets: Ground truth masks, shape (B, 1, H, W)

        Returns:
            Tversky loss value
        """
        # Apply sigmoid
        predictions = torch.sigmoid(predictions)

        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate Tversky index components
        true_pos = (predictions * targets).sum()
        false_pos = (predictions * (1 - targets)).sum()
        false_neg = ((1 - predictions) * targets).sum()

        # Calculate Tversky index
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )

        return 1.0 - tversky


class CombinedLoss(nn.Module):
    """
    Combined loss function for grout segmentation.
    Combines Dice Loss, BCE Loss, and Connectivity Loss.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.3,
        connectivity_weight: float = 0.2,
        use_focal: bool = False,
    ):
        """
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            connectivity_weight: Weight for Connectivity loss
            use_focal: Use Focal loss instead of BCE
        """
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.connectivity_weight = connectivity_weight

        self.dice_loss = DiceLoss()
        self.bce_loss = FocalLoss() if use_focal else BCELoss()
        self.connectivity_loss = ConnectivityLoss()

        # Validate weights
        total_weight = dice_weight + bce_weight + connectivity_weight
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Loss weights sum to {total_weight}, not 1.0")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False,
    ):
        """
        Args:
            predictions: Model predictions (logits), shape (B, 1, H, W)
            targets: Ground truth masks, shape (B, 1, H, W)
            return_components: If True, return individual loss components

        Returns:
            Combined loss value (or dict of components if return_components=True)
        """
        # Calculate individual losses
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)

        # Connectivity loss is expensive, skip sometimes during training
        if self.connectivity_weight > 0:
            connectivity = self.connectivity_loss(predictions, targets)
        else:
            connectivity = torch.tensor(0.0, device=predictions.device)

        # Combined loss
        total_loss = (
            self.dice_weight * dice
            + self.bce_weight * bce
            + self.connectivity_weight * connectivity
        )

        if return_components:
            return {
                'total': total_loss,
                'dice': dice,
                'bce': bce,
                'connectivity': connectivity,
            }

        return total_loss


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")

    # Create dummy data
    batch_size = 4
    height, width = 256, 256

    predictions = torch.randn(batch_size, 1, height, width)
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    # Test Dice Loss
    dice_loss = DiceLoss()
    dice_value = dice_loss(predictions, targets)
    print(f"\nDice Loss: {dice_value.item():.4f}")

    # Test BCE Loss
    bce_loss = BCELoss()
    bce_value = bce_loss(predictions, targets)
    print(f"BCE Loss: {bce_value.item():.4f}")

    # Test Connectivity Loss
    connectivity_loss = ConnectivityLoss()
    connectivity_value = connectivity_loss(predictions, targets)
    print(f"Connectivity Loss: {connectivity_value.item():.4f}")

    # Test Combined Loss
    combined_loss = CombinedLoss()
    loss_dict = combined_loss(predictions, targets, return_components=True)
    print(f"\nCombined Loss Components:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value.item():.4f}")

    print("\nLoss functions test completed successfully!")
