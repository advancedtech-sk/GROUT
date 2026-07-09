"""
Inference and testing module for grout segmentation.
Supports single image, batch inference, and evaluation.
"""
import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
try:  # optional: only needed for the ONNX path
    import onnxruntime as ort
except ImportError:
    ort = None

from config import Config
from model import build_model, load_checkpoint
from utils import (
    calculate_metrics,
    post_process_grout,
    overlay_grout_on_image,
    visualize_prediction,
    denormalize_image,
)
from dataset import load_data_paths, split_dataset


class GroutSegmenter:
    """Grout segmentation inference engine."""

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        use_onnx: bool = False,
        temperature: Optional[float] = None,
    ):
        """
        Initialize segmentation model.

        Args:
            model_path: Path to model checkpoint (.pth) or ONNX model (.onnx)
            device: Device to run inference on
            use_onnx: Whether to use ONNX runtime
            temperature: Calibration temperature (sigmoid(logits/T)). None ->
                resolve from calibration.json (model dir, then cwd), else 1.0.
        """
        from calibrate import resolve_temperature
        self.temperature = resolve_temperature(temperature, model_path)
        if self.temperature != 1.0:
            print(f"Calibration: temperature T = {self.temperature:.4f}")
        # Verify device is actually available and PyTorch is CUDA-enabled
        if device == 'cuda':
            try:
                # Try to create a CUDA tensor to verify CUDA works
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
            except (RuntimeError, AssertionError):
                print("Warning: CUDA requested but PyTorch is not CUDA-enabled. Using CPU instead.")
                print("To enable CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                device = 'cpu'

        self.device = device
        self.use_onnx = use_onnx or model_path.endswith('.onnx')

        if self.use_onnx:
            if ort is None:
                raise ImportError("onnxruntime is required for ONNX models: "
                                  "pip install onnxruntime")
            print(f"Loading ONNX model from {model_path}")
            self.session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.model = None
        else:
            print(f"Loading PyTorch model from {model_path}")
            self.model = build_model(
                encoder=Config.ENCODER,
                encoder_weights=None,
                in_channels=Config.IN_CHANNELS,
                num_classes=Config.NUM_CLASSES,
            )
            self.model = load_checkpoint(self.model, model_path, device)
            self.model = self.model.to(device)  # Ensure model is on correct device
            self.model.eval()
            self.session = None

        print("Model loaded successfully!")

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        apply_post_processing: bool = True,
        tile_size_est: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict grout mask for an image.

        Args:
            image: Input RGB image (H, W, 3)
            apply_post_processing: Whether to apply post-processing
            tile_size_est: Estimated tile size for post-processing

        Returns:
            Tuple of (raw_prediction, processed_prediction)
        """
        # Preprocess image
        input_tensor = self.preprocess(image)

        # Run inference
        if self.use_onnx:
            output = self.session.run(
                None,
                {self.session.get_inputs()[0].name: input_tensor.cpu().numpy()}
            )[0]
            output = torch.from_numpy(output)
        else:
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
            output = output.cpu()

        # Apply (temperature-scaled) sigmoid to get probabilities.
        # T = 1.0 is exactly torch.sigmoid(output); any T > 0 preserves
        # per-pixel ranking. Kept float32 throughout.
        prob = torch.sigmoid(output / self.temperature).squeeze().numpy()
        prob = prob.astype(np.float32)

        # Threshold to binary mask
        # INVARIANT STAGE: (prob > threshold) here is the native-grid raw
        # mask that the exported _prob map reproduces bit-exactly.
        binary_mask = (prob > Config.PREDICTION_THRESHOLD).astype(np.uint8) * 255

        # Post-process
        if apply_post_processing:
            processed_mask = post_process_grout(
                binary_mask,
                tile_size_est=tile_size_est,
                fill_gaps=True,
                remove_small=True,
                skeletonize_output=False,
            )
        else:
            processed_mask = binary_mask

        return prob, processed_mask

    def predict_large_image(
        self,
        image: np.ndarray,
        patch_size: int = 512,
        overlap: int = 64,
        apply_post_processing: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict grout mask for large image using sliding window.

        Args:
            image: Input RGB image (H, W, 3)
            patch_size: Size of patches for inference
            overlap: Overlap between patches
            apply_post_processing: Whether to apply post-processing

        Returns:
            Tuple of (raw_prediction, processed_prediction)
        """
        h, w = image.shape[:2]

        # If image is small enough, use regular prediction
        if h <= patch_size and w <= patch_size:
            return self.predict(image, apply_post_processing)

        # Create output mask
        prob_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        stride = patch_size - overlap

        # Sliding window
        print(f"Processing large image ({h}x{w}) with sliding window...")
        patches_y = list(range(0, h - patch_size + 1, stride)) + [h - patch_size] if h > patch_size else [0]
        patches_x = list(range(0, w - patch_size + 1, stride)) + [w - patch_size] if w > patch_size else [0]

        total_patches = len(patches_y) * len(patches_x)

        with tqdm(total=total_patches, desc="Processing patches") as pbar:
            for y in patches_y:
                for x in patches_x:
                    # Extract patch
                    patch = image[y:y+patch_size, x:x+patch_size]

                    # Predict
                    prob, _ = self.predict(patch, apply_post_processing=False)

                    # Accumulate
                    prob_map[y:y+patch_size, x:x+patch_size] += prob
                    count_map[y:y+patch_size, x:x+patch_size] += 1

                    pbar.update(1)

        # Average overlapping predictions
        prob_map = prob_map / np.maximum(count_map, 1)

        # Threshold to binary mask
        binary_mask = (prob_map > Config.PREDICTION_THRESHOLD).astype(np.uint8) * 255

        # Post-process
        if apply_post_processing:
            processed_mask = post_process_grout(
                binary_mask,
                tile_size_est=20,
                fill_gaps=True,
                remove_small=True,
                skeletonize_output=False,
            )
        else:
            processed_mask = binary_mask

        return prob_map, processed_mask

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Resize if needed
        if image.shape[0] != Config.IMG_SIZE or image.shape[1] != Config.IMG_SIZE:
            image = cv2.resize(image, (Config.IMG_SIZE, Config.IMG_SIZE))

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(Config.MEAN)) / np.array(Config.STD)

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

        return image


def _tta_variants(n: int):
    """First n of 8 dihedral (flip/rot90) transforms as (forward, inverse)
    pairs operating on HxW[xC] numpy arrays."""
    ident = (lambda x: x, lambda x: x)
    variants = [
        ident,
        (lambda x: x[:, ::-1], lambda x: x[:, ::-1]),                    # hflip
        (lambda x: x[::-1, :], lambda x: x[::-1, :]),                    # vflip
        (lambda x: np.rot90(x, 1), lambda x: np.rot90(x, -1)),           # rot90
        (lambda x: np.rot90(x, 2), lambda x: np.rot90(x, -2)),           # rot180
        (lambda x: np.rot90(x, 3), lambda x: np.rot90(x, -3)),           # rot270
        (lambda x: np.rot90(x[:, ::-1], 1),
         lambda x: np.rot90(x, -1)[:, ::-1]),                            # hflip+rot90
        (lambda x: np.rot90(x[::-1, :], 1),
         lambda x: np.rot90(x, -1)[::-1, :]),                            # vflip+rot90
    ]
    return variants[:max(2, min(n, 8))]


def inference_single_image(
    image_path: str,
    model_path: str,
    output_dir: str,
    device: str = 'cuda',
    visualize: bool = True,
    save_prob: bool = True,
    temperature: Optional[float] = None,
    tta_var: int = 0,
    segmenter: Optional['GroutSegmenter'] = None,
):
    """
    Run inference on a single image.

    Args:
        image_path: Path to input image
        model_path: Path to model checkpoint
        output_dir: Directory to save results
        device: Device to run on
        visualize: Whether to create visualizations
        save_prob: Export the pre-threshold confidence map
            (_prob.npy float32 + _prob8.png) at the native inference grid
        temperature: Calibration temperature (None -> calibration.json or 1.0)
        tta_var: If > 0, run N flip/rot90 passes and export a per-pixel
            variance map (_var.npy + _var8.png). The shipped mask still comes
            from the plain pass (unchanged outputs).
        segmenter: Optional pre-loaded GroutSegmenter (avoids reloading)
    """
    from prob_output import save_prob_outputs, save_var_outputs

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Loaded image: {image.shape}")

    # Initialize segmenter
    if segmenter is None:
        segmenter = GroutSegmenter(model_path, device, temperature=temperature)

    # Predict
    print("Running inference...")
    original_size = image.shape[:2]
    image_name = Path(image_path).stem

    if max(original_size) > Config.IMG_SIZE:
        # Use sliding window for large images
        prob, pred_mask = segmenter.predict_large_image(image)
        # native grid == full resolution here
        prob_native = prob
        native_input = image
        use_sliding = True
    else:
        # Resize for inference
        image_resized = cv2.resize(image, (Config.IMG_SIZE, Config.IMG_SIZE))
        prob, pred_mask = segmenter.predict(image_resized)
        prob_native = prob                     # native grid == IMG_SIZE
        native_input = image_resized
        use_sliding = False

        # Resize back to original size (existing shipped behavior, unchanged:
        # bilinear on both; the mask therefore leaves the strictly-binary
        # domain at this stage -- see README pipeline-order note)
        prob = cv2.resize(prob, (original_size[1], original_size[0]))
        pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]))

    # Soft output: canonical float32 at the NATIVE grid, where
    # threshold(prob) == raw pre-morphology mask bit-exactly.
    if save_prob:
        paths = save_prob_outputs(prob_native, output_dir, image_name)
        print(f"Confidence map saved to: {paths['npy'].name} / "
              f"{paths['png8'].name}")

    # Optional TTA uncertainty channel (does not alter shipped outputs)
    if tta_var and tta_var > 0:
        variants = _tta_variants(tta_var)
        probs = []
        for fwd, inv in variants:
            aug = fwd(native_input)
            if use_sliding:
                p, _ = segmenter.predict_large_image(
                    np.ascontiguousarray(aug), apply_post_processing=False)
            else:
                p, _ = segmenter.predict(
                    np.ascontiguousarray(aug), apply_post_processing=False)
            probs.append(inv(p))
        var = np.var(np.stack(probs, axis=0), axis=0).astype(np.float32)
        vpaths = save_var_outputs(var, output_dir, image_name)
        print(f"TTA variance ({len(variants)} passes) saved to: "
              f"{vpaths['npy'].name}")

    print("Inference complete!")

    # Generate timestamped filenames
    mask_filename = Config.get_result_filename(image_name, 'mask')
    overlay_filename = Config.get_result_filename(image_name, 'overlay')
    vis_filename = Config.get_result_filename(image_name, 'visualization')

    # Save mask
    mask_path = os.path.join(output_dir, mask_filename)
    cv2.imwrite(mask_path, pred_mask)
    print(f"Mask saved to: {mask_filename}")

    # Save overlay
    if image.shape[:2] != pred_mask.shape:
        image_for_overlay = cv2.resize(image, (pred_mask.shape[1], pred_mask.shape[0]))
    else:
        image_for_overlay = image

    overlay = overlay_grout_on_image(image_for_overlay, pred_mask)
    overlay_path = os.path.join(output_dir, overlay_filename)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Overlay saved to: {overlay_filename}")

    # Create visualization
    if visualize:
        # Create a dummy mask for visualization (empty)
        dummy_mask = np.zeros_like(pred_mask, dtype=np.float32)
        vis_path = os.path.join(output_dir, vis_filename)

        # Resize image to match prediction size
        if image.shape[:2] != pred_mask.shape:
            image_vis = cv2.resize(image, (pred_mask.shape[1], pred_mask.shape[0]))
        else:
            image_vis = image

        visualize_prediction(
            image_vis,
            dummy_mask,
            pred_mask / 255.0,
            save_path=vis_path,
        )


def inference_directory(
    input_dir: str,
    model_path: str,
    output_dir: str,
    device: str = 'cuda',
    save_prob: bool = True,
    temperature: Optional[float] = None,
    tta_var: int = 0,
):
    """
    Run inference on all images in a directory.

    Args:
        input_dir: Directory containing input images
        model_path: Path to model checkpoint
        output_dir: Directory to save results
        device: Device to run on
        save_prob / temperature / tta_var: see inference_single_image
    """
    # Find all images. Skip GROUT's own derived outputs so re-running over a
    # results folder (or input_dir == output_dir) does not feed exported
    # _prob8/_var8/mask/overlay PNGs back into the model.
    _derived_markers = ('_prob8_', '_var8_', '_mask_', '_overlay_',
                        '_visualization_')
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_exts:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    image_files = [p for p in image_files
                   if not any(m in p.stem for m in _derived_markers)]

    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize segmenter once and reuse
    segmenter = GroutSegmenter(model_path, device, temperature=temperature)

    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            inference_single_image(
                str(img_path),
                model_path,
                output_dir,
                device,
                visualize=False,
                save_prob=save_prob,
                tta_var=tta_var,
                segmenter=segmenter,
            )
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def evaluate_test_set(
    test_imgs: list,
    test_masks: list,
    model_path: str,
    device: str = 'cuda',
) -> dict:
    """
    Evaluate model on test set.

    Args:
        test_imgs: List of test image paths
        test_masks: List of test mask paths
        model_path: Path to model checkpoint
        device: Device to run on

    Returns:
        Dictionary of average metrics
    """
    print(f"\nEvaluating on {len(test_imgs)} test images...")

    # Initialize segmenter
    segmenter = GroutSegmenter(model_path, device)

    all_metrics = []

    for img_path, mask_path in tqdm(zip(test_imgs, test_masks), total=len(test_imgs)):
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_mask > 127).astype(np.float32)

        # Resize to model input size
        image_resized = cv2.resize(image, (Config.IMG_SIZE, Config.IMG_SIZE))
        gt_mask_resized = cv2.resize(gt_mask, (Config.IMG_SIZE, Config.IMG_SIZE))

        # Predict
        prob, pred_mask = segmenter.predict(image_resized, apply_post_processing=True)

        # Calculate metrics
        pred_tensor = torch.from_numpy(pred_mask / 255.0).float()
        gt_tensor = torch.from_numpy(gt_mask_resized).float()

        metrics = calculate_metrics(pred_tensor, gt_tensor)
        all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }

    # Print results
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    for key, value in avg_metrics.items():
        print(f"{key.capitalize():12s}: {value:.4f}")
    print("=" * 60)

    return avg_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Inference for mosaic grout segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/output
    parser.add_argument('--image', type=str, help='Path to single input image')
    parser.add_argument('--input_dir', type=str, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')

    # Model
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')

    # Evaluation
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate on test set')
    parser.add_argument('--test_dir', type=str, default='data',
                       help='Test data directory (for evaluation)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to run on')

    # Soft output / calibration / uncertainty
    parser.add_argument('--save-prob', dest='save_prob', action='store_true',
                       default=True,
                       help='Export pre-threshold confidence map '
                            '(_prob.npy float32 + _prob8.png). Default: on')
    parser.add_argument('--no-prob', dest='save_prob', action='store_false',
                       help='Disable confidence-map export')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Calibration temperature T (sigmoid(logits/T)). '
                            'Default: calibration.json if present, else 1.0')
    parser.add_argument('--tta-var', type=int, default=0, metavar='N',
                       help='N flip/rot90 TTA passes -> per-pixel variance '
                            'map (_var.npy + _var8.png). 0 = off')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    if args.evaluate:
        # Evaluate on test set
        Config.DATA_DIR = args.test_dir
        Config.IMAGES_DIR = os.path.join(args.test_dir, 'images')
        Config.MASKS_DIR = os.path.join(args.test_dir, 'masks')

        # Load and split data
        image_paths, mask_paths = load_data_paths(Config.IMAGES_DIR, Config.MASKS_DIR)
        _, _, _, _, test_imgs, test_masks = split_dataset(
            image_paths, mask_paths, force_train_files=Config.FORCE_TRAIN_FILES
        )

        # Evaluate
        evaluate_test_set(test_imgs, test_masks, args.model, args.device)

    elif args.image:
        # Single image inference
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return

        inference_single_image(
            args.image,
            args.model,
            args.output_dir,
            args.device,
            save_prob=args.save_prob,
            temperature=args.temperature,
            tta_var=args.tta_var,
        )

    elif args.input_dir:
        # Directory inference
        if not os.path.exists(args.input_dir):
            print(f"Error: Directory not found at {args.input_dir}")
            return

        inference_directory(
            args.input_dir,
            args.model,
            args.output_dir,
            args.device,
            save_prob=args.save_prob,
            temperature=args.temperature,
            tta_var=args.tta_var,
        )

    else:
        print("Error: Please specify --image, --input_dir, or --evaluate")
        return


if __name__ == "__main__":
    main()
