"""
Export trained model to different formats (ONNX, TorchScript).
"""
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

from config import Config
from model import build_model, load_checkpoint


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config: Config,
    opset_version: int = 12,
    dynamic_axes: bool = True,
):
    """
    Export model to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save ONNX model
        config: Configuration object
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic input/output shapes
    """
    print("=" * 60)
    print("EXPORTING MODEL TO ONNX")
    print("=" * 60)

    # Load model
    print(f"Loading checkpoint from {checkpoint_path}")
    model = build_model(
        encoder=config.ENCODER,
        encoder_weights=None,
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
    )
    model = load_checkpoint(model, checkpoint_path, config.DEVICE)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(
        1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE,
        device=config.DEVICE
    )

    # Define input/output names
    input_names = ['input']
    output_names = ['output']

    # Define dynamic axes if needed
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'},
        }
    else:
        dynamic_axes_dict = None

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    print(f"Exporting to {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
    )

    print("Export complete!")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")

        # Print model info
        print(f"\nModel info:")
        print(f"  Inputs: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")

        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")

    except Exception as e:
        print(f"Error verifying ONNX model: {e}")
        return

    # Test ONNX inference
    print("\nTesting ONNX inference...")
    try:
        # Create ONNX Runtime session
        session = ort.InferenceSession(
            output_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Run inference
        test_input = np.random.randn(1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE).astype(np.float32)
        onnx_output = session.run(None, {'input': test_input})[0]

        print(f"ONNX inference successful!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {onnx_output.shape}")

        # Compare with PyTorch
        with torch.no_grad():
            torch_input = torch.from_numpy(test_input).to(config.DEVICE)
            torch_output = model(torch_input).cpu().numpy()

        # Calculate difference
        max_diff = np.abs(onnx_output - torch_output).max()
        print(f"  Max difference between ONNX and PyTorch: {max_diff:.6f}")

        if max_diff < 1e-4:
            print("  ✓ ONNX and PyTorch outputs match!")
        else:
            print("  ⚠ Warning: Significant difference between outputs")

    except Exception as e:
        print(f"Error testing ONNX inference: {e}")


def export_to_torchscript(
    checkpoint_path: str,
    output_path: str,
    config: Config,
    method: str = 'trace',
):
    """
    Export model to TorchScript format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save TorchScript model
        config: Configuration object
        method: Export method ('trace' or 'script')
    """
    print("=" * 60)
    print("EXPORTING MODEL TO TORCHSCRIPT")
    print("=" * 60)

    # Load model
    print(f"Loading checkpoint from {checkpoint_path}")
    model = build_model(
        encoder=config.ENCODER,
        encoder_weights=None,
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
    )
    model = load_checkpoint(model, checkpoint_path, config.DEVICE)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(
        1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE,
        device=config.DEVICE
    )

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Export
    print(f"Exporting to {output_path} using method: {method}")

    try:
        if method == 'trace':
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(output_path)
        elif method == 'script':
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_path)
        else:
            raise ValueError(f"Unknown export method: {method}")

        print("Export complete!")

        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File size: {file_size:.2f} MB")

        # Test inference
        print("\nTesting TorchScript inference...")
        loaded_model = torch.jit.load(output_path)
        loaded_model.eval()

        with torch.no_grad():
            test_input = torch.randn(
                1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE,
                device=config.DEVICE
            )
            output = loaded_model(test_input)

        print(f"TorchScript inference successful!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")

    except Exception as e:
        print(f"Error exporting to TorchScript: {e}")


def quantize_model(
    checkpoint_path: str,
    output_path: str,
    config: Config,
):
    """
    Quantize model for faster inference (experimental).

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_path: Path to save quantized model
        config: Configuration object
    """
    print("=" * 60)
    print("QUANTIZING MODEL")
    print("=" * 60)

    # Load model
    print(f"Loading checkpoint from {checkpoint_path}")
    model = build_model(
        encoder=config.ENCODER,
        encoder_weights=None,
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
    )
    model = load_checkpoint(model, checkpoint_path, 'cpu')  # Quantization requires CPU
    model.eval()

    # Apply dynamic quantization
    print("Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Conv2d, nn.Linear},
        dtype=torch.qint8
    )

    # Save quantized model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), output_path)

    print(f"Quantized model saved to {output_path}")

    # Compare file sizes
    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\nFile sizes:")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Quantized: {quantized_size:.2f} MB")
    print(f"  Compression: {original_size / quantized_size:.2f}x")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Export trained model to different formats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/output
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='exports/grout_segmenter.onnx',
                       help='Output path')

    # Export format
    parser.add_argument('--format', type=str, default='onnx',
                       choices=['onnx', 'torchscript', 'quantize'],
                       help='Export format')

    # ONNX options
    parser.add_argument('--opset', type=int, default=12,
                       help='ONNX opset version')
    parser.add_argument('--no-dynamic', action='store_true',
                       help='Disable dynamic axes for ONNX')

    # TorchScript options
    parser.add_argument('--script-method', type=str, default='trace',
                       choices=['trace', 'script'],
                       help='TorchScript export method')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # Export based on format
    if args.format == 'onnx':
        export_to_onnx(
            args.checkpoint,
            args.output,
            Config,
            opset_version=args.opset,
            dynamic_axes=not args.no_dynamic,
        )

    elif args.format == 'torchscript':
        export_to_torchscript(
            args.checkpoint,
            args.output,
            Config,
            method=args.script_method,
        )

    elif args.format == 'quantize':
        quantize_model(
            args.checkpoint,
            args.output,
            Config,
        )

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
