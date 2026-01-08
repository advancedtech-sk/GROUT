# GROUT: Geometric Reasoning Over Unstructured Tessellations

### Zero-Shot Semantic Segmentation of Mosaic Surfaces via Synthetic Priors

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/advancedtech-sk/GROUT)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/advancedtech-sk/GROUT)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

**GROUT** is a deep learning model for detecting and segmenting grout lines (mortar joints) in mosaic images. The model uses a U-Net architecture with an EfficientNet-B3 encoder, trained **exclusively on procedurally generated synthetic data** to achieve zero-shot transfer to real-world mosaics.

**Author:** Radoslav Lovecky, Institute of Advanced Technologies, Slovakia

## 🌟 Overview

Precise segmentation of tessellated surfaces is a challenge for cultural heritage documentation. GROUT overcomes the scarcity of annotated data by training on synthetic geometric priors. It generalizes to real-world Roman, Byzantine, and Modern mosaics without seeing a single real image during training.

## ✨ Features

- **U-Net Architecture:** EfficientNet-B3 encoder (~12M parameters).
- **Zero-shot Transfer:** Trained on 100% synthetic data, works instantly on real images.
- **Multiple Geometry Support:** Handles *Opus Vermiculatum* (flow lines), *Opus Tessellatum* (running bond), and irregular Voronoi.
- **Sliding Window Inference:** Processes gigapixel archival images at full resolution without downsampling artifacts.
- **Robustness:** Explicitly trained to ignore stone grain, dirt, and weathering artifacts.

## 🚀 Quick Start

### Online Demo

Try GROUT online without installing anything:
👉 [**Hugging Face Space**](https://huggingface.co/spaces/advancedtech-sk/GROUT)

### Installation

```bash
git clone [https://github.com/advancedtech-sk/GROUT.git](https://github.com/advancedtech-sk/GROUT.git)
cd GROUT
pip install -r requirements.txt
```

### Download Pre-trained Model

Download the model from [Hugging Face](https://huggingface.co/advancedtech-sk/GROUT):

```bash
# Using huggingface_hub
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('advancedtech-sk/GROUT', 'grout_b3_zeroshot_v1.pth', local_dir='checkpoints')"
```

### Inference

```bash
# Single image
python inference.py --image path/to/mosaic.jpg --model checkpoints/grout_b3_zeroshot_v1.pth

# Directory of images
python inference.py --input_dir path/to/images --model checkpoints/grout_b3_zeroshot_v1.pth
```

## Training

### Generate Synthetic Data

```bash
python generate_synthetic.py
```

This generates 1,500 synthetic mosaic images with masks in `data_local/`:
- **40%** Running Bond (brick pattern with parallel lines)
- **30%** Opus Vermiculatum (concentric circles)
- **30%** Voronoi (random tessellation)

Special modes:
- **15%** Monochrome tiles (model must rely on grout lines only)
- **10%** Zero grout (model must rely on color differences only)

### Train from Scratch

```bash
python train.py --epochs 200 --batch_size 32 --encoder efficientnet-b3
```

### Fine-tune on Custom Data

Prepare your data in `data_finetune/images/` and `data_finetune/masks/`, then:

```bash
# Stage 1: Frozen encoder (safe adaptation)
python finetune.py --stage 1 --checkpoint checkpoints/grout_b3_zeroshot_v1.pth --data_dir data_finetune

# Stage 2: Unfrozen with low LR (gentle refinement)
python finetune.py --stage 2 --checkpoint checkpoints/finetune_stage1.pth --data_dir data_finetune
```

## Model Architecture

```
GROUT v1
├── Encoder: EfficientNet-B3 (pretrained on ImageNet)
├── Decoder: U-Net style with skip connections
├── Input: 512x512 RGB images
├── Output: Binary mask of grout lines
└── Parameters: ~12M
```

## File Structure

```
GROUT/
├── config.py              # Configuration (hyperparameters, paths)
├── dataset.py             # Data loading and augmentation
├── model.py               # U-Net model definition
├── losses.py              # Dice, BCE, and combined losses
├── utils.py               # Metrics, visualization utilities
├── train.py               # Main training script
├── inference.py           # Inference and evaluation
├── generate_synthetic.py  # Synthetic data generator
├── finetune.py            # Fine-tuning script
├── finetune_config.py     # Fine-tuning configuration
├── export_model.py        # ONNX export
└── requirements.txt       # Dependencies
```

## Configuration

Key settings in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 512 | Input image size |
| `ENCODER` | efficientnet-b3 | Encoder backbone |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 1e-4 | Initial learning rate |
| `DICE_WEIGHT` | 0.5 | Weight for Dice loss |
| `BCE_WEIGHT` | 0.3 | Weight for BCE loss |

## Synthetic Data Generation

The synthetic data generator creates realistic mosaic patterns:

### Geometry Modes
- **Running Bond**: Parallel lines with offset vertical joints (brick pattern)
- **Opus Vermiculatum**: Concentric circles with random centers
- **Voronoi**: Random tessellation (400-1500 seeds)

### Features
- Random rotation and scaling
- Adaptive grout color (contrast against tiles)
- Realistic noise and blur
- Variable grout thickness (1-4px image, 2-3px mask)

## Evaluation

```bash
# Evaluate on test set
python inference.py --evaluate --test_dir data_local --model checkpoints/best_model.pth
```

Metrics:
- **Dice Score**: Overlap between prediction and ground truth
- **IoU**: Intersection over Union
- **Precision/Recall**: Detection accuracy

## Export to ONNX

```bash
python export_model.py --checkpoint checkpoints/best_model.pth --output exports/grout.onnx
```

## Citation

If you use GROUT in your research, please cite:

```bibtex
@software{grout2025,
  author = {Lovecky, Radoslav},
  title = {GROUT: Geometric Reasoning Over Unstructured Tessellations},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/advancedtech-sk/GROUT}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Links

- [Hugging Face Space (Demo)](https://huggingface.co/spaces/advancedtech-sk/GROUT)
- [Hugging Face Model](https://huggingface.co/advancedtech-sk/GROUT)
- [GitHub Repository](https://github.com/advancedtech-sk/GROUT)
