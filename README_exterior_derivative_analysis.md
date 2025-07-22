# Exterior Derivative Analysis of Score Function Models

This repository provides tools for analyzing the exterior derivative of score function models during the reverse diffusion process. The analysis compares UNet, ResNet, and ELS (Equivalent Locality Score) models.

## Overview

The exterior derivative measures the antisymmetric part of the Jacobian of vector-valued functions, providing insights into the "curl-like" behavior of score functions. This analysis helps understand how different model architectures behave during the reverse diffusion process.

## Key Components

### 1. Main Analysis Script (`analyze_exterior_derivative.py`)

A comprehensive script for analyzing exterior derivative magnitudes across different models and multiple noise realizations.

**Features:**

- Supports UNet, ResNet, and ELS models
- Single and multiple realization analysis
- Automatic model loading from checkpoints
- Comprehensive visualization and data saving
- Progress tracking and error handling

### 2. Demo Script (`examples/exterior_derivative_demo.py`)

A simple demonstration script for quick testing and learning.

**Features:**

- Quick setup with MNIST dataset
- Tests UNet and ResNet models
- Simple ELS demonstration
- Educational output with emojis and clear progress

## Quick Start

### 1. Run the Demo

```bash
cd examples
python exterior_derivative_demo.py
```

This will:

- Load MNIST dataset
- Create simple UNet and ResNet models
- Compute exterior derivatives at different timesteps
- Save results and visualizations to `./results/exterior_derivative_demo/`

### 2. Single Realization Analysis

```bash
python analyze_exterior_derivative.py --dataset mnist --single --nsteps 20
```

### 3. Multiple Realization Analysis

```bash
python analyze_exterior_derivative.py --dataset mnist --multiple 10 --nsteps 20
```

### 4. Using Pre-trained Models

```bash
python analyze_exterior_derivative.py \
    --dataset cifar10 \
    --multiple 5 \
    --unet_path ./checkpoints/backbone_CIFAR10_UNet_zeros_conditional.pt \
    --resnet_path ./checkpoints/backbone_CIFAR10_ResNet_zeros_conditional.pt \
    --scales_path ./checkpoints/scales_CIFAR10_ResNet_zeros_conditional.pt
```

## Command Line Options

### `analyze_exterior_derivative.py`

```bash
python analyze_exterior_derivative.py [OPTIONS]
```

**Options:**

- `--dataset`: Dataset to analyze (`mnist`, `cifar10`, `fashionmnist`)
- `--single`: Run single realization analysis
- `--multiple N`: Run analysis with N different noise realizations
- `--nsteps`: Number of diffusion steps (default: 20)
- `--output_dir`: Output directory for results (default: `./results/exterior_derivative`)
- `--unet_path`: Path to UNet checkpoint (auto-detected if not specified)
- `--resnet_path`: Path to ResNet checkpoint (auto-detected if not specified)
- `--scales_path`: Path to scales file for ELS (auto-detected if not specified)

## Available Datasets

- **MNIST**: Grayscale handwritten digits (1×32×32)
- **CIFAR-10**: Color natural images (3×32×32)
- **FashionMNIST**: Grayscale fashion items (1×32×32)

## Pre-trained Models

The analysis automatically detects pre-trained models in the `./checkpoints/` directory:

### Available Checkpoints

- `backbone_MNIST_UNet_zeros.pt` / `backbone_MNIST_ResNet_zeros.pt`
- `backbone_CIFAR10_UNet_zeros_conditional.pt` / `backbone_CIFAR10_ResNet_zeros_conditional.pt`
- `backbone_FashionMNIST_UNet_zeros_conditional.pt` / `backbone_FashionMNIST_ResNet_zeros_conditional.pt`
- Corresponding scales files: `scales_DATASET_MODEL_*.pt`

## Output Structure

### Single Realization Analysis

```
results/exterior_derivative/{dataset}/single_realization/
├── ed_analysis_single.pt          # Numerical data
└── ed_analysis_single.png          # Visualizations
```

### Multiple Realization Analysis

```
results/exterior_derivative/{dataset}/multiple_realizations/
├── ed_analysis_aggregated.pt       # Aggregated numerical data
├── ed_analysis_aggregated.png      # Aggregated visualizations
└── summary_stats.json              # Summary statistics
```

### ELS Script Outputs

```
results/{expname}/
├── seeds/                          # Initial noise seeds
├── els_outputs/                    # ELS generated samples
└── labels/                         # Labels (if conditional)
```

## Visualizations

### Single Realization Plots

1. **Exterior Derivative vs Time**: ED magnitude during reverse process
2. **Log Scale Plot**: Same data on logarithmic scale
3. **Generated Images**: Final outputs from each model
4. **Mean Comparison**: Bar chart of mean ED magnitudes

### Multiple Realization Plots

1. **Mean ED with Error Bars**: Average across realizations with standard deviation
2. **Coefficient of Variation**: Variability measure (std/mean)
3. **Heatmap**: ED magnitudes across all realizations and timesteps
4. **Summary Statistics**: Comparison of mean and std across models

## Understanding the Results

### Exterior Derivative Magnitude

- **Low values**: Indicates more "conservative" or "potential-like" behavior
- **High values**: Indicates more "curl-like" or rotational behavior
- **Trends over time**: How behavior changes during denoising process

### Model Comparisons

- **UNet vs ResNet**: Architectural differences in score function behavior
- **Neural Networks vs ELS**: Learned vs analytical score functions
- **Temporal Evolution**: How different models evolve during reverse process

## Example Results

### Demo Output

```
🚀 Starting Exterior Derivative Demo
==================================================
Using device: cuda

📁 Loading MNIST dataset...
Dataset loaded: 60000 samples
Image size: 32x32, Channels: 1

🧠 Creating models...
✅ Models created successfully

🔍 Computing exterior derivatives...

Analyzing at t = 0.1
  UNet ED magnitude: 0.002341
  ResNet ED magnitude: 0.001876

Analyzing at t = 0.5
  UNet ED magnitude: 0.004523
  ResNet ED magnitude: 0.003912

📈 Summary:
  UNET: mean ED = 0.003456 ± 0.001234
  RESNET: mean ED = 0.002987 ± 0.000987

✅ Demo completed successfully!
```

## Technical Details

### Exterior Derivative Computation

The exterior derivative of a function f: ℝⁿ → ℝⁿ is computed as:

```
(df)ᵢⱼ = ∂ᵢfⱼ - ∂ⱼfᵢ
```

This is the antisymmetric part of the Jacobian matrix, capturing the "curl-like" behavior of the vector field.

### Score Function Interface

The analysis wraps score functions to work with the exterior derivative computation:

```python
def score_function(x_flat):
    # Reshape flattened input back to image shape
    x_img = x_flat.view(batch_size, channels, height, width)
    # Get score prediction
    score = model.backbone(t, x_img)
    return score.flatten()
```

### Model Integration

- **UNet/ResNet**: Uses DDIM wrapper with backbone models
- **ELS**: Uses ScheduledScoreMachine with LocalEquivBordersScoreModule
- **Timesteps**: Evenly spaced from 1.0 (pure noise) to 0.0 (clean image)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce `--nsteps` or use `--multiple` with fewer realizations
   - Use CPU by setting `CUDA_VISIBLE_DEVICES=""`

2. **Model Loading Errors**

   - Ensure checkpoint files exist in `./checkpoints/`
   - Check file permissions and paths
   - Use explicit paths with `--unet_path`, `--resnet_path`, `--scales_path`

3. **ELS Computational Requirements**

   - ELS models require dataset access and can be slow
   - Consider reducing `max_samples` in ELS creation
   - Use smaller datasets for testing

4. **Missing Dependencies**
   ```bash
   pip install torch torchvision matplotlib numpy scipy
   ```

### Performance Tips

- Start with the demo script to verify installation
- Use single realization analysis first
- Begin with MNIST dataset (smaller and faster)
- Consider reducing diffusion steps for initial testing

## Example Workflows

### Research Workflow

1. Run demo to verify setup
2. Single realization analysis on your dataset
3. Multiple realization analysis (5-10 realizations)
4. Compare across different architectures
5. Analyze results and create custom visualizations

### Quick Testing

1. `python examples/exterior_derivative_demo.py`
2. Check `./results/exterior_derivative_demo/`
3. Examine plots and numerical outputs

### Production Analysis

1. Use pre-trained models with `--unet_path`, `--resnet_path`
2. Run multiple realizations (`--multiple 20`)
3. Use appropriate `--nsteps` for your application
4. Save to organized output directories

## Contributing

To extend the analysis:

1. **Add New Models**: Modify `load_models()` in `ExteriorDerivativeAnalyzer`
2. **Custom Visualizations**: Add methods to the analyzer class
3. **New Datasets**: Extend dataset support in `src.utils.data`
4. **Analysis Metrics**: Add new metrics beyond exterior derivative magnitude

## Citation

If you use this analysis in your research, please cite:

```bibtex
@article{your_paper,
  title={An Analytic Theory of Creativity in Convolutional Diffusion Models},
  url={https://arxiv.org/abs/2412.20292},
  year={2024}
}
```
