# Patch Statistics Analysis Module

The `patch_statistics.py` module provides comprehensive tools for analyzing image patch distance distributions, computing radial power spectra, and fitting statistical distributions to patch data.

## Features

### Core Analysis Functions

- **`analyze_patch_distances()`** - Analyze patch distance distributions for multiple patch sizes
- **`plot_and_save_results()`** - Create comprehensive visualizations and save results
- **`analyze_multiple_datasets()`** - Batch analysis across multiple datasets
- **`fit_distributions()`** - Fit Weibull and Gumbel distributions to distance data

### Enhanced RandomCrop Transform

- **`RandomCrop`** - Robust random cropping with multi-format support
  - Handles 2D grayscale, 3D RGB, and 4D batch tensors
  - Automatic shape normalization
  - Comprehensive error handling with detailed debugging

### Distribution Analysis

- **`weibull_pdf()`** - Weibull probability density function
- **`gumbel_pdf()`** - Gumbel probability density function
- AIC-based model comparison
- Kolmogorov-Smirnov goodness-of-fit tests

## Usage Examples

### Basic Single Dataset Analysis

```python
from src.utils.patch_statistics import analyze_patch_distances, plot_and_save_results
from src.utils.data import get_dataset

# Load dataset
dataset, metadata = get_dataset('mnist', root='./data', train=True)

# Analyze patch distances
results = analyze_patch_distances(
    dataset, 
    patch_sizes=[3, 6, 10], 
    num_samples=100,
    verbose=True
)

# Create visualizations
plot_and_save_results(results, 'mnist', 'results/mnist')
```

### Multi-Dataset Analysis

```python
from src.utils.patch_statistics import analyze_multiple_datasets
from src.utils.data import get_dataset

# Analyze multiple datasets
results = analyze_multiple_datasets(
    dataset_names=['mnist', 'cifar10', 'fashion_mnist'],
    get_dataset_func=get_dataset,
    data_root='./data',
    patch_sizes=[3, 6, 10],
    num_samples=100,
    results_dir='results'
)
```

### RandomCrop Usage

```python
from src.utils.patch_statistics import RandomCrop
import torch

# Create crop transform
crop = RandomCrop(5, 5)

# Works with different tensor shapes
rgb_img = torch.randn(3, 32, 32)      # RGB image
gray_img = torch.randn(28, 28)        # Grayscale image  
batch_img = torch.randn(4, 3, 32, 32) # Batch of images

# Apply cropping
rgb_crop = crop(rgb_img)     # -> (3, 5, 5)
gray_crop = crop(gray_img)   # -> (1, 5, 5) [adds channel dim]
batch_crop = crop(batch_img) # -> (3, 5, 5) [takes first image]
```

## Error Handling Features

The module includes comprehensive error handling with:

- **Line number reporting** using `sys.exc_info()[2].tb_lineno`
- **Error type classification** with `type(e).__name__`
- **Full stack traces** using `traceback.print_exc()`
- **Contextual information** including tensor shapes and variable states
- **Graceful degradation** - continues processing when individual components fail

### Example Error Output

```
Error in RandomCrop.forward at line 45:
Error type: ValueError
Error message: Crop size (10, 10) larger than image size (8, 8)
Input shape: torch.Size([3, 8, 8])
Original input shape: torch.Size([3, 8, 8])
Crop size: (10, 10)
Full traceback:
[Complete stack trace with line numbers]
```

## Output Structure

### Analysis Results

```python
results = {
    '3x3': {
        'num_patches': 100,
        'distance_stats': {
            'mean': 2.45,
            'std': 0.82,
            'min': 0.95,
            'max': 5.12
        },
        'distribution_fits': {
            'weibull': {
                'params': [1.8, 2.3, 0.4],
                'aic': 245.6,
                'ks_statistic': 0.045,
                'ks_pvalue': 0.82
            },
            'gumbel': {
                'params': [2.1, 0.9],
                'aic': 251.2,
                'ks_statistic': 0.067,
                'ks_pvalue': 0.65
            }
        },
        'radial_power_spectrum': [12.4, 8.9, 6.2, 4.1]
    }
}
```

### Saved Files

For each analyzed dataset, the module saves:

- **`{dataset}_comprehensive_analysis.png`** - Multi-panel visualization
  - Radial power spectra
  - Distance distributions  
  - AIC comparison plots
  - Summary statistics table
- **`{dataset}_results.json`** - Complete numerical results

## Visualization Components

The `plot_and_save_results()` function creates:

1. **Radial Power Spectra** - Log-log plot of spatial frequency vs power
2. **Distance Distributions** - Gaussian approximations of patch distances
3. **AIC Comparison** - Bar chart comparing Weibull vs Gumbel fit quality
4. **Summary Statistics** - Table with mean/std distances and patch counts

## Performance Considerations

- **Patch sampling** is limited to `num_samples` for performance
- **Progress tracking** every 50 samples during analysis
- **Memory management** with batch processing for large datasets
- **Error recovery** allows partial results when some components fail

## Dependencies

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
```

## Testing

Comprehensive test suite available in `tests/test_patch_statistics.py`:

```bash
python -m pytest tests/test_patch_statistics.py -v
```

Tests cover:
- Distribution fitting functions
- RandomCrop with various input formats
- Patch distance analysis pipeline
- Visualization and file saving
- Error handling scenarios
- Integration tests with mock datasets

## Demo Script

Run the demo to see the module in action:

```bash
cd examples
python patch_analysis_demo.py
```

The demo includes:
- RandomCrop functionality demonstration
- Single dataset analysis
- Multi-dataset batch processing
- Results visualization and saving 