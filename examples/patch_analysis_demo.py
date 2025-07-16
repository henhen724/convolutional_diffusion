#!/usr/bin/env python3
"""
Patch Statistics Analysis Demo

This script demonstrates how to use the patch_statistics module to analyze
image datasets and compute patch distance distributions, radial power spectra,
and distribution fits.
"""

import os
import sys

# Add src to path so we can import our utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data import get_dataset
from utils.patch_statistics import (
    RandomCrop,
    analyze_multiple_datasets,
    analyze_patch_distances,
    plot_and_save_results,
)


def demo_single_dataset():
    """Demonstrate analysis on a single dataset."""
    print("=== Single Dataset Analysis Demo ===")
    
    try:
        # Load a dataset
        print("Loading MNIST dataset...")
        dataset, metadata = get_dataset('mnist', root='../data', train=True)
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Metadata: {metadata}")
        
        # Analyze patch distances
        print("\nAnalyzing patch distances...")
        results = analyze_patch_distances(
            dataset, 
            patch_sizes=[3, 6], 
            num_samples=50,
            verbose=True
        )
        
        if 'error' in results:
            print(f"Analysis failed: {results['error']}")
            return
        
        print("Analysis completed successfully!")
        print(f"Analyzed patch sizes: {list(results.keys())}")
        
        # Show some results
        for patch_size, data in results.items():
            print(f"\n{patch_size} patches:")
            print(f"  Number of patches: {data['num_patches']}")
            print(f"  Mean distance: {data['distance_stats']['mean']:.3f}")
            print(f"  Std distance: {data['distance_stats']['std']:.3f}")
            
            # Check if distribution fits are available
            if 'distribution_fits' in data and 'error' not in data['distribution_fits']:
                fits = data['distribution_fits']
                if 'weibull' in fits and 'gumbel' in fits:
                    print(f"  Weibull AIC: {fits['weibull']['aic']:.2f}")
                    print(f"  Gumbel AIC: {fits['gumbel']['aic']:.2f}")
                    best_fit = 'Weibull' if fits['weibull']['aic'] < fits['gumbel']['aic'] else 'Gumbel'
                    print(f"  Best fit: {best_fit}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        plot_and_save_results(results, 'mnist_demo', 'results/demo')
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Error in single dataset demo: {e}")
        import traceback
        traceback.print_exc()


def demo_multiple_datasets():
    """Demonstrate analysis on multiple datasets."""
    print("\n=== Multiple Dataset Analysis Demo ===")
    
    try:
        # Analyze multiple datasets
        results = analyze_multiple_datasets(
            dataset_names=['mnist', 'cifar10'],
            get_dataset_func=get_dataset,
            data_root='../data',
            patch_sizes=[3, 6],
            num_samples=30,
            results_dir='results/multi_demo'
        )
        
        print("\nMulti-dataset analysis summary:")
        for dataset_name, dataset_results in results.items():
            if 'error' in dataset_results:
                print(f"{dataset_name}: FAILED - {dataset_results['error']}")
            else:
                patch_count = len([k for k in dataset_results.keys() if 'x' in k])
                print(f"{dataset_name}: SUCCESS - {patch_count} patch sizes analyzed")
                
                # Show mean distances for each patch size
                for patch_size, data in dataset_results.items():
                    if 'x' in patch_size:
                        mean_dist = data['distance_stats']['mean']
                        print(f"  {patch_size}: mean distance = {mean_dist:.3f}")
        
    except Exception as e:
        print(f"Error in multiple dataset demo: {e}")
        import traceback
        traceback.print_exc()


def demo_random_crop():
    """Demonstrate the RandomCrop functionality."""
    print("\n=== RandomCrop Demo ===")
    
    try:
        import numpy as np
        import torch

        # Test with different image types
        print("Testing RandomCrop with different image formats...")
        
        crop = RandomCrop(5, 5)
        
        # RGB image (3, 32, 32)
        rgb_img = torch.randn(3, 32, 32)
        rgb_crop = crop(rgb_img)
        print(f"RGB image: {rgb_img.shape} -> {rgb_crop.shape}")
        
        # Grayscale image (28, 28)
        gray_img = torch.randn(28, 28)
        gray_crop = crop(gray_img)
        print(f"Grayscale image: {gray_img.shape} -> {gray_crop.shape}")
        
        # Batch of images (4, 3, 32, 32)
        batch_img = torch.randn(4, 3, 32, 32)
        batch_crop = crop(batch_img)
        print(f"Batch image: {batch_img.shape} -> {batch_crop.shape}")
        
        # NumPy array
        numpy_img = np.random.randn(3, 32, 32)
        numpy_crop = crop(numpy_img)
        print(f"NumPy image: {numpy_img.shape} -> {numpy_crop.shape}")
        
        print("RandomCrop demo completed successfully!")
        
    except Exception as e:
        print(f"Error in RandomCrop demo: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all demonstrations."""
    print("Patch Statistics Analysis Module Demo")
    print("=" * 50)
    
    # Create results directories
    os.makedirs('results/demo', exist_ok=True)
    os.makedirs('results/multi_demo', exist_ok=True)
    
    # Run demonstrations
    demo_random_crop()
    demo_single_dataset()
    demo_multiple_datasets()
    
    print("\n" + "=" * 50)
    print("All demos completed! Check the results/ directory for outputs.")


if __name__ == '__main__':
    main() 