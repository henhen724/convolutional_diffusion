#!/usr/bin/env python3
"""Script to run distribution comparison analysis for all datasets."""

import os
import sys
sys.path.append('../src')

from src.utils.data import get_dataset
from src.utils.patch_statistics import analyze_multiple_datasets
import torch

def main():
    """Run distribution analysis for all available datasets."""
    
    # List of datasets to analyze
    dataset_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CelebA']
    
    # Configuration
    patch_sizes = [3, 6, 10]
    num_samples = 200  # Reduced for faster computation but still good statistics
    results_dir = '../results'
    data_root = '../data'
    
    print("Starting distribution comparison analysis...")
    print(f"Datasets: {dataset_names}")
    print(f"Patch sizes: {patch_sizes}")
    print(f"Samples per dataset: {num_samples}")
    print(f"Results directory: {results_dir}")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Run analysis
    try:
        results = analyze_multiple_datasets(
            dataset_names=dataset_names,
            get_dataset_func=get_dataset,
            data_root=data_root,
            patch_sizes=patch_sizes,
            num_samples=num_samples,
            results_dir=results_dir
        )
        
        print("\n" + "="*60)
        print("DISTRIBUTION ANALYSIS COMPLETE")
        print("="*60)
        
        # Print summary
        for dataset_name, dataset_results in results.items():
            if 'error' in dataset_results:
                print(f"❌ {dataset_name}: Failed - {dataset_results['error']}")
            else:
                print(f"✅ {dataset_name}: Success")
                for patch_size in patch_sizes:
                    if patch_size in dataset_results:
                        patch_results = dataset_results[patch_size]
                        if 'distribution_fits' in patch_results and 'error' not in patch_results['distribution_fits']:
                            dist_fits = patch_results['distribution_fits']
                            if 'weibull' in dist_fits and 'gumbel' in dist_fits and 'gev' in dist_fits:
                                weibull_aic = dist_fits['weibull']['aic']
                                gumbel_aic = dist_fits['gumbel']['aic']
                                gev_aic = dist_fits['gev']['aic']
                                
                                # Find the best fit (lowest AIC)
                                aic_values = {'Weibull': weibull_aic, 'Gumbel': gumbel_aic, 'GEV': gev_aic}
                                better_fit = min(aic_values.keys(), key=lambda k: aic_values[k])
                                best_aic = min(weibull_aic, gumbel_aic, gev_aic)
                                print(f"   📊 {patch_size}x{patch_size}: {better_fit} better fit (AIC: {best_aic:.1f})")
        
        print(f"\n📁 Results saved to: {results_dir}/")
        print("📈 Distribution comparison plots created for each dataset")
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 