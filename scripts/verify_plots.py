#!/usr/bin/env python3
"""Script to verify distribution comparison plots were created successfully."""

import os
from pathlib import Path

def main():
    """Verify the distribution plots were created."""
    
    results_dir = Path('../results')
    datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CelebA']
    
    print("Distribution Comparison Plot Verification")
    print("=" * 50)
    
    all_found = True
    
    for dataset in datasets:
        plot_path = results_dir / dataset / f'{dataset}_distribution_comparison.png'
        
        if plot_path.exists():
            file_size = plot_path.stat().st_size / 1024  # Size in KB
            print(f"✅ {dataset}: {plot_path} ({file_size:.1f} KB)")
        else:
            print(f"❌ {dataset}: Plot not found at {plot_path}")
            all_found = False
    
    print("\n" + "=" * 50)
    if all_found:
        print("🎉 All distribution comparison plots created successfully!")
        print("\nThese plots show:")
        print("- Histogram of patch distances (light blue)")
        print("- Weibull distribution fit (red line)")
        print("- Gumbel distribution fit (green line)")
        print("- AIC values for comparison (lower is better)")
        print("- Multiple patch sizes (3x3, 6x6, 10x10) side by side")
        
        print(f"\n📁 Plots are saved in: {results_dir.absolute()}")
        print("You can open these PNG files to view the distribution comparisons.")
    else:
        print("⚠️  Some plots are missing. Please check the analysis.")
    
    return 0 if all_found else 1

if __name__ == "__main__":
    exit(main()) 