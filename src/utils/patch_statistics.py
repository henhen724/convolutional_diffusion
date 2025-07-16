"""
Patch Statistics Analysis Utilities

This module provides functions for analyzing image patch distance distributions,
fitting statistical distributions, computing radial power spectra, and creating
comprehensive visualizations.
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch


def weibull_pdf(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """Weibull probability density function.
    
    Args:
        x: Input values
        shape: Shape parameter
        scale: Scale parameter
        
    Returns:
        Probability density values
    """
    return stats.weibull_min.pdf(x, shape, scale=scale)


def gumbel_pdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    """Gumbel probability density function.
    
    Args:
        x: Input values
        loc: Location parameter
        scale: Scale parameter
        
    Returns:
        Probability density values
    """
    return stats.gumbel_r.pdf(x, loc=loc, scale=scale)


def fit_distributions(distances: np.ndarray) -> Dict:
    """Fit Weibull and Gumbel distributions to distance data.
    
    Args:
        distances: Array of distance values
        
    Returns:
        Dictionary with fit parameters and quality metrics
    """
    try:
        # Remove any NaN or infinite values
        distances = distances[np.isfinite(distances)]
        if len(distances) == 0:
            return {'error': 'No valid distance values'}
        
        # Fit Weibull distribution
        weibull_params = stats.weibull_min.fit(distances)
        weibull_aic = 2 * len(weibull_params) - 2 * stats.weibull_min.logpdf(distances, *weibull_params).sum()
        
        # Fit Gumbel distribution  
        gumbel_params = stats.gumbel_r.fit(distances)
        gumbel_aic = 2 * len(gumbel_params) - 2 * stats.gumbel_r.logpdf(distances, *gumbel_params).sum()
        
        # Kolmogorov-Smirnov tests
        weibull_ks = stats.kstest(distances, lambda x: stats.weibull_min.cdf(x, *weibull_params))
        gumbel_ks = stats.kstest(distances, lambda x: stats.gumbel_r.cdf(x, *gumbel_params))
        
        return {
            'weibull': {
                'params': weibull_params,
                'aic': weibull_aic,
                'ks_statistic': weibull_ks.statistic,
                'ks_pvalue': weibull_ks.pvalue
            },
            'gumbel': {
                'params': gumbel_params,
                'aic': gumbel_aic,
                'ks_statistic': gumbel_ks.statistic,
                'ks_pvalue': gumbel_ks.pvalue
            }
        }
    except Exception as e:
        print(f"Error in fit_distributions at line {sys.exc_info()[2].tb_lineno}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return {'error': str(e)}


class RandomCrop(torch.nn.Module):
    """Enhanced random crop transform with comprehensive error handling.
    
    Args:
        crop_height: Height of the crop
        crop_width: Width of the crop
    """
    
    def __init__(self, crop_height: int, crop_width: int):
        super().__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random crop to input image with detailed error reporting.
        
        Args:
            img: Input image tensor
            
        Returns:
            Cropped image tensor
            
        Raises:
            ValueError: If image dimensions are incompatible with crop size
            TypeError: If input is not a tensor
        """
        try:
            # Handle different input types and shapes
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)
            
            # Handle different tensor dimensions
            original_shape = img.shape
            if len(img.shape) == 2:  # (H, W) grayscale
                img = img.unsqueeze(0)  # Add channel dimension -> (1, H, W)
            elif len(img.shape) == 4:  # (B, C, H, W) batch
                img = img[0]  # Take first image -> (C, H, W)
            elif len(img.shape) != 3:  # Not (C, H, W)
                raise ValueError(f"Unexpected image shape: {original_shape}")
            
            # Ensure we have (C, H, W)
            if len(img.shape) != 3:
                raise ValueError(f"After preprocessing, expected 3D tensor, got shape: {img.shape}")
                
            c, h, w = img.shape
            
            # Check if crop size is valid
            if self.crop_height > h or self.crop_width > w:
                raise ValueError(f"Crop size ({self.crop_height}, {self.crop_width}) larger than image size ({h}, {w})")
            
            # Generate random crop coordinates
            top = torch.randint(0, h - self.crop_height + 1, (1,)).item()
            left = torch.randint(0, w - self.crop_width + 1, (1,)).item()
            
            # Crop the image
            cropped = img[:, top:top + self.crop_height, left:left + self.crop_width]
            return cropped
            
        except Exception as e:
            print(f"Error in RandomCrop.forward at line {sys.exc_info()[2].tb_lineno}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print(f"Input shape: {img.shape if 'img' in locals() else 'Unknown'}")
            print(f"Original input shape: {original_shape if 'original_shape' in locals() else 'Unknown'}")
            print(f"Crop size: ({self.crop_height}, {self.crop_width})")
            print("Full traceback:")
            traceback.print_exc()
            raise


def analyze_patch_distances(dataset, patch_sizes: List[int] = [3, 6, 10], 
                          num_samples: int = 100, verbose: bool = True) -> Dict:
    """Analyze patch distance distributions with comprehensive error handling.
    
    Args:
        dataset: PyTorch dataset
        patch_sizes: List of patch sizes to analyze
        num_samples: Number of samples to analyze
        verbose: Whether to print progress messages
        
    Returns:
        Dictionary with analysis results for each patch size
    """
    results = {}
    
    try:
        if verbose:
            print(f"Analyzing dataset with {len(dataset)} samples...")
        
        # Test dataset structure first
        if verbose:
            print("Testing dataset structure...")
        try:
            sample = dataset[0]
            if isinstance(sample, tuple):
                img, label = sample
                if verbose:
                    print(f"Dataset returns (image, label) tuples. Image shape: {img.shape}")
            else:
                img = sample
                if verbose:
                    print(f"Dataset returns images directly. Image shape: {img.shape}")
                
        except Exception as e:
            print(f"Error testing dataset structure at line {sys.exc_info()[2].tb_lineno}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return {'error': f'Dataset structure test failed: {e}'}
        
        for patch_size in patch_sizes:
            try:
                if verbose:
                    print(f"\nAnalyzing {patch_size}x{patch_size} patches...")
                
                # Create crop transform
                random_crop = RandomCrop(patch_size, patch_size)
                
                # Test crop transform with first sample
                if verbose:
                    print("Testing crop transform...")
                test_img = dataset[0][0] if isinstance(dataset[0], tuple) else dataset[0]
                test_crop = random_crop(test_img)
                if verbose:
                    print(f"Test crop successful. Crop shape: {test_crop.shape}")
                
                # Collect patches
                patches = []
                for i in range(min(num_samples, len(dataset))):
                    try:
                        if verbose and i % 50 == 0:
                            print(f"Processing sample {i}/{num_samples}...")
                            
                        # Get image from dataset
                        sample = dataset[i]
                        if isinstance(sample, tuple):
                            img = sample[0]
                        else:
                            img = sample
                        
                        # Apply crop
                        patch = random_crop(img)
                        patch_flat = patch.flatten()
                        patches.append(patch_flat)
                        
                    except Exception as e:
                        print(f"Error processing sample {i} at line {sys.exc_info()[2].tb_lineno}:")
                        print(f"Error type: {type(e).__name__}")
                        print(f"Error message: {e}")
                        print("Full traceback:")
                        traceback.print_exc()
                        continue
                
                if len(patches) == 0:
                    if verbose:
                        print(f"No valid patches collected for size {patch_size}")
                    continue
                    
                if verbose:
                    print(f"Collected {len(patches)} patches of size {patch_size}x{patch_size}")
                
                # Convert to tensor
                patches_tensor = torch.stack(patches)
                
                # Compute pairwise distances
                if verbose:
                    print("Computing pairwise distances...")
                distances = torch.cdist(patches_tensor, patches_tensor)
                
                # Get upper triangular distances (excluding diagonal)
                mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
                distances_flat = distances[mask].cpu().numpy()
                
                if verbose:
                    print(f"Computed {len(distances_flat)} pairwise distances")
                
                # Fit distributions
                if verbose:
                    print("Fitting distributions...")
                dist_fits = fit_distributions(distances_flat)
                
                # Compute radial power spectrum
                if verbose:
                    print("Computing radial power spectrum...")
                # Handle both grayscale and RGB patches
                if patches_tensor.shape[1] == patch_size * patch_size:
                    # Grayscale patches
                    avg_patch = patches_tensor.mean(dim=0).reshape(patch_size, patch_size)
                else:
                    # RGB patches (shape is [num_patches, patch_size * patch_size * 3])
                    avg_patch = patches_tensor.mean(dim=0).reshape(3, patch_size, patch_size)
                    # Convert to grayscale for power spectrum analysis
                    avg_patch = avg_patch.mean(dim=0)  # Average across channels
                
                fft_result = torch.fft.fft2(avg_patch)
                power_spectrum = torch.abs(fft_result) ** 2
                
                # Compute radial average
                center = patch_size // 2
                y, x = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size), indexing='ij')
                r = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
                
                radial_profile = []
                for radius in range(int(r.max()) + 1):
                    mask = (r >= radius - 0.5) & (r < radius + 0.5)
                    if mask.sum() > 0:
                        radial_profile.append(power_spectrum[mask].mean().item())
                
                results[f'{patch_size}x{patch_size}'] = {
                    'num_patches': len(patches),
                    'distance_stats': {
                        'mean': float(distances_flat.mean()),
                        'std': float(distances_flat.std()),
                        'min': float(distances_flat.min()),
                        'max': float(distances_flat.max())
                    },
                    'distribution_fits': dist_fits,
                    'radial_power_spectrum': radial_profile
                }
                
                if verbose:
                    print(f"Analysis complete for {patch_size}x{patch_size} patches")
                
            except Exception as e:
                print(f"Error analyzing patch size {patch_size} at line {sys.exc_info()[2].tb_lineno}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {e}")
                print("Full traceback:")
                traceback.print_exc()
                continue
                
        return results
        
    except Exception as e:
        print(f"Error in analyze_patch_distances at line {sys.exc_info()[2].tb_lineno}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return {'error': str(e)}


def plot_and_save_results(results: Dict, dataset_name: str, save_dir: Union[str, Path]) -> None:
    """Create comprehensive visualizations and save results with detailed error reporting.
    
    Args:
        results: Analysis results dictionary
        dataset_name: Name of the dataset
        save_dir: Directory to save results
    """
    try:
        print(f"Creating visualizations for {dataset_name}...")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{dataset_name} Dataset Analysis', fontsize=16)
        
        patch_sizes = sorted([key for key in results.keys() if 'x' in key and 'error' not in results[key]])
        
        if len(patch_sizes) == 0:
            print(f"No valid patch size results found for {dataset_name}")
            return
        
        colors = ['blue', 'red', 'green']
        
        try:
            # Plot 1: Radial Power Spectra
            ax = axes[0, 0]
            for i, patch_size in enumerate(patch_sizes):
                try:
                    spectrum = results[patch_size]['radial_power_spectrum']
                    ax.loglog(range(len(spectrum)), spectrum, 
                             color=colors[i % len(colors)], 
                             label=f'{patch_size} patches', marker='o', markersize=3)
                except Exception as e:
                    print(f"Error plotting radial spectrum for {patch_size} at line {sys.exc_info()[2].tb_lineno}:")
                    print(f"Error: {e}")
                    traceback.print_exc()
                    
            ax.set_xlabel('Spatial Frequency')
            ax.set_ylabel('Power')
            ax.set_title('Radial Power Spectra')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error creating radial power spectra plot at line {sys.exc_info()[2].tb_lineno}:")
            print(f"Error: {e}")
            traceback.print_exc()
        
        try:
            # Plot 2: Distance Distribution Histograms  
            ax = axes[0, 1]
            for i, patch_size in enumerate(patch_sizes):
                try:
                    stats_data = results[patch_size]['distance_stats']
                    # Create synthetic histogram data from stats (simplified)
                    mean_val = stats_data['mean']
                    std_val = stats_data['std']
                    x = np.linspace(mean_val - 3*std_val, mean_val + 3*std_val, 100)
                    y = np.exp(-0.5 * ((x - mean_val) / std_val) ** 2)  # Gaussian approximation
                    ax.plot(x, y, color=colors[i % len(colors)], 
                           label=f'{patch_size} (μ={mean_val:.2f})', alpha=0.7)
                except Exception as e:
                    print(f"Error plotting distance distribution for {patch_size} at line {sys.exc_info()[2].tb_lineno}:")
                    print(f"Error: {e}")
                    traceback.print_exc()
                    
            ax.set_xlabel('Distance')
            ax.set_ylabel('Density')
            ax.set_title('Distance Distributions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error creating distance distribution plot at line {sys.exc_info()[2].tb_lineno}:")
            print(f"Error: {e}")
            traceback.print_exc()
        
        try:
            # Plot 3: AIC Comparison
            ax = axes[0, 2]
            weibull_aics = []
            gumbel_aics = []
            patch_labels = []
            
            for patch_size in patch_sizes:
                try:
                    fits = results[patch_size]['distribution_fits']
                    if 'weibull' in fits and 'gumbel' in fits:
                        weibull_aics.append(fits['weibull']['aic'])
                        gumbel_aics.append(fits['gumbel']['aic'])
                        patch_labels.append(patch_size)
                except Exception as e:
                    print(f"Error extracting AIC data for {patch_size} at line {sys.exc_info()[2].tb_lineno}:")
                    print(f"Error: {e}")
                    traceback.print_exc()
            
            if patch_labels:
                x_pos = np.arange(len(patch_labels))
                ax.bar(x_pos - 0.2, weibull_aics, 0.4, label='Weibull', alpha=0.7)
                ax.bar(x_pos + 0.2, gumbel_aics, 0.4, label='Gumbel', alpha=0.7)
                ax.set_xlabel('Patch Size')
                ax.set_ylabel('AIC')
                ax.set_title('Distribution Fit Quality (AIC)')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(patch_labels)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error creating AIC comparison plot at line {sys.exc_info()[2].tb_lineno}:")
            print(f"Error: {e}")
            traceback.print_exc()
        
        try:
            # Plot 4: Summary Statistics
            ax = axes[1, 0]
            stats_table = []
            for patch_size in patch_sizes:
                try:
                    stats_data = results[patch_size]['distance_stats']
                    stats_table.append([
                        patch_size,
                        f"{stats_data['mean']:.3f}",
                        f"{stats_data['std']:.3f}",
                        f"{results[patch_size]['num_patches']}"
                    ])
                except Exception as e:
                    print(f"Error processing stats for {patch_size} at line {sys.exc_info()[2].tb_lineno}:")
                    print(f"Error: {e}")
                    traceback.print_exc()
            
            if stats_table:
                ax.axis('tight')
                ax.axis('off')
                table = ax.table(cellText=stats_table,
                               colLabels=['Patch Size', 'Mean Distance', 'Std Distance', 'Num Patches'],
                               cellLoc='center',
                               loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                ax.set_title('Summary Statistics', pad=20)
            
        except Exception as e:
            print(f"Error creating summary statistics table at line {sys.exc_info()[2].tb_lineno}:")
            print(f"Error: {e}")
            traceback.print_exc()
        
        # Remove empty subplots
        for i in range(1, 3):
            fig.delaxes(axes[1, i])
        
        plt.tight_layout()
        
        try:
            # Save figure
            fig_path = save_path / f'{dataset_name}_comprehensive_analysis.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {fig_path}")
            
        except Exception as e:
            print(f"Error saving figure at line {sys.exc_info()[2].tb_lineno}:")
            print(f"Error: {e}")
            traceback.print_exc()
        
        plt.show()
        
        try:
            # Save results as JSON
            json_path = save_path / f'{dataset_name}_results.json'
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict) and 'error' not in value:
                    json_results[key] = {
                        'num_patches': value['num_patches'],
                        'distance_stats': value['distance_stats'],
                        'radial_power_spectrum': value['radial_power_spectrum']
                    }
                    # Add distribution fits if they exist
                    if 'distribution_fits' in value and 'error' not in value['distribution_fits']:
                        json_results[key]['distribution_fits'] = {}
                        for dist_name, dist_data in value['distribution_fits'].items():
                            if isinstance(dist_data, dict):
                                json_results[key]['distribution_fits'][dist_name] = {
                                    'params': [float(p) for p in dist_data['params']],
                                    'aic': float(dist_data['aic']),
                                    'ks_statistic': float(dist_data['ks_statistic']),
                                    'ks_pvalue': float(dist_data['ks_pvalue'])
                                }
            
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"Saved results to {json_path}")
            
        except Exception as e:
            print(f"Error saving JSON results at line {sys.exc_info()[2].tb_lineno}:")
            print(f"Error: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error in plot_and_save_results at line {sys.exc_info()[2].tb_lineno}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Full traceback:")
        traceback.print_exc()


def analyze_multiple_datasets(dataset_names: List[str], get_dataset_func, 
                            data_root: str = './data', patch_sizes: List[int] = [3, 6, 10],
                            num_samples: int = 100, results_dir: str = 'results') -> Dict:
    """Analyze multiple datasets with comprehensive error handling.
    
    Args:
        dataset_names: List of dataset names to analyze
        get_dataset_func: Function to load datasets
        data_root: Root directory for data
        patch_sizes: List of patch sizes to analyze
        num_samples: Number of samples per dataset
        results_dir: Directory to save results
        
    Returns:
        Dictionary with results for all datasets
    """
    all_results = {}
    
    for dataset_name in dataset_names:
        try:
            print(f"\n{'='*50}")
            print(f"ANALYZING DATASET: {dataset_name.upper()}")
            print(f"{'='*50}")
            
            # Load dataset with detailed error reporting
            print(f"Loading {dataset_name} dataset...")
            dataset, metadata = get_dataset_func(dataset_name, root=data_root, train=True)
            print(f"Dataset loaded successfully. Length: {len(dataset)}")
            
            # Test dataset format
            sample = dataset[0]
            if isinstance(sample, tuple):
                img, label = sample
                print(f"Dataset returns (image, label) tuples. Image shape: {img.shape}")
            else:
                img = sample
                print(f"Dataset returns images directly. Image shape: {img.shape}")
                
            # Run analysis
            print(f"Starting patch distance analysis...")
            results = analyze_patch_distances(dataset, patch_sizes=patch_sizes, num_samples=num_samples)
            
            if 'error' in results:
                print(f"Analysis failed: {results['error']}")
                all_results[dataset_name] = results
                continue
                
            all_results[dataset_name] = results
            
            # Create visualizations
            print(f"Creating visualizations...")
            save_dir = f'{results_dir}/{dataset_name}'
            plot_and_save_results(results, dataset_name, save_dir)
            print(f"Completed processing {dataset_name}")
            
        except Exception as e:
            print(f"Error processing {dataset_name} at line {sys.exc_info()[2].tb_lineno}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("Full traceback:")
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}
            print("-" * 50)

    print("\nAnalysis Summary:")
    for dataset_name, results in all_results.items():
        if 'error' in results:
            print(f"{dataset_name}: FAILED - {results['error']}")
        else:
            patch_count = len([k for k in results.keys() if 'x' in k])
            print(f"{dataset_name}: SUCCESS - {patch_count} patch sizes analyzed")
    
    return all_results 