"""
Tests for patch statistics analysis utilities.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.utils.patch_statistics import (
    RandomCrop,
    analyze_multiple_datasets,
    analyze_patch_distances,
    fit_distributions,
    gev_pdf,
    gumbel_pdf,
    plot_and_save_results,
    weibull_pdf,
)


class MockDataset(Dataset):
    """Mock dataset for testing."""
    
    def __init__(self, num_samples=100, image_shape=(3, 32, 32), return_tuples=True):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.return_tuples = return_tuples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create random image
        img = torch.randn(*self.image_shape)
        
        if self.return_tuples:
            return img, idx  # Return (image, label) tuple
        else:
            return img  # Return image only


class MockGrayscaleDataset(Dataset):
    """Mock grayscale dataset for testing MNIST-like data."""
    
    def __init__(self, num_samples=100, image_shape=(28, 28)):
        self.num_samples = num_samples
        self.image_shape = image_shape
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create random grayscale image (H, W)
        img = torch.randn(*self.image_shape)
        return img, idx


class TestDistributionFunctions:
    """Test distribution PDF functions."""
    
    def test_weibull_pdf(self):
        """Test Weibull PDF function."""
        x = np.array([1.0, 2.0, 3.0])
        shape, scale = 2.0, 1.0
        
        result = weibull_pdf(x, shape, scale)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        assert all(result >= 0)  # PDF values should be non-negative
    
    def test_gumbel_pdf(self):
        """Test Gumbel PDF function."""
        x = np.array([1.0, 2.0, 3.0])
        loc, scale = 0.0, 1.0
        
        result = gumbel_pdf(x, loc, scale)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        assert all(result >= 0)  # PDF values should be non-negative
    
    def test_gev_pdf(self):
        """Test GEV PDF function."""
        x = np.array([1.0, 2.0, 3.0])
        shape, loc, scale = 0.1, 1.0, 0.5
        
        result = gev_pdf(x, shape, loc, scale)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        assert all(result >= 0)  # PDF values should be non-negative


class TestFitDistributions:
    """Test distribution fitting function."""
    
    def test_fit_distributions_valid_data(self):
        """Test fitting distributions to valid distance data."""
        # Generate some test distance data
        np.random.seed(42)
        distances = np.random.exponential(2.0, 1000)
        
        result = fit_distributions(distances)
        
        assert 'error' not in result
        assert 'weibull' in result
        assert 'gumbel' in result
        assert 'gev' in result
        
        # Check Weibull results
        weibull = result['weibull']
        assert 'params' in weibull
        assert 'aic' in weibull
        assert 'ks_statistic' in weibull
        assert 'ks_pvalue' in weibull
        assert len(weibull['params']) == 3  # Weibull has 3 parameters
        
        # Check Gumbel results
        gumbel = result['gumbel']
        assert 'params' in gumbel
        assert 'aic' in gumbel
        assert 'ks_statistic' in gumbel
        assert 'ks_pvalue' in gumbel
        assert len(gumbel['params']) == 2  # Gumbel has 2 parameters
        
        # Check GEV results
        gev = result['gev']
        assert 'params' in gev
        assert 'aic' in gev
        assert 'ks_statistic' in gev
        assert 'ks_pvalue' in gev
        assert len(gev['params']) == 3  # GEV has 3 parameters
    
    def test_fit_distributions_empty_data(self):
        """Test fitting distributions to empty data."""
        distances = np.array([])
        
        result = fit_distributions(distances)
        
        assert 'error' in result
        assert result['error'] == 'No valid distance values'
    
    def test_fit_distributions_invalid_data(self):
        """Test fitting distributions to data with NaN/inf values."""
        distances = np.array([1.0, 2.0, np.nan, np.inf, 3.0])
        
        result = fit_distributions(distances)
        
        # Should filter out invalid values and process remaining data
        assert 'error' not in result or 'No valid distance values' in result['error']


class TestRandomCrop:
    """Test RandomCrop transform."""
    
    def test_random_crop_3d_image(self):
        """Test random crop on 3D image (C, H, W)."""
        crop = RandomCrop(5, 5)
        img = torch.randn(3, 32, 32)  # RGB image
        
        result = crop(img)
        
        assert result.shape == (3, 5, 5)
    
    def test_random_crop_2d_image(self):
        """Test random crop on 2D grayscale image (H, W)."""
        crop = RandomCrop(5, 5)
        img = torch.randn(28, 28)  # Grayscale image
        
        result = crop(img)
        
        assert result.shape == (1, 5, 5)  # Should add channel dimension
    
    def test_random_crop_4d_batch(self):
        """Test random crop on 4D batch (B, C, H, W)."""
        crop = RandomCrop(5, 5)
        img = torch.randn(4, 3, 32, 32)  # Batch of RGB images
        
        result = crop(img)
        
        assert result.shape == (3, 5, 5)  # Should take first image
    
    def test_random_crop_invalid_size(self):
        """Test random crop with crop size larger than image."""
        crop = RandomCrop(50, 50)
        img = torch.randn(3, 32, 32)
        
        with pytest.raises(ValueError, match="Crop size .* larger than image size"):
            crop(img)
    
    def test_random_crop_invalid_shape(self):
        """Test random crop with invalid image shape."""
        crop = RandomCrop(5, 5)
        img = torch.randn(32)  # 1D tensor
        
        with pytest.raises(ValueError, match="Unexpected image shape"):
            crop(img)
    
    def test_random_crop_non_tensor_input(self):
        """Test random crop with non-tensor input."""
        crop = RandomCrop(5, 5)
        img = np.random.randn(3, 32, 32)  # NumPy array
        
        result = crop(img)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 5, 5)


class TestAnalyzePatchDistances:
    """Test patch distance analysis function."""
    
    def test_analyze_patch_distances_rgb_dataset(self):
        """Test analysis on RGB dataset."""
        dataset = MockDataset(num_samples=20, image_shape=(3, 32, 32))
        
        results = analyze_patch_distances(dataset, patch_sizes=[3, 6], num_samples=10, verbose=False)
        
        assert 'error' not in results
        assert '3x3' in results
        assert '6x6' in results
        
        # Check structure of results
        for patch_size in ['3x3', '6x6']:
            patch_result = results[patch_size]
            assert 'num_patches' in patch_result
            assert 'distance_stats' in patch_result
            assert 'distribution_fits' in patch_result
            assert 'radial_power_spectrum' in patch_result
            
            # Check distance stats
            stats = patch_result['distance_stats']
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
    
    def test_analyze_patch_distances_grayscale_dataset(self):
        """Test analysis on grayscale dataset (MNIST-like)."""
        dataset = MockGrayscaleDataset(num_samples=20, image_shape=(28, 28))
        
        results = analyze_patch_distances(dataset, patch_sizes=[3], num_samples=10, verbose=False)
        
        assert 'error' not in results
        assert '3x3' in results
    
    def test_analyze_patch_distances_empty_dataset(self):
        """Test analysis on empty dataset."""
        dataset = MockDataset(num_samples=0)
        
        results = analyze_patch_distances(dataset, patch_sizes=[3], num_samples=10, verbose=False)
        
        # Should handle empty dataset gracefully
        assert isinstance(results, dict)
    
    def test_analyze_patch_distances_large_patch_size(self):
        """Test analysis with patch size larger than image."""
        dataset = MockDataset(num_samples=10, image_shape=(3, 8, 8))
        
        results = analyze_patch_distances(dataset, patch_sizes=[10], num_samples=5, verbose=False)
        
        # Should skip patches that are too large
        assert '10x10' not in results or len(results) == 0
    
    def test_analyze_patch_distances_single_image_dataset(self):
        """Test analysis on dataset with only single image type."""
        dataset = MockDataset(num_samples=10, image_shape=(3, 32, 32), return_tuples=False)
        
        results = analyze_patch_distances(dataset, patch_sizes=[3], num_samples=5, verbose=False)
        
        assert isinstance(results, dict)


class TestPlotAndSaveResults:
    """Test visualization and saving function."""
    
    def test_plot_and_save_results_valid_data(self):
        """Test plotting and saving with valid results."""
        # Create mock results
        results = {
            '3x3': {
                'num_patches': 50,
                'distance_stats': {
                    'mean': 2.5,
                    'std': 0.8,
                    'min': 1.0,
                    'max': 5.0
                },
                'distribution_fits': {
                    'weibull': {
                        'params': [1.5, 2.0, 0.5],
                        'aic': 100.0,
                        'ks_statistic': 0.05,
                        'ks_pvalue': 0.8
                    },
                    'gumbel': {
                        'params': [2.0, 1.0],
                        'aic': 105.0,
                        'ks_statistic': 0.07,
                        'ks_pvalue': 0.6
                    },
                    'gev': {
                        'params': [0.1, 2.0, 1.0],
                        'aic': 98.0,
                        'ks_statistic': 0.04,
                        'ks_pvalue': 0.9
                    }
                },
                'radial_power_spectrum': [10.0, 8.0, 6.0, 4.0]
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock plt.show to avoid displaying plots during testing
            with patch('matplotlib.pyplot.show'):
                plot_and_save_results(results, 'test_dataset', temp_dir)
            
            # Check that files were created
            save_path = Path(temp_dir)
            assert (save_path / 'test_dataset_comprehensive_analysis.png').exists()
            assert (save_path / 'test_dataset_results.json').exists()
            
            # Check JSON content
            with open(save_path / 'test_dataset_results.json') as f:
                saved_results = json.load(f)
            
            assert '3x3' in saved_results
            assert saved_results['3x3']['num_patches'] == 50
    
    def test_plot_and_save_results_empty_results(self):
        """Test plotting with empty results."""
        results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('matplotlib.pyplot.show'):
                # Should handle empty results gracefully
                plot_and_save_results(results, 'test_dataset', temp_dir)
    
    def test_plot_and_save_results_error_results(self):
        """Test plotting with error results."""
        results = {
            '3x3': {'error': 'Some error occurred'}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('matplotlib.pyplot.show'):
                # Should handle error results gracefully
                plot_and_save_results(results, 'test_dataset', temp_dir)


class TestAnalyzeMultipleDatasets:
    """Test multi-dataset analysis function."""
    
    def test_analyze_multiple_datasets_success(self):
        """Test successful analysis of multiple datasets."""
        def mock_get_dataset(name, root, train):
            if name == 'dataset1':
                return MockDataset(20, (3, 32, 32)), {'name': 'dataset1'}
            elif name == 'dataset2':
                return MockGrayscaleDataset(20, (28, 28)), {'name': 'dataset2'}
            else:
                raise ValueError(f"Unknown dataset: {name}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('matplotlib.pyplot.show'):
                results = analyze_multiple_datasets(
                    ['dataset1', 'dataset2'], 
                    mock_get_dataset,
                    data_root=temp_dir,
                    patch_sizes=[3],
                    num_samples=5,
                    results_dir=temp_dir
                )
        
        assert 'dataset1' in results
        assert 'dataset2' in results
    
    def test_analyze_multiple_datasets_with_failures(self):
        """Test analysis with some dataset failures."""
        def mock_get_dataset(name, root, train):
            if name == 'good_dataset':
                return MockDataset(20, (3, 32, 32)), {'name': 'good_dataset'}
            else:
                raise ValueError(f"Failed to load {name}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('matplotlib.pyplot.show'):
                results = analyze_multiple_datasets(
                    ['good_dataset', 'bad_dataset'],
                    mock_get_dataset,
                    data_root=temp_dir,
                    patch_sizes=[3],
                    num_samples=5,
                    results_dir=temp_dir
                )
        
        assert 'good_dataset' in results
        assert 'bad_dataset' in results
        assert 'error' not in results['good_dataset']
        assert 'error' in results['bad_dataset']


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_rgb_dataset(self):
        """Test complete analysis pipeline on RGB dataset."""
        dataset = MockDataset(num_samples=30, image_shape=(3, 32, 32))
        
        # Run analysis
        results = analyze_patch_distances(dataset, patch_sizes=[3, 6], num_samples=20, verbose=False)
        
        assert 'error' not in results
        assert len(results) == 2
        
        # Test visualization
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('matplotlib.pyplot.show'):
                plot_and_save_results(results, 'integration_test', temp_dir)
            
            # Verify outputs
            save_path = Path(temp_dir)
            assert (save_path / 'integration_test_comprehensive_analysis.png').exists()
            assert (save_path / 'integration_test_results.json').exists()
    
    def test_full_pipeline_grayscale_dataset(self):
        """Test complete analysis pipeline on grayscale dataset."""
        dataset = MockGrayscaleDataset(num_samples=30, image_shape=(28, 28))
        
        # Run analysis
        results = analyze_patch_distances(dataset, patch_sizes=[3], num_samples=20, verbose=False)
        
        assert 'error' not in results
        assert '3x3' in results
        
        # Test visualization
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('matplotlib.pyplot.show'):
                plot_and_save_results(results, 'grayscale_test', temp_dir)


if __name__ == '__main__':
    pytest.main([__file__]) 