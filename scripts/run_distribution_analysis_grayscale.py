#!/usr/bin/env python3
"""Script to run distribution comparison analysis for all datasets with grayscale conversion."""

import os
import sys
sys.path.append('../src')

from src.utils.data import get_dataset
from src.utils.patch_statistics import analyze_multiple_datasets
import torch
import torchvision.transforms as transforms

def convert_to_grayscale(dataset):
    """Convert a dataset to grayscale."""
    grayscale_transform = transforms.Grayscale(num_output_channels=1)
    
    # Create a new dataset with grayscale conversion
    class GrayscaleDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset):
            self.dataset = original_dataset
            self.transform = grayscale_transform
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            if isinstance(item, tuple):
                # Dataset returns (image, label)
                image, label = item
                if torch.is_tensor(image):
                    # Convert tensor to PIL for transform, then back to tensor
                    if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
                        image_pil = transforms.ToPILImage()(image)
                        gray_pil = self.transform(image_pil)
                        gray_tensor = transforms.ToTensor()(gray_pil)
                        return gray_tensor, label
                    else:
                        # Already grayscale or different format
                        return image, label
                else:
                    # PIL Image
                    gray_image = self.transform(image)
                    return gray_image, label
            else:
                # Dataset returns just image
                image = item
                if torch.is_tensor(image):
                    if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
                        image_pil = transforms.ToPILImage()(image)
                        gray_pil = self.transform(image_pil)
                        gray_tensor = transforms.ToTensor()(gray_pil)
                        return gray_tensor
                    else:
                        return image
                else:
                    # PIL Image
                    return self.transform(image)
    
    return GrayscaleDataset(dataset)

def get_grayscale_dataset(dataset_name, data_root, train=True):
    """Get dataset and convert to grayscale."""
    print(f"Loading {dataset_name} dataset...")
    original_dataset, metadata = get_dataset(dataset_name, data_root, train)
    print(f"Converting {dataset_name} to grayscale...")
    grayscale_dataset = convert_to_grayscale(original_dataset)
    return grayscale_dataset, metadata

def main():
    """Run distribution analysis for all available datasets with grayscale conversion."""
    
    # List of datasets to analyze
    dataset_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CelebA']
    
    # Configuration
    patch_sizes = [3, 6, 10]
    num_samples = 200  # Reduced for faster computation but still good statistics
    results_dir = '../results'
    data_root = '../data'
    
    print("Starting GRAYSCALE distribution comparison analysis...")
    print(f"Original datasets: {dataset_names}")
    print(f"Grayscale dataset folders: {[f'grayscale_{name}' for name in dataset_names]}")
    print(f"Patch sizes: {patch_sizes}")
    print(f"Samples per dataset: {num_samples}")
    print(f"Results directory: {results_dir}")
    print("NOTE: All datasets will be converted to grayscale before analysis")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Create grayscale dataset names by prepending "grayscale_"
    grayscale_dataset_names = [f"grayscale_{name}" for name in dataset_names]
    
    # Create a mapping for the get_dataset function
    def get_grayscale_dataset_mapped(dataset_name, root=None, train=True):
        # Remove the "grayscale_" prefix to get the original dataset name
        original_name = dataset_name.replace("grayscale_", "")
        return get_grayscale_dataset(original_name, root, train)
    
    # Run analysis with grayscale conversion
    try:
        results = analyze_multiple_datasets(
            dataset_names=grayscale_dataset_names,
            get_dataset_func=get_grayscale_dataset_mapped,
            data_root=data_root,
            patch_sizes=patch_sizes,
            num_samples=num_samples,
            results_dir=results_dir
        )
        
        print("\n" + "="*60)
        print("GRAYSCALE DISTRIBUTION ANALYSIS COMPLETE")
        print("="*60)
        
        # Print summary
        for dataset_name, dataset_results in results.items():
            if 'error' in dataset_results:
                print(f"❌ {dataset_name} (grayscale): Failed - {dataset_results['error']}")
            else:
                print(f"✅ {dataset_name} (grayscale): Success")
                for patch_size in patch_sizes:
                    if patch_size in dataset_results:
                        patch_results = dataset_results[patch_size]
                        if 'distribution_fits' in patch_results and 'error' not in patch_results['distribution_fits']:
                            dist_fits = patch_results['distribution_fits']
                            if 'weibull' in dist_fits and 'gumbel' in dist_fits:
                                weibull_aic = dist_fits['weibull']['aic']
                                gumbel_aic = dist_fits['gumbel']['aic']
                                better_fit = "Weibull" if weibull_aic < gumbel_aic else "Gumbel"
                                print(f"   📊 {patch_size}x{patch_size}: {better_fit} better fit (AIC: {min(weibull_aic, gumbel_aic):.1f})")
        
        print(f"\n📁 Grayscale results saved to: {results_dir}/")
        print("📈 Grayscale distribution comparison plots created for each dataset")
        print("📂 Grayscale results are in folders prefixed with 'grayscale_'")
        print(f"\n💡 Compare with original color results in the same directory: {results_dir}/")
        
    except Exception as e:
        print(f"Error running grayscale analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 