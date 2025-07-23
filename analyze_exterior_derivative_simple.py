#!/usr/bin/env python3
"""
Simplified Exterior Derivative Analysis Script - 64x64 ResNet vs ELS

This script analyzes the exterior derivative of score function models 
during the reverse diffusion process, comparing 64x64 ResNet and ELS only.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.models import DDIM, MinimalResNet
from src.utils.data import get_metadata
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.idealscore import LocalEquivBordersScoreModule, ScheduledScoreMachine
from src.utils.exterior_derivative import compute_exterior_derivative, exterior_derivative_magnitude


def get_dataset_64x64(name, root='./data', train=True):
    """Modified dataset loader for 64x64 resolution."""
    name_lower = name.lower()
    
    # Get base metadata but override image size
    metadata = get_metadata(name)
    metadata['image_size'] = 64  # Override to 64x64
    
    # Create transform for 64x64
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=metadata['mean'], std=metadata['std'])
    ])
    
    if name_lower == 'celeba':
        dataset = datasets.CelebA(
            root=root,
            split='train' if train else 'valid',
            download=True,
            transform=transform
        )
    elif name_lower == 'cifar10':
        dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    elif name_lower == 'mnist':
        dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    elif name_lower == 'fashionmnist' or name_lower == 'fashion_mnist':
        dataset = datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset for 64x64: {name}")
    
    return dataset, metadata


def create_score_function_wrapper(model, t_value, device, in_channels, image_size, model_type='neural'):
    """Create a score function wrapper with proper shape handling."""
    def score_fn(x_flat):
        # Get batch size from input
        batch_size = x_flat.shape[0]
        
        # Reshape flattened input back to image shape
        x_img = x_flat.view(batch_size, in_channels, image_size, image_size)
        
        # Get score prediction based on model type
        with torch.no_grad():
            if model_type == 'els':
                score = model.backbone(t_value, x_img, device=device)
            else:
                score = model.backbone(t_value, x_img)
        
        # Return flattened score but keep batch dimension
        return score.view(batch_size, -1)
    
    return score_fn


def analyze_single_realization(dataset_name='cifar10', nsteps=20, noise_seed=42, 
                             save_results=True, output_dir='./results'):
    """
    Analyze exterior derivative for 64x64 ResNet vs ELS comparison.
    """
    print(f"🔬 Analyzing Exterior Derivative for {dataset_name.upper()} (64x64 ResNet vs ELS)")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load 64x64 dataset
    print(f"\n📁 Loading {dataset_name} dataset at 64x64 resolution...")
    dataset, metadata = get_dataset_64x64(dataset_name, root='./data')
    in_channels = metadata['num_channels']
    image_size = 64  # Fixed at 64x64
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"Image size: {image_size}x{image_size}, Channels: {in_channels}")
    
    # Create models - only 64x64 ResNet and ELS
    print("\n🧠 Creating models...")
    models = {}
    
    # Check for existing 64x64 ResNet checkpoint
    checkpoint_pattern = f'./checkpoints/backbone_{metadata["name"].upper()}_ResNet_*_64x64*.pt'
    import glob
    resnet_checkpoints = glob.glob(checkpoint_pattern)
    
    # 64x64 ResNet model
    try:
        if resnet_checkpoints:
            # Load pre-trained 64x64 ResNet
            print(f"Loading pre-trained 64x64 ResNet from {resnet_checkpoints[0]}")
            models['resnet_64x64'] = torch.load(resnet_checkpoints[0], map_location=device, weights_only=False)
            models['resnet_64x64'].to(device)
            models['resnet_64x64'].eval()
            print("✅ Pre-trained 64x64 ResNet model loaded")
        else:
            # Create new 64x64 ResNet with appropriate architecture
            print("Creating new 64x64 ResNet model...")
            resnet = MinimalResNet(
                channels=in_channels, 
                conditional=False, 
                emb_dim=256,  # Larger embedding for 64x64
                num_layers=8,  # 8 layers as requested
                kernel_size=3
            )
            models['resnet_64x64'] = DDIM(
                backbone=resnet, 
                in_channels=in_channels, 
                default_imsize=image_size,
                noise_schedule=cosine_noise_schedule
            )
            models['resnet_64x64'].to(device)
            models['resnet_64x64'].eval()
            print("✅ New 64x64 ResNet model created")
    except Exception as e:
        print(f"❌ 64x64 ResNet model failed: {e}")
    
    # ELS model
    try:
        # Use subset for speed but ensure it's representative
        subset_size = min(500, len(dataset))  # Larger subset for 64x64
        subset_indices = list(range(subset_size))
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        
        print(f"Creating ELS model with {subset_size} samples...")
        els_backbone = LocalEquivBordersScoreModule(
            subset_dataset,
            kernel_size=5,  # Larger kernel for 64x64
            batch_size=16,  # Smaller batch for memory
            image_size=image_size,
            channels=in_channels,
            schedule=cosine_noise_schedule,
            max_samples=subset_size
        )
        
        models['els'] = ScheduledScoreMachine(
            els_backbone,
            in_channels=in_channels,
            imsize=image_size,
            noise_schedule=cosine_noise_schedule,
            score_backbone=True
        )
        models['els'].to(device)
        print("✅ ELS model created")
    except Exception as e:
        print(f"❌ ELS model failed: {e}")
        print("   (ELS setup can be complex for large images)")
    
    # Generate test noise
    print(f"\n🎲 Generating test noise (seed={noise_seed})...")
    torch.manual_seed(noise_seed)
    x_init = torch.randn(1, in_channels, image_size, image_size, device=device)
    
    # Analyze exterior derivatives
    print(f"\n🔍 Computing exterior derivatives over {nsteps} steps...")
    results = {}
    
    for model_name, model in models.items():
        print(f"\nAnalyzing {model_name.upper()}...")
        
        timesteps = []
        ed_magnitudes = []
        
        # Test different diffusion timesteps
        for step in range(nsteps, 0, -1):
            t_val = step / nsteps
            t = torch.tensor([t_val], device=device)
            
            try:
                # Create score function wrapper
                model_type = 'els' if model_name == 'els' else 'neural'
                score_fn = create_score_function_wrapper(
                    model, t, device, in_channels, image_size, model_type
                )
                
                # Flatten input for exterior derivative computation
                x_flat = x_init.view(1, -1)
                
                # Compute exterior derivative
                exterior_deriv = compute_exterior_derivative(x_flat, score_fn)
                ed_magnitude = exterior_derivative_magnitude(exterior_deriv)
                
                timesteps.append(step)
                ed_magnitudes.append(ed_magnitude.item())
                
                if step % 5 == 0:  # Print every 5 steps
                    print(f"  Step {step:2d}: ED magnitude = {ed_magnitude.item():.6f}")
                
            except Exception as e:
                print(f"  Step {step:2d}: Failed - {e}")
                timesteps.append(step)
                ed_magnitudes.append(0.0)
        
        results[model_name] = {
            'timesteps': timesteps,
            'ed_magnitudes': ed_magnitudes,
            'mean_ed': np.mean(ed_magnitudes),
            'std_ed': np.std(ed_magnitudes),
            'max_ed': np.max(ed_magnitudes)
        }
        
        print(f"  Summary: mean = {results[model_name]['mean_ed']:.6f}, "
              f"std = {results[model_name]['std_ed']:.6f}, "
              f"max = {results[model_name]['max_ed']:.6f}")
    
    # Generate final images using each model
    print("\n🖼️  Generating final images...")
    generated_images = {}
    
    # Prepare class labels for conditional models if needed
    label = None
    # Note: For simplicity, assume unconditional generation for 64x64 models
    
    for model_name, model in models.items():
        try:
            print(f"Generating image with {model_name}...")
            
            if model_name == 'els':
                # ELS uses different interface
                with torch.no_grad():
                    generated_img = model(x_init.clone(), nsteps=20, device=device).clamp(-1, 1)
                    generated_images[model_name] = generated_img.squeeze().cpu().numpy()
                print(f"✅ {model_name} image generated")
            else:
                # Neural network models (ResNet)
                with torch.no_grad():
                    generated_img = model.sample(x=x_init.clone(), nsteps=20, label=label).clamp(-1, 1)
                    generated_images[model_name] = generated_img.squeeze().cpu().numpy()
                print(f"✅ {model_name} image generated")
                
        except Exception as e:
            print(f"❌ {model_name} generation failed: {e}")
            generated_images[model_name] = None
    
    # Create visualizations with generated images (similar to demo)
    print("\n📊 Creating visualizations...")
    
    # Create figure with grid layout similar to demo
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{dataset_name.upper()} - 64x64 ResNet vs ELS Comparison', fontsize=18)
    
    # Helper function to prepare images for display
    def prepare_image_for_display(img):
        """Convert image from tensor format to matplotlib format"""
        if img.ndim == 3:
            if img.shape[0] in [1, 3]:  # Channel first format
                img = img.transpose(1, 2, 0)  # Convert to channel last
            if img.shape[2] == 1:  # Grayscale with channel dimension
                img = img.squeeze(2)  # Remove channel dimension
        
        # Normalize from [-1, 1] to [0, 1] if needed
        if img.min() < 0:
            img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        return img
    
    # Plot 1: ED magnitude vs timestep (top row, spanning two columns)
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, fig=fig)
    colors = ['#1f77b4', '#ff7f0e']  # Blue for ResNet, Orange for ELS
    for i, (model_name, data) in enumerate(results.items()):
        display_name = '64x64 ResNet' if 'resnet' in model_name else 'ELS'
        ax_main.plot(data['timesteps'], data['ed_magnitudes'], 
                    marker='o', linewidth=3, label=display_name, alpha=0.8,
                    color=colors[i % len(colors)], markersize=8)
    
    ax_main.set_xlabel('Diffusion Timestep', fontsize=14)
    ax_main.set_ylabel('Exterior Derivative Magnitude', fontsize=14)
    ax_main.set_title('Exterior Derivative During Reverse Process', fontsize=16)
    ax_main.legend(fontsize=14)
    ax_main.grid(True, alpha=0.3)
    
    # Plot 2: Initial noise (top right)
    ax_noise = plt.subplot2grid((3, 3), (0, 2), fig=fig)
    test_img = x_init.squeeze().cpu().numpy()
    display_noise = prepare_image_for_display(test_img.copy())
    
    if display_noise.ndim == 2:  # Grayscale
        ax_noise.imshow(display_noise, cmap='gray')
    else:  # RGB
        ax_noise.imshow(display_noise)
    
    ax_noise.set_title(f'Initial Noise 64x64\n(seed={noise_seed})', fontsize=14)
    ax_noise.axis('off')
    
    # Plot 3: Log scale ED (middle row, spanning two columns)
    ax_log = plt.subplot2grid((3, 3), (1, 0), colspan=2, fig=fig)
    for i, (model_name, data) in enumerate(results.items()):
        display_name = '64x64 ResNet' if 'resnet' in model_name else 'ELS'
        ed_vals = np.array(data['ed_magnitudes'])
        ed_vals = ed_vals[ed_vals > 0]  # Remove zeros for log plot
        steps = np.array(data['timesteps'])[:len(ed_vals)]
        if len(ed_vals) > 0:
            ax_log.semilogy(steps, ed_vals, marker='o', linewidth=3, 
                           label=display_name, alpha=0.8,
                           color=colors[i % len(colors)], markersize=8)
    
    ax_log.set_xlabel('Diffusion Timestep', fontsize=14)
    ax_log.set_ylabel('Exterior Derivative Magnitude (log)', fontsize=14)
    ax_log.set_title('Exterior Derivative (Log Scale)', fontsize=16)
    ax_log.legend(fontsize=14)
    ax_log.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics (middle right)
    ax_stats = plt.subplot2grid((3, 3), (1, 2), fig=fig)
    model_names = list(results.keys())
    display_names = ['64x64 ResNet' if 'resnet' in name else 'ELS' for name in model_names]
    means = [results[name]['mean_ed'] for name in model_names]
    stds = [results[name]['std_ed'] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    bars = ax_stats.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                       color=colors[:len(model_names)])
    ax_stats.set_xlabel('Model', fontsize=12)
    ax_stats.set_ylabel('Mean ED Magnitude', fontsize=12)
    ax_stats.set_title('Mean Comparison', fontsize=14)
    ax_stats.set_xticks(x_pos)
    ax_stats.set_xticklabels([name.split()[0] for name in display_names], fontsize=10)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        ax_stats.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.01,
                     f'{mean_val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 5 & 6: Generated images (bottom row)
    model_names_list = list(results.keys())
    model_titles = ['64x64 ResNet Generated', 'ELS Generated']
    
    for i, model_name in enumerate(model_names_list):
        ax = plt.subplot2grid((3, 3), (2, i), fig=fig)
        
        if model_name in generated_images and generated_images[model_name] is not None:
            img = generated_images[model_name]
            display_img = prepare_image_for_display(img.copy())
            
            if display_img.ndim == 2:  # Grayscale
                ax.imshow(display_img, cmap='gray')
            else:  # RGB
                ax.imshow(display_img)
        else:
            # Show placeholder if generation failed
            ax.text(0.5, 0.5, 'Generation\nFailed', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        display_name = '64x64 ResNet' if 'resnet' in model_name else 'ELS'
        ax.set_title(f'{display_name}\nGenerated', fontsize=14)
        ax.axis('off')
    
    # If only one model, center the generated image
    if len(model_names_list) == 1:
        ax_empty = plt.subplot2grid((3, 3), (2, 1), fig=fig)
        ax_empty.axis('off')
    
    plt.tight_layout()
    
    # Save results
    if save_results:
        output_path = Path(output_dir) / f'exterior_derivative_{dataset_name}_64x64'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save plot with same naming as 32x32 version
        plt.savefig(output_path / f'analysis_64x64_resnet_vs_els_seed{noise_seed}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n💾 Plot saved to {output_path}")
        
        # Save numerical data
        torch.save(results, output_path / f'results_64x64_resnet_vs_els_seed{noise_seed}.pt')
        print(f"💾 Data saved to {output_path}")
        
        # Save summary
        summary = {
            'dataset': dataset_name,
            'analysis_type': '64x64_resnet_vs_els',
            'nsteps': nsteps,
            'noise_seed': noise_seed,
            'image_size': image_size,
            'in_channels': in_channels,
            'models_analyzed': list(results.keys()),
            'comparison': '64x64 ResNet vs ELS',
            'summary_stats': {name: {
                'mean_ed': float(data['mean_ed']),
                'std_ed': float(data['std_ed']),
                'max_ed': float(data['max_ed'])
            } for name, data in results.items()}
        }
        
        import json
        with open(output_path / f'summary_64x64_resnet_vs_els_seed{noise_seed}.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    plt.show()
    
    # Print final summary
    print("\n📈 FINAL SUMMARY - 64x64 ResNet vs ELS")
    print("-" * 50)
    for model_name, data in results.items():
        display_name = '64x64 ResNet' if 'resnet' in model_name else 'ELS'
        print(f"{display_name:12s}: mean={data['mean_ed']:.6f} ± {data['std_ed']:.6f}, "
              f"max={data['max_ed']:.6f}")
    
    print(f"\n✅ 64x64 ResNet vs ELS analysis completed for {dataset_name} (seed={noise_seed})")
    print("\n🖼️  The visualization includes:")
    print("  • Top row: Exterior derivative analysis + initial noise")
    print("  • Middle row: Log scale ED + summary statistics")
    print("  • Bottom row: Generated images from 64x64 ResNet and ELS")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='64x64 ResNet vs ELS Exterior Derivative Analysis')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist', 'fashionmnist', 'celeba'],
                       help='Dataset to analyze with 64x64 ResNet vs ELS comparison')
    parser.add_argument('--nsteps', type=int, default=20,
                       help='Number of diffusion steps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for noise generation')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to disk')
    
    args = parser.parse_args()
    
    print("🧪 64x64 ResNet vs ELS Exterior Derivative Analysis")
    print("This script compares exterior derivative behavior between 64x64 ResNet and ELS models")
    print("Generates comprehensive visualizations including exterior derivative analysis and generated images")
    print("Similar to exterior_derivative_demo.py but focused on 64x64 ResNet vs ELS comparison")
    print()
    
    try:
        results = analyze_single_realization(
            dataset_name=args.dataset,
            nsteps=args.nsteps,
            noise_seed=args.seed,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
        
        print("\n🎉 64x64 ResNet vs ELS analysis completed successfully!")
        print("📊 Comprehensive visualization with generated images saved!")
        print("🖼️  Similar to exterior_derivative_demo.py but for 64x64 ResNet vs ELS")
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 