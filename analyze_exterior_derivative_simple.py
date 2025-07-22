#!/usr/bin/env python3
"""
Simplified Exterior Derivative Analysis Script

This script analyzes the exterior derivative of score function models 
during the reverse diffusion process, with proper shape handling.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import DDIM, MinimalResNet, MinimalUNet
from src.utils.data import get_dataset
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.idealscore import LocalEquivBordersScoreModule, ScheduledScoreMachine
from src.utils.exterior_derivative import compute_exterior_derivative, exterior_derivative_magnitude


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


def analyze_single_realization(dataset_name='mnist', nsteps=20, noise_seed=42, 
                             save_results=True, output_dir='./results'):
    """
    Analyze exterior derivative for a single noise realization.
    """
    print(f"🔬 Analyzing Exterior Derivative for {dataset_name.upper()}")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\n📁 Loading {dataset_name} dataset...")
    dataset, metadata = get_dataset(dataset_name, root='./data')
    in_channels = metadata['num_channels']
    image_size = metadata['image_size']
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"Image size: {image_size}x{image_size}, Channels: {in_channels}")
    
    # Create models
    print("\n🧠 Creating models...")
    models = {}
    
    # UNet model
    try:
        unet = MinimalUNet(channels=in_channels, conditional=False, emb_dim=128)
        models['unet'] = DDIM(backbone=unet, in_channels=in_channels, default_imsize=image_size)
        models['unet'].to(device)
        models['unet'].eval()
        print("✅ UNet model created")
    except Exception as e:
        print(f"❌ UNet model failed: {e}")
    
    # ResNet model
    try:
        resnet = MinimalResNet(channels=in_channels, conditional=False, emb_dim=128)
        models['resnet'] = DDIM(backbone=resnet, in_channels=in_channels, default_imsize=image_size)
        models['resnet'].to(device)
        models['resnet'].eval()
        print("✅ ResNet model created")
    except Exception as e:
        print(f"❌ ResNet model failed: {e}")
    
    # ELS model (simplified version)
    try:
        # Use small subset for speed
        subset_size = min(100, len(dataset))
        subset_indices = list(range(subset_size))
        subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
        
        els_backbone = LocalEquivBordersScoreModule(
            subset_dataset,
            kernel_size=3,
            batch_size=32,
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
        print("   (This is normal - ELS requires more setup)")
    
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
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: ED magnitude vs timestep
    ax = axes[0, 0]
    for model_name, data in results.items():
        ax.plot(data['timesteps'], data['ed_magnitudes'], 
               marker='o', linewidth=2, label=model_name.upper(), alpha=0.8)
    
    ax.set_xlabel('Diffusion Timestep')
    ax.set_ylabel('Exterior Derivative Magnitude')
    ax.set_title('Exterior Derivative During Reverse Process')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Log scale
    ax = axes[0, 1]
    for model_name, data in results.items():
        ed_vals = np.array(data['ed_magnitudes'])
        ed_vals = ed_vals[ed_vals > 0]  # Remove zeros for log plot
        steps = np.array(data['timesteps'])[:len(ed_vals)]
        if len(ed_vals) > 0:
            ax.semilogy(steps, ed_vals, marker='o', linewidth=2, 
                       label=model_name.upper(), alpha=0.8)
    
    ax.set_xlabel('Diffusion Timestep')
    ax.set_ylabel('Exterior Derivative Magnitude (log)')
    ax.set_title('Exterior Derivative (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Summary statistics
    ax = axes[1, 0]
    model_names = list(results.keys())
    means = [results[name]['mean_ed'] for name in model_names]
    stds = [results[name]['std_ed'] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Model')
    ax.set_ylabel('Mean Exterior Derivative Magnitude')
    ax.set_title('Mean ED Magnitude Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name.upper() for name in model_names])
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.01,
               f'{mean_val:.4f}', ha='center', va='bottom')
    
    # Plot 4: Test noise
    ax = axes[1, 1]
    test_img = x_init.squeeze().cpu().numpy()
    if test_img.ndim == 3 and test_img.shape[0] in [1, 3]:
        test_img = test_img.transpose(1, 2, 0)
    if test_img.ndim == 3 and test_img.shape[2] == 1:
        test_img = test_img.squeeze(2)
    
    if test_img.ndim == 3:
        ax.imshow(test_img)
    else:
        ax.imshow(test_img, cmap='gray')
    ax.set_title(f'Initial Noise (seed={noise_seed})')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save results
    if save_results:
        output_path = Path(output_dir) / f'exterior_derivative_{dataset_name}'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save plot
        plt.savefig(output_path / f'analysis_single_seed{noise_seed}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n💾 Plot saved to {output_path}")
        
        # Save numerical data
        torch.save(results, output_path / f'results_single_seed{noise_seed}.pt')
        print(f"💾 Data saved to {output_path}")
        
        # Save summary
        summary = {
            'dataset': dataset_name,
            'nsteps': nsteps,
            'noise_seed': noise_seed,
            'image_size': image_size,
            'in_channels': in_channels,
            'models_analyzed': list(results.keys()),
            'summary_stats': {name: {
                'mean_ed': float(data['mean_ed']),
                'std_ed': float(data['std_ed']),
                'max_ed': float(data['max_ed'])
            } for name, data in results.items()}
        }
        
        import json
        with open(output_path / f'summary_seed{noise_seed}.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    plt.show()
    
    # Print final summary
    print("\n📈 FINAL SUMMARY")
    print("-" * 40)
    for model_name, data in results.items():
        print(f"{model_name.upper():8s}: mean={data['mean_ed']:.6f} ± {data['std_ed']:.6f}, "
              f"max={data['max_ed']:.6f}")
    
    print(f"\n✅ Analysis completed for {dataset_name} (seed={noise_seed})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Simplified Exterior Derivative Analysis')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10', 'fashionmnist'],
                       help='Dataset to analyze')
    parser.add_argument('--nsteps', type=int, default=20,
                       help='Number of diffusion steps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for noise generation')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to disk')
    
    args = parser.parse_args()
    
    print("🧪 Simplified Exterior Derivative Analysis")
    print("This script analyzes exterior derivative behavior of score functions")
    print()
    
    try:
        results = analyze_single_realization(
            dataset_name=args.dataset,
            nsteps=args.nsteps,
            noise_seed=args.seed,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
        
        print("\n🎉 Analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 