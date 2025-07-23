#!/usr/bin/env python3
"""
64x64 Exterior Derivative Demo - Testing Locality Breakdown Hypothesis

This script tests the hypothesis that:
1. CNNs maintain zero curl at 64x64 (convolution operations remain closed)
2. ELS develops non-zero curl at 64x64 (pixels fall outside patch distribution)

Uses untrained models for controlled comparison since we don't have 64x64 checkpoints.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import DDIM, MinimalResNet, MinimalUNet
from src.utils.data import get_dataset, get_metadata
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.idealscore import LocalEquivBordersScoreModule, ScheduledScoreMachine
from src.utils.exterior_derivative import compute_exterior_derivative, exterior_derivative_magnitude


def get_dataset_64x64(name, root='./data', train=True):
    """Modified dataset loader for 64x64 resolution."""
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    
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
    
    if name_lower == 'mnist':
        dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    elif name_lower == 'cifar10':
        dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    elif name_lower == 'fashionmnist' or name_lower == 'fashion_mnist':
        dataset = datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset for 64x64: {name}")
    
    return dataset, metadata


def create_score_function_64x64(model, t_value, device, in_channels=1, model_type='neural'):
    """Create score function wrapper for 64x64 models."""
    def score_fn(x_flat):
        batch_size = x_flat.shape[0]
        x_img = x_flat.view(batch_size, in_channels, 64, 64)  # 64x64 images
        
        with torch.no_grad():
            if model_type == 'els':
                score = model.backbone(t_value, x_img, device=device)
            else:
                score = model.backbone(t_value, x_img)
        
        return score.view(batch_size, -1)
    
    return score_fn


def demo_64x64_analysis():
    """Test exterior derivative at 64x64 to validate locality breakdown hypothesis."""
    
    print("🔬 64×64 Exterior Derivative Analysis - Testing Locality Breakdown")
    print("=" * 70)
    print("Hypothesis: CNNs remain closed, ELS develops non-zero curl")
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load 64x64 dataset 
    dataset_name = 'cifar10'
    print(f"\n📁 Loading {dataset_name.upper()} at 64×64 resolution...")
    dataset, metadata = get_dataset_64x64(dataset_name, root='./data')
    in_channels = metadata['num_channels']
    image_size = 64
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Image size: {image_size}×{image_size}, Channels: {in_channels}")
    
    # Create UNTRAINED models for controlled comparison
    print("\n🧠 Creating untrained models for controlled testing...")
    
    # UNet model (untrained)
    unet = MinimalUNet(channels=in_channels, conditional=False, emb_dim=128)
    unet_model = DDIM(backbone=unet, in_channels=in_channels, default_imsize=image_size)
    unet_model.to(device)
    unet_model.eval()
    print("✅ UNet model created (untrained)")
    
    # ResNet model (untrained)
    resnet = MinimalResNet(channels=in_channels, conditional=False, emb_dim=128)
    resnet_model = DDIM(backbone=resnet, in_channels=in_channels, default_imsize=image_size)
    resnet_model.to(device)
    resnet_model.eval()
    print("✅ ResNet model created (untrained)")
    
    # Generate test noise at 64x64
    print("\n🎲 Generating 64×64 test noise...")
    torch.manual_seed(42)  # For reproducibility
    x_test = torch.randn(1, in_channels, image_size, image_size, device=device)
    print(f"Test noise shape: {x_test.shape}")
    
    # Create ELS model with larger patches for 64x64
    print("\n🔬 Creating ELS model with larger patches...")
    els_model = None
    try:
        # Use larger patch sizes appropriate for 64x64 images
        # Key hypothesis: larger patches will cause locality breakdown
        kernel_size = 5  # Moderate increase from 3×3 (7×7 too slow)
        max_samples = min(500, len(dataset))  # Reduced for faster computation
        
        print(f"Using kernel size: {kernel_size}×{kernel_size}")
        print(f"Expected locality breakdown with larger patches relative to features")
        
        els_backbone = LocalEquivBordersScoreModule(
            dataset,
            kernel_size=kernel_size,  # Larger patches
            batch_size=32,  # Smaller batch for memory
            image_size=image_size,
            channels=in_channels,
            schedule=cosine_noise_schedule,
            max_samples=max_samples,
            shuffle=False
        )
        
        els_model = ScheduledScoreMachine(
            els_backbone,
            in_channels=in_channels,
            imsize=image_size,
            noise_schedule=cosine_noise_schedule,
            score_backbone=True,
            scales=None  # No pre-calibrated scales for 64×64
        )
        els_model.to(device)
        print("✅ ELS model created successfully")
        print(f"   Using {max_samples} samples with {kernel_size}×{kernel_size} patches")
        
    except Exception as e:
        print(f"⚠️  ELS model creation failed: {e}")
        print("   (Continuing without ELS)")

    # Test at more diffusion times for better resolution
    timesteps_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = {
        'unet': {'timesteps': [], 'ed_magnitudes': []},
        'resnet': {'timesteps': [], 'ed_magnitudes': []},
        'els': {'timesteps': [], 'ed_magnitudes': []}
    }
    
    print("\n🔍 Computing exterior derivatives at 64×64...")
    print("Testing hypothesis: CNNs=0 curl, ELS>0 curl due to locality breakdown")
    
    for t_val in timesteps_to_test:
        print(f"\nAnalyzing at t = {t_val:.1f}")
        
        t = torch.tensor([t_val], device=device)
        
        # Test UNet (should remain zero curl)
        try:
            score_fn_unet = create_score_function_64x64(unet_model, t, device, in_channels, 'neural')
            x_flat = x_test.view(1, -1)
            
            exterior_deriv = compute_exterior_derivative(x_flat, score_fn_unet)
            ed_magnitude = exterior_derivative_magnitude(exterior_deriv)
            
            results['unet']['timesteps'].append(t_val)
            results['unet']['ed_magnitudes'].append(ed_magnitude.item())
            
            print(f"  UNet ED magnitude: {ed_magnitude.item():.8f}")
            
        except Exception as e:
            print(f"  UNet failed: {e}")
        
        # Test ResNet (should remain zero curl)
        try:
            score_fn_resnet = create_score_function_64x64(resnet_model, t, device, in_channels, 'neural')
            x_flat = x_test.view(1, -1)
            
            exterior_deriv = compute_exterior_derivative(x_flat, score_fn_resnet)
            ed_magnitude = exterior_derivative_magnitude(exterior_deriv)
            
            results['resnet']['timesteps'].append(t_val)
            results['resnet']['ed_magnitudes'].append(ed_magnitude.item())
            
            print(f"  ResNet ED magnitude: {ed_magnitude.item():.8f}")
            
        except Exception as e:
            print(f"  ResNet failed: {e}")
        
        # Test ELS (should develop non-zero curl)
        if els_model is not None:
            try:
                def score_fn_els(x_flat):
                    batch_size = x_flat.shape[0]
                    x_img = x_flat.view(batch_size, in_channels, 64, 64)
                    
                    with torch.no_grad():
                        score = els_model.backbone(t, x_img, device=device)
                    
                    return score.view(batch_size, -1)
                
                x_flat = x_test.view(1, -1)
                
                exterior_deriv = compute_exterior_derivative(x_flat, score_fn_els)
                ed_magnitude = exterior_derivative_magnitude(exterior_deriv)
                
                results['els']['timesteps'].append(t_val)
                results['els']['ed_magnitudes'].append(ed_magnitude.item())
                
                print(f"  ELS ED magnitude: {ed_magnitude.item():.8f} {'🔄' if ed_magnitude.item() > 1e-6 else ''}")
                
            except Exception as e:
                print(f"  ELS failed: {e}")
        else:
            print(f"  ELS skipped (not available)")
    
    # Generate images (these will be poor quality since models are untrained)
    print("\n🖼️  Generating sample images (will be poor - untrained models)...")
    generated_images = {}
    
    try:
        with torch.no_grad():
            unet_img = unet_model.sample(x=x_test.clone(), nsteps=50).clamp(-1, 1)
            generated_images['unet'] = unet_img.squeeze().cpu().numpy()
        print("✅ UNet image generated")
    except Exception as e:
        print(f"❌ UNet generation failed: {e}")
        generated_images['unet'] = None
    
    try:
        with torch.no_grad():
            resnet_img = resnet_model.sample(x=x_test.clone(), nsteps=50).clamp(-1, 1)
            generated_images['resnet'] = resnet_img.squeeze().cpu().numpy()
        print("✅ ResNet image generated")
    except Exception as e:
        print(f"❌ ResNet generation failed: {e}")
        generated_images['resnet'] = None
    
    if els_model is not None:
        try:
            with torch.no_grad():
                els_img = els_model(x_test.clone(), nsteps=50, device=device).clamp(-1, 1)
                generated_images['els'] = els_img.squeeze().cpu().numpy()
            print("✅ ELS image generated")
        except Exception as e:
            print(f"❌ ELS generation failed: {e}")
            generated_images['els'] = None
    else:
        generated_images['els'] = None

    # Create enhanced visualization focused on hypothesis testing
    print("\n📊 Creating hypothesis testing visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Main plot: Exterior derivative magnitudes (focus on hypothesis)
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    
    colors = {'unet': 'blue', 'resnet': 'orange', 'els': 'green'}
    markers = {'unet': 'o', 'resnet': 's', 'els': '^'}
    
    for model_name, data in results.items():
        if data['timesteps']:
            ax_main.plot(data['timesteps'], data['ed_magnitudes'], 
                        color=colors[model_name], marker=markers[model_name], 
                        linewidth=2, markersize=8, label=f'{model_name.upper()} (64×64)',
                        alpha=0.8)
    
    ax_main.set_xlabel('Diffusion Time (t)', fontsize=12)
    ax_main.set_ylabel('Exterior Derivative Magnitude', fontsize=12)
    ax_main.set_title('64×64 Exterior Derivative Analysis\nHypothesis: CNNs=closed, ELS=non-closed', fontsize=14, fontweight='bold')
    ax_main.legend(fontsize=11)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_yscale('log')  # Log scale to see small differences
    
    # Add hypothesis annotation
    ax_main.text(0.02, 0.98, 
                'Expected:\n• CNNs: ~0 (closed)\n• ELS: >0 (locality breakdown)', 
                transform=ax_main.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Initial noise (top right)
    ax_noise = plt.subplot2grid((3, 3), (0, 2))
    test_img = x_test.squeeze().cpu().numpy()
    
    def prepare_image_for_display(img):
        """Convert image from tensor format to matplotlib format"""
        if img.ndim == 3:
            if img.shape[0] in [1, 3]:
                img = img.transpose(1, 2, 0)
            if img.shape[2] == 1:
                img = img.squeeze(2)
        
        if img.min() < 0:
            img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        return img
    
    display_noise = prepare_image_for_display(test_img.copy())
    
    if display_noise.ndim == 2:
        ax_noise.imshow(display_noise, cmap='gray')
    else:
        ax_noise.imshow(display_noise)
    
    ax_noise.set_title('Initial Noise\n(64×64)', fontsize=10)
    ax_noise.axis('off')
    
    # Generated images (bottom row)
    model_names = ['unet', 'resnet', 'els']
    model_titles = ['UNet Generated\n(64×64)', 'ResNet Generated\n(64×64)', 'ELS Generated\n(64×64)']
    
    for i, (model_name, title) in enumerate(zip(model_names, model_titles)):
        ax = plt.subplot2grid((3, 3), (2, i))
        
        if generated_images[model_name] is not None:
            img = generated_images[model_name]
            display_img = prepare_image_for_display(img.copy())
            
            if display_img.ndim == 2:
                ax.imshow(display_img, cmap='gray')
            else:
                ax.imshow(display_img)
        else:
            ax.text(0.5, 0.5, 'Generation\nFailed', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Log scale comparison subplot
    ax_log = plt.subplot2grid((3, 3), (1, 2))
    
    # Create bar chart of mean magnitudes
    model_means = {}
    for model_name, data in results.items():
        if data['ed_magnitudes']:
            model_means[model_name] = np.mean(data['ed_magnitudes'])
    
    if model_means:
        models = list(model_means.keys())
        means = list(model_means.values())
        bars = ax_log.bar(models, means, color=[colors[m] for m in models], alpha=0.7)
        ax_log.set_ylabel('Mean ED Magnitude', fontsize=10)
        ax_log.set_title('Mean Exterior\nDerivative', fontsize=10)
        ax_log.set_yscale('log')
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax_log.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path('./results/exterior_derivative_64x64')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'locality_breakdown_hypothesis_test.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Results saved to {output_dir}")
    
    # Save numerical results
    torch.save(results, output_dir / 'hypothesis_test_data.pt')
    
    # Print detailed hypothesis testing results
    print("\n" + "="*70)
    print("🔬 HYPOTHESIS TEST RESULTS")
    print("="*70)
    print("Hypothesis: At 64×64 resolution:")
    print("  • CNNs should have ~0 curl (convolution operations remain closed)")
    print("  • ELS should develop >0 curl (locality breakdown with larger patches)")
    print()
    
    for model_name, data in results.items():
        if data['ed_magnitudes']:
            mean_ed = np.mean(data['ed_magnitudes'])
            max_ed = np.max(data['ed_magnitudes'])
            min_ed = np.min(data['ed_magnitudes'])
            
            print(f"{model_name.upper():6s}: mean={mean_ed:.2e}, max={max_ed:.2e}, min={min_ed:.2e}")
            
            # Interpret results
            if model_name in ['unet', 'resnet']:
                if mean_ed < 1e-6:
                    print(f"         ✅ Supports hypothesis: {model_name.upper()} remains closed")
                else:
                    print(f"         ❓ Unexpected: {model_name.upper()} shows non-zero curl")
            elif model_name == 'els':
                if mean_ed > 1e-6:
                    print(f"         ✅ Supports hypothesis: ELS shows locality breakdown")
                else:
                    print(f"         ❓ Unexpected: ELS remains closed at 64×64")
        else:
            print(f"{model_name.upper():6s}: no data collected")
    
    print("\n✅ 64×64 Hypothesis test completed!")
    print(f"Check {output_dir} for detailed results and visualizations.")
    
    return results


if __name__ == '__main__':
    print("🧪 64×64 Exterior Derivative Analysis - Locality Breakdown Hypothesis")
    print("Testing theoretical prediction about ELS behavior at higher resolution")
    print()
    
    try:
        results = demo_64x64_analysis()
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Analysis finished!") 