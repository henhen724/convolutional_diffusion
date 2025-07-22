#!/usr/bin/env python3
"""
Demo script for analyzing exterior derivative of score function models.

This script demonstrates how to use the exterior derivative analysis tools
with moderate ELS parameters for reasonable computation time.
Uses CIFAR-10 by default but can be easily changed to MNIST or FashionMNIST.
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
from src.utils.data import get_dataset
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.idealscore import LocalEquivBordersScoreModule, ScheduledScoreMachine
from src.utils.exterior_derivative import compute_exterior_derivative, exterior_derivative_magnitude


def create_simple_score_function(model, t_value, device, in_channels=1, image_size=32, is_conditional=False):
    """Create a simple score function wrapper for exterior derivative computation."""
    def score_fn(x_flat):
        # Get batch size from input
        batch_size = x_flat.shape[0]
        
        # Reshape flattened input back to image shape
        x_img = x_flat.view(batch_size, in_channels, image_size, image_size)
        
        # Get score prediction
        with torch.no_grad():
            if is_conditional:
                # Use class 0 for conditional models
                label = torch.zeros(batch_size, dtype=torch.long, device=device)
                score = model.backbone(t_value, x_img, label=label)
            else:
                score = model.backbone(t_value, x_img)
        
        # Return flattened score but keep batch dimension
        return score.view(batch_size, -1)
    
    return score_fn


def demo_single_step_analysis():
    """Demonstrate exterior derivative analysis for a single diffusion step."""
    
    print("🚀 Starting Exterior Derivative Demo")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset (modify this line to change dataset)
    dataset_name = 'cifar10'  # Change this to 'mnist', 'fashionmnist', etc.
    print(f"\n📁 Loading {dataset_name.upper()} dataset...")
    dataset, metadata = get_dataset(dataset_name, root='./data')
    in_channels = metadata['num_channels']
    image_size = metadata['image_size']
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Image size: {image_size}x{image_size}, Channels: {in_channels}")
    
    # Load trained models from checkpoints
    print("\n🧠 Loading trained models...")
    
    # Determine conditional vs unconditional based on available checkpoints
    dataset_name_upper = metadata['name'].upper()
    
    # Look for CIFAR-10 trained models (they are conditional)
    unet_path = f'./checkpoints/backbone_{dataset_name_upper}_UNet_zeros_conditional.pt'
    resnet_path = f'./checkpoints/backbone_{dataset_name_upper}_ResNet_zeros_conditional.pt'
    
    # Check if conditional models exist, otherwise try unconditional
    is_conditional = os.path.exists(unet_path) or os.path.exists(resnet_path)
    if not is_conditional:
        unet_path = f'./checkpoints/backbone_{dataset_name_upper}_UNet_zeros.pt'
        resnet_path = f'./checkpoints/backbone_{dataset_name_upper}_ResNet_zeros.pt'
    
    print(f"Using conditional models: {is_conditional}")
    
    # Load UNet model
    if os.path.exists(unet_path):
        print(f"📂 Loading UNet from {unet_path}")
        unet_model = torch.load(unet_path, map_location=device, weights_only=False)
        unet_model.to(device)
        unet_model.eval()
        print("✅ UNet loaded successfully")
    else:
        print(f"⚠️  UNet checkpoint not found at {unet_path}, creating untrained model")
        unet = MinimalUNet(channels=in_channels, conditional=is_conditional, emb_dim=128)
        unet_model = DDIM(backbone=unet, in_channels=in_channels, default_imsize=image_size)
        unet_model.to(device)
        unet_model.eval()
    
    # Load ResNet model  
    if os.path.exists(resnet_path):
        print(f"📂 Loading ResNet from {resnet_path}")
        resnet_model = torch.load(resnet_path, map_location=device, weights_only=False)
        resnet_model.to(device)
        resnet_model.eval()
        print("✅ ResNet loaded successfully")
    else:
        print(f"⚠️  ResNet checkpoint not found at {resnet_path}, creating untrained model")
        resnet = MinimalResNet(channels=in_channels, conditional=is_conditional, emb_dim=128)
        resnet_model = DDIM(backbone=resnet, in_channels=in_channels, default_imsize=image_size)
        resnet_model.to(device)
        resnet_model.eval()
    
    # Generate test noise
    print("\n🎲 Generating test noise...")
    torch.manual_seed(42)  # For reproducibility
    x_test = torch.randn(1, in_channels, image_size, image_size, device=device)
    print(f"Test noise shape: {x_test.shape}")
    
    # Create ELS model for comparison (using parameters from els_script.py)
    print("\n🔬 Creating ELS model...")
    els_model = None
    try:
        # Use moderate sample size for demo (training condition mismatch likely causes quality issues)
        max_samples = min(1000, len(dataset))  # Reasonable sample size
        scorebatchsize = 64  # Moderate batch size for speed
        
        els_backbone = LocalEquivBordersScoreModule(
            dataset,  # Use full dataset, not subset
            kernel_size=3,
            batch_size=scorebatchsize,
            image_size=image_size,
            channels=in_channels,
            schedule=cosine_noise_schedule,
            max_samples=max_samples,
            shuffle=False  # Explicitly set shuffle=False like els_script
        )
        
        # Try to load scales file like els_script.py does
        scales = None
        dataset_name_upper = metadata['name'].upper()
        possible_scales_files = [
            f'./checkpoints/scales_{dataset_name_upper}_ResNet_zeros_conditional.pt',
            f'./checkpoints/scales_{dataset_name_upper}_ResNet_zeros.pt',
            f'./checkpoints/scales_{dataset_name_upper}_UNet_zeros_conditional.pt',
            f'./checkpoints/scales_{dataset_name_upper}_UNet_zeros.pt'
        ]
        
        for scales_file in possible_scales_files:
            if os.path.exists(scales_file):
                try:
                    scales_data = torch.load(scales_file, map_location=device, weights_only=False)
                    # Handle both tensor and list formats
                    if isinstance(scales_data, torch.Tensor):
                        scales = list(scales_data.int().numpy())
                        scales = [int(s) for s in scales]
                    elif isinstance(scales_data, list):
                        scales = [int(s) for s in scales_data]
                    else:
                        scales = scales_data
                    print(f"✅ Loaded scales from {scales_file}")
                    break
                except Exception as e:
                    print(f"⚠️  Failed to load scales from {scales_file}: {e}")
        
        if scales is None:
            print("ℹ️  No scales file found, using default scaling")
        
        els_model = ScheduledScoreMachine(
            els_backbone,
            in_channels=in_channels,
            imsize=image_size,
            noise_schedule=cosine_noise_schedule,
            score_backbone=True,
            scales=scales  # Use loaded scales if available
        )
        els_model.to(device)
        print("✅ ELS model created successfully")
        print(f"   Using {max_samples} samples with batch size {scorebatchsize}")
        
    except Exception as e:
        print(f"⚠️  ELS model creation failed: {e}")
        print("   (Continuing without ELS - may require training-matched conditions)")

    # Test at different diffusion times
    timesteps_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    results = {
        'unet': {'timesteps': [], 'ed_magnitudes': []},
        'resnet': {'timesteps': [], 'ed_magnitudes': []},
        'els': {'timesteps': [], 'ed_magnitudes': []}
    }
    
    print("\n🔍 Computing exterior derivatives...")
    
    for t_val in timesteps_to_test:
        print(f"\nAnalyzing at t = {t_val:.1f}")
        
        t = torch.tensor([t_val], device=device)
        
        # Test UNet
        try:
            score_fn_unet = create_simple_score_function(unet_model, t, device, in_channels, image_size, is_conditional)
            x_flat = x_test.view(1, -1)
            
            # Compute exterior derivative
            exterior_deriv = compute_exterior_derivative(x_flat, score_fn_unet)
            ed_magnitude = exterior_derivative_magnitude(exterior_deriv)
            
            results['unet']['timesteps'].append(t_val)
            results['unet']['ed_magnitudes'].append(ed_magnitude.item())
            
            print(f"  UNet ED magnitude: {ed_magnitude.item():.6f}")
            
        except Exception as e:
            print(f"  UNet failed: {e}")
        
        # Test ResNet
        try:
            score_fn_resnet = create_simple_score_function(resnet_model, t, device, in_channels, image_size, is_conditional)
            x_flat = x_test.view(1, -1)
            
            # Compute exterior derivative
            exterior_deriv = compute_exterior_derivative(x_flat, score_fn_resnet)
            ed_magnitude = exterior_derivative_magnitude(exterior_deriv)
            
            results['resnet']['timesteps'].append(t_val)
            results['resnet']['ed_magnitudes'].append(ed_magnitude.item())
            
            print(f"  ResNet ED magnitude: {ed_magnitude.item():.6f}")
            
        except Exception as e:
            print(f"  ResNet failed: {e}")
        
        # Test ELS
        if els_model is not None:
            try:
                # ELS uses a different interface
                def score_fn_els(x_flat):
                    batch_size = x_flat.shape[0]
                    x_img = x_flat.view(batch_size, in_channels, image_size, image_size)
                    
                    with torch.no_grad():
                        score = els_model.backbone(t, x_img, device=device)
                    
                    return score.view(batch_size, -1)
                
                x_flat = x_test.view(1, -1)
                
                # Compute exterior derivative
                exterior_deriv = compute_exterior_derivative(x_flat, score_fn_els)
                ed_magnitude = exterior_derivative_magnitude(exterior_deriv)
                
                results['els']['timesteps'].append(t_val)
                results['els']['ed_magnitudes'].append(ed_magnitude.item())
                
                print(f"  ELS ED magnitude: {ed_magnitude.item():.6f}")
                
            except Exception as e:
                print(f"  ELS failed: {e}")
        else:
            print(f"  ELS skipped (not available)")
    
    # Generate final images using each model
    print("\n🖼️  Generating final images...")
    generated_images = {}
    
    # Prepare class labels for conditional models
    label = None
    if is_conditional:
        # Use a fixed class for reproducible results (e.g., class 0)
        label = torch.zeros(1, dtype=torch.long, device=device)
        print(f"Using class label: {label.item()} for conditional generation")
    
    # Generate with UNet (using same nsteps as els_script.py)
    try:
        with torch.no_grad():
            unet_img = unet_model.sample(x=x_test.clone(), nsteps=20, label=label).clamp(-1, 1)
            generated_images['unet'] = unet_img.squeeze().cpu().numpy()
        print("✅ UNet image generated")
    except Exception as e:
        print(f"❌ UNet generation failed: {e}")
        generated_images['unet'] = None
    
    # Generate with ResNet
    try:
        with torch.no_grad():
            resnet_img = resnet_model.sample(x=x_test.clone(), nsteps=20, label=label).clamp(-1, 1)
            generated_images['resnet'] = resnet_img.squeeze().cpu().numpy()
        print("✅ ResNet image generated")
    except Exception as e:
        print(f"❌ ResNet generation failed: {e}")
        generated_images['resnet'] = None
    
    # Generate with ELS
    if els_model is not None:
        try:
            with torch.no_grad():
                els_img = els_model(x_test.clone(), nsteps=20, device=device).clamp(-1, 1)
                generated_images['els'] = els_img.squeeze().cpu().numpy()
            print("✅ ELS image generated")
        except Exception as e:
            print(f"❌ ELS generation failed: {e}")
            generated_images['els'] = None
    else:
        generated_images['els'] = None

    # Create visualization
    print("\n📊 Creating visualization...")
    
    # Create figure with proper subplot grid
    fig = plt.figure(figsize=(15, 10))
    
    # Plot exterior derivative magnitudes (top row, spanning two columns)
    ax_main = plt.subplot2grid((2, 3), (0, 0), colspan=2, fig=fig)
    for model_name, data in results.items():
        if data['timesteps']:
            ax_main.plot(data['timesteps'], data['ed_magnitudes'], 
                        marker='o', linewidth=2, label=model_name.upper(), markersize=6)
    
    ax_main.set_xlabel('Diffusion Time (t)')
    ax_main.set_ylabel('Exterior Derivative Magnitude')
    ax_main.set_title('Exterior Derivative vs Diffusion Time')
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    # Show initial noise (top right)
    ax_noise = plt.subplot2grid((2, 3), (0, 2), fig=fig)
    test_img = x_test.squeeze().cpu().numpy()
    
    # Handle image format for display
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
    
    display_noise = prepare_image_for_display(test_img.copy())
    
    if display_noise.ndim == 2:  # Grayscale
        ax_noise.imshow(display_noise, cmap='gray')
    else:  # RGB
        ax_noise.imshow(display_noise)
    
    ax_noise.set_title('Initial Noise')
    ax_noise.axis('off')
    
    # Show generated images (bottom row)
    model_names = ['unet', 'resnet', 'els']
    model_titles = ['UNet Generated', 'ResNet Generated', 'ELS Generated']
    
    for i, (model_name, title) in enumerate(zip(model_names, model_titles)):
        ax = plt.subplot2grid((2, 3), (1, i), fig=fig)
        
        if generated_images[model_name] is not None:
            img = generated_images[model_name]
            
            # Use the same image preparation function
            display_img = prepare_image_for_display(img.copy())
            
            if display_img.ndim == 2:  # Grayscale
                ax.imshow(display_img, cmap='gray')
            else:  # RGB
                ax.imshow(display_img)
        else:
            # Show placeholder if generation failed
            ax.text(0.5, 0.5, 'Generation\nFailed', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path('./results/exterior_derivative_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'comprehensive_demo_results.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Comprehensive results saved to {output_dir}")
    
    # Save numerical results
    torch.save(results, output_dir / 'demo_data.pt')
    
    # Print summary
    print("\n📈 Summary:")
    for model_name, data in results.items():
        if data['ed_magnitudes']:
            mean_ed = np.mean(data['ed_magnitudes'])
            std_ed = np.std(data['ed_magnitudes'])
            print(f"  {model_name.upper():6s}: mean ED = {mean_ed:.6f} ± {std_ed:.6f}")
        else:
            print(f"  {model_name.upper():6s}: no data collected")
    
    print("\n✅ Demo completed successfully!")
    print(f"Check {output_dir} for saved results and plots.")
    print("\n🖼️  The visualization shows:")
    print("  • Top row: Exterior derivative analysis + initial noise")
    print("  • Bottom row: Generated images from UNet, ResNet, and ELS")
    
    return results



if __name__ == '__main__':
    print("🧪 Exterior Derivative Analysis Demo")
    print("This demo compares UNet, ResNet, and ELS score functions")
    print("analyzing their exterior derivative behavior across diffusion timesteps")
    print("Note: Using moderate ELS parameters for reasonable computation time")
    print()
    
    try:
        # Run main demo
        results = demo_single_step_analysis()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n�� Demo finished!") 