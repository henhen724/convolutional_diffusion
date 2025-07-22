#!/usr/bin/env python3
"""
Quick test script to verify the exterior derivative shape fix.
"""

import torch
import numpy as np
from src.models import DDIM, MinimalUNet
from src.utils.data import get_dataset
from src.utils.exterior_derivative import compute_exterior_derivative, exterior_derivative_magnitude


def test_shape_fix():
    print("🧪 Testing Exterior Derivative Shape Fix")
    print("=" * 40)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load MNIST for testing
    dataset, metadata = get_dataset('mnist', root='./data')
    in_channels = metadata['num_channels']
    image_size = metadata['image_size']
    
    print(f"Dataset: MNIST")
    print(f"Image size: {image_size}, Channels: {in_channels}")
    
    # Create simple UNet model
    unet = MinimalUNet(channels=in_channels, conditional=False, emb_dim=64)
    model = DDIM(backbone=unet, in_channels=in_channels, default_imsize=image_size)
    model.to(device)
    model.eval()
    
    print("✅ Model created")
    
    # Create test input
    torch.manual_seed(42)
    x_test = torch.randn(1, in_channels, image_size, image_size, device=device)
    t_test = torch.tensor([0.5], device=device)
    
    print(f"Test input shape: {x_test.shape}")
    
    # Create FIXED score function wrapper
    def score_function_fixed(x_flat):
        batch_size = x_flat.shape[0]  # Get batch size from input
        x_img = x_flat.view(batch_size, in_channels, image_size, image_size)
        
        with torch.no_grad():
            score = model.backbone(t_test, x_img)
        
        return score.view(batch_size, -1)  # Keep batch dimension!
    
    # Test the fixed function
    print("\n🔍 Testing fixed score function...")
    
    x_flat = x_test.view(1, -1)
    print(f"Flattened input shape: {x_flat.shape}")
    
    # Test score function output
    score_output = score_function_fixed(x_flat)
    print(f"Score output shape: {score_output.shape}")
    
    # Check shapes match
    if x_flat.shape == score_output.shape:
        print("✅ Input and output shapes match!")
    else:
        print(f"❌ Shape mismatch: input {x_flat.shape} vs output {score_output.shape}")
        return False
    
    # Test exterior derivative computation
    try:
        print("\n🔄 Computing exterior derivative...")
        exterior_deriv = compute_exterior_derivative(x_flat, score_function_fixed)
        ed_magnitude = exterior_derivative_magnitude(exterior_deriv)
        
        print(f"✅ Exterior derivative computed successfully!")
        print(f"ED shape: {exterior_deriv.shape}")
        print(f"ED magnitude: {ed_magnitude.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Exterior derivative computation failed: {e}")
        return False


def test_broken_function():
    """Test the old broken version to show the difference."""
    print("\n🚫 Testing BROKEN score function (for comparison)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, metadata = get_dataset('mnist', root='./data')
    in_channels = metadata['num_channels']
    image_size = metadata['image_size']
    
    unet = MinimalUNet(channels=in_channels, conditional=False, emb_dim=64)
    model = DDIM(backbone=unet, in_channels=in_channels, default_imsize=image_size)
    model.to(device)
    model.eval()
    
    x_test = torch.randn(1, in_channels, image_size, image_size, device=device)
    t_test = torch.tensor([0.5], device=device)
    
    # Create BROKEN score function wrapper (old version)
    def score_function_broken(x_flat):
        batch_size = 1  # Hardcoded - WRONG!
        channels = 1    # Hardcoded - WRONG!
        img_size = int(np.sqrt(x_flat.shape[-1] // channels))
        
        x_img = x_flat.view(batch_size, channels, img_size, img_size)
        
        with torch.no_grad():
            score = model.backbone(t_test, x_img)
        
        return score.flatten()  # Removes batch dimension - WRONG!
    
    x_flat = x_test.view(1, -1)
    print(f"Flattened input shape: {x_flat.shape}")
    
    score_output = score_function_broken(x_flat)
    print(f"Broken score output shape: {score_output.shape}")
    
    # Try exterior derivative (should fail)
    try:
        exterior_deriv = compute_exterior_derivative(x_flat, score_function_broken)
        print("❌ This shouldn't have worked!")
    except Exception as e:
        print(f"✅ Expected error: {e}")


if __name__ == '__main__':
    print("Testing exterior derivative shape handling...")
    print()
    
    success = test_shape_fix()
    test_broken_function()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Shape fix is working correctly!")
        print("You can now run the analysis scripts without shape errors.")
    else:
        print("❌ Shape fix needs more work.")
    
    print("\nTo run the actual analysis:")
    print("python examples/exterior_derivative_demo.py")
    print("python analyze_exterior_derivative_simple.py --dataset mnist") 