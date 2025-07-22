#!/usr/bin/env python3
"""
Test script for the exterior derivative implementation.
"""

import numpy as np
import torch

from src.utils.exterior_derivative import (
    ExteriorDerivative,
    cnn_like_function,
    compute_exterior_derivative,
    compute_exterior_derivative_nd,
    exterior_derivative_magnitude,
    image_processing_function,
    is_exact,
    linear_function,
    rotation_function,
)


def test_conservative_field():
    """Test that a conservative field has zero exterior derivative."""
    print("Testing conservative field...")
    
    def conservative_field(x):
        """Conservative field: f(x,y) = (x, y)"""
        return x
    
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    exterior_deriv = compute_exterior_derivative(x, conservative_field)
    
    print(f"Exterior derivative:\n{exterior_deriv}")
    print(f"Is exact (should be True): {is_exact(exterior_deriv)}")
    
    # Should be exactly zero for conservative field
    assert torch.allclose(exterior_deriv, torch.zeros_like(exterior_deriv)), \
        "Conservative field should have zero exterior derivative"
    print("✓ Conservative field test passed\n")


def test_non_conservative_field():
    """Test that a non-conservative field has non-zero exterior derivative."""
    print("Testing non-conservative field...")
    
    def non_conservative_field(x):
        """Non-conservative field: f(x,y) = (-y, x)"""
        return torch.stack([-x[:, 1], x[:, 0]], dim=1)
    
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    exterior_deriv = compute_exterior_derivative(x, non_conservative_field)
    
    print(f"Exterior derivative:\n{exterior_deriv}")
    print(f"Is exact (should be False): {is_exact(exterior_deriv)}")
    print(f"Magnitude: {exterior_derivative_magnitude(exterior_deriv)}")
    
    # Should be non-zero for non-conservative field
    assert not torch.allclose(exterior_deriv, torch.zeros_like(exterior_deriv)), \
        "Non-conservative field should have non-zero exterior derivative"
    print("✓ Non-conservative field test passed\n")


def test_linear_function():
    """Test with the provided linear function."""
    print("Testing linear function...")
    
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    exterior_deriv = compute_exterior_derivative(x, linear_function)
    
    print(f"Exterior derivative:\n{exterior_deriv}")
    print(f"Is exact: {is_exact(exterior_deriv)}")
    print(f"Magnitude: {exterior_derivative_magnitude(exterior_deriv)}")
    print("✓ Linear function test passed\n")


def test_rotation_function():
    """Test with the provided rotation function."""
    print("Testing rotation function...")
    
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    exterior_deriv = compute_exterior_derivative(x, rotation_function)
    
    print(f"Exterior derivative:\n{exterior_deriv}")
    print(f"Is exact: {is_exact(exterior_deriv)}")
    print(f"Magnitude: {exterior_derivative_magnitude(exterior_deriv)}")
    print("✓ Rotation function test passed\n")


def test_3d_field():
    """Test with a 3D vector field."""
    print("Testing 3D field...")
    
    def curl_field_3d(x):
        """3D field with curl: f(x,y,z) = (-y, x, 0)"""
        return torch.stack([-x[:, 1], x[:, 0], torch.zeros_like(x[:, 0])], dim=1)
    
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    exterior_deriv = compute_exterior_derivative(x, curl_field_3d)
    
    print(f"Input shape: {x.shape}")
    print(f"Exterior derivative shape: {exterior_deriv.shape}")
    print(f"Exterior derivative:\n{exterior_deriv}")
    print(f"Magnitude: {exterior_derivative_magnitude(exterior_deriv)}")
    print("✓ 3D field test passed\n")


def test_batch_processing():
    """Test batch processing capabilities."""
    print("Testing batch processing...")
    
    def simple_field(x):
        return x**2
    
    # Test with different batch sizes
    batch_sizes = [1, 5, 10]
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 2, requires_grad=True)
        exterior_deriv = compute_exterior_derivative(x, simple_field)
        
        print(f"Batch size {batch_size}: input shape {x.shape}, output shape {exterior_deriv.shape}")
        
        # Check shapes are correct
        assert exterior_deriv.shape == (batch_size, 2, 2), \
            f"Expected shape ({batch_size}, 2, 2), got {exterior_deriv.shape}"
    
    print("✓ Batch processing test passed\n")


def test_batch_independence():
    """Test that batch elements are processed independently."""
    print("Testing batch independence...")
    
    def complex_field(x):
        """A non-conservative field that varies with input values"""
        return torch.stack([
            -x[:, 1] * (x[:, 0]**2 + x[:, 1]**2),  # f₁(x,y) = -y(x² + y²)
            x[:, 0] * (x[:, 0]**2 + x[:, 1]**2)    # f₂(x,y) = x(x² + y²)
        ], dim=1)
    
    # Create a batch with very different inputs
    x = torch.tensor([
        [1.0, 2.0],   # First batch element
        [10.0, 0.1],  # Second batch element - very different
        [-5.0, 3.0]   # Third batch element - different again
    ], requires_grad=True)
    
    exterior_deriv = compute_exterior_derivative(x, complex_field)
    
    print(f"Input:\n{x}")
    print(f"Exterior derivative shape: {exterior_deriv.shape}")
    print(f"Exterior derivative:\n{exterior_deriv}")
    
    # Check that each batch element has different exterior derivatives
    # (since inputs are very different)
    assert not torch.allclose(exterior_deriv[0], exterior_deriv[1], atol=1e-6), \
        "Batch elements should have different exterior derivatives for different inputs"
    assert not torch.allclose(exterior_deriv[0], exterior_deriv[2], atol=1e-6), \
        "Batch elements should have different exterior derivatives for different inputs"
    
    # Check antisymmetry for each batch element
    for i in range(x.shape[0]):
        batch_exterior = exterior_deriv[i]
        antisymmetric = batch_exterior + batch_exterior.T
        assert torch.allclose(antisymmetric, torch.zeros_like(antisymmetric), atol=1e-6), \
            f"Exterior derivative for batch element {i} should be antisymmetric"
    
    print("✓ Batch independence test passed\n")


def test_2d_tensors():
    """Test exterior derivative with 2D tensors (like images)."""
    print("Testing 2D tensors...")
    
    # Create a 2D tensor (batch_size, channels, height, width)
    x = torch.randn(2, 3, 4, 4, requires_grad=True)  # 2 batches, 3 channels, 4x4 images
    
    exterior_deriv = compute_exterior_derivative_nd(x, cnn_like_function)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {exterior_deriv.shape}")
    print(f"Expected output shape: {x.shape + x.shape[1:]}")
    
    # Check that output shape is correct
    expected_shape = (x.shape[0],) + x.shape[1:] + x.shape[1:]
    assert exterior_deriv.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {exterior_deriv.shape}"
    
    # Check antisymmetry for a few batch elements
    for b in range(min(2, x.shape[0])):
        N = x.shape[1] * x.shape[2] * x.shape[3]
        point_exterior = exterior_deriv[b].reshape(N, N)
        antisymmetric = point_exterior + point_exterior.T
        assert torch.allclose(antisymmetric, torch.zeros_like(antisymmetric), atol=1e-6), \
            f"Exterior derivative for batch element {b} should be antisymmetric"
    
    print("✓ 2D tensors test passed\n")


def test_3d_tensors():
    """Test exterior derivative with 3D tensors."""
    print("Testing 3D tensors...")
    
    # Create a 3D tensor (batch_size, depth, height, width)
    x = torch.randn(2, 2, 3, 3, requires_grad=True)  # 2 batches, 2 depth, 3x3 spatial
    
    exterior_deriv = compute_exterior_derivative_nd(x, image_processing_function)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {exterior_deriv.shape}")
    
    # Check that output shape is correct
    expected_shape = (x.shape[0],) + x.shape[1:] + x.shape[1:]
    assert exterior_deriv.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {exterior_deriv.shape}"
    
    print("✓ 3D tensors test passed\n")


def test_cnn_compatibility():
    """Test that the exterior derivative works with CNN-like operations."""
    print("Testing CNN compatibility...")
    
    # Create a simple CNN-like function using PyTorch layers
    class SimpleCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
            self.activation = torch.nn.ReLU()
        
        def forward(self, x):
            return self.activation(self.conv(x))
    
    cnn = SimpleCNN()
    
    # Create input tensor
    x = torch.randn(2, 3, 4, 4, requires_grad=True)
    
    # Compute exterior derivative
    exterior_deriv = compute_exterior_derivative_nd(x, cnn)
    
    print(f"CNN input shape: {x.shape}")
    print(f"CNN output shape: {exterior_deriv.shape}")
    
    # Check that we can compute gradients through the exterior derivative
    # Reshape to 2D for magnitude computation
    batch_size = exterior_deriv.shape[0]
    total_dims = exterior_deriv.shape[1] * exterior_deriv.shape[2] * exterior_deriv.shape[3]
    exterior_deriv_2d = exterior_deriv.view(batch_size, total_dims, total_dims)
    loss = torch.mean(exterior_derivative_magnitude(exterior_deriv_2d)**2)
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Input has gradients: {x.grad is not None}")
    print("✓ CNN compatibility test passed\n")


def test_curl_magnitude():
    """Test that a field with non-zero curl has a non-zero curl magnitude."""
    print("Testing curl magnitude...")
    
    def vortex_field(x):
        """Vortex field: f(x,y) = (-y, x) - has constant curl = 2"""
        return torch.stack([-x[:, 1], x[:, 0]], dim=1)
    
    def uniform_field(x):
        """Uniform field: f(x,y) = (1, 0) - has zero curl"""
        batch_size = x.shape[0]
        return torch.stack([
            torch.ones(batch_size, device=x.device),
            torch.zeros(batch_size, device=x.device)
        ], dim=1)
    
    def shear_field(x):
        """Shear field: f(x,y) = (y, 0) - has curl = -1"""
        return torch.stack([x[:, 1], torch.zeros_like(x[:, 0])], dim=1)
    
    def varying_field(x):
        """Spatially varying field: f(x,y) = (x*y, x²) - curl varies with position"""
        return torch.stack([x[:, 0] * x[:, 1], x[:, 0]**2], dim=1)
    
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.5, -1.5]], requires_grad=True)
    
    # Test vortex field (should have non-zero curl)
    exterior_deriv_vortex = compute_exterior_derivative(x, vortex_field)
    curl_magnitude_vortex = exterior_derivative_magnitude(exterior_deriv_vortex)
    
    print(f"Vortex field exterior derivative:\n{exterior_deriv_vortex}")
    print(f"Vortex field curl magnitude: {curl_magnitude_vortex}")
    
    # Verify magnitude calculation manually for first point
    first_point_matrix = exterior_deriv_vortex[0]  # 2x2 matrix for first point
    manual_magnitude = torch.sqrt(torch.sum(first_point_matrix**2))
    print(f"Manual magnitude calculation for first point: {manual_magnitude}")
    print(f"Should match first magnitude entry: {curl_magnitude_vortex[0]}")
    
    # Test uniform field (should have zero curl)
    exterior_deriv_uniform = compute_exterior_derivative(x, uniform_field)
    curl_magnitude_uniform = exterior_derivative_magnitude(exterior_deriv_uniform)
    
    print(f"Uniform field curl magnitude: {curl_magnitude_uniform}")
    
    # Test shear field (should have non-zero curl)
    exterior_deriv_shear = compute_exterior_derivative(x, shear_field)
    curl_magnitude_shear = exterior_derivative_magnitude(exterior_deriv_shear)
    
    print(f"Shear field curl magnitude: {curl_magnitude_shear}")
    
    # Test spatially varying field (should have different magnitudes at different points)
    exterior_deriv_varying = compute_exterior_derivative(x, varying_field)
    curl_magnitude_varying = exterior_derivative_magnitude(exterior_deriv_varying)
    
    print(f"Varying field curl magnitude: {curl_magnitude_varying}")
    print("Note: Different values show curl varies with position!")
    
    # Assertions
    assert torch.all(curl_magnitude_vortex > 1e-6), \
        f"Vortex field should have non-zero curl magnitude, got {curl_magnitude_vortex}"
    
    assert torch.all(curl_magnitude_uniform < 1e-6), \
        f"Uniform field should have zero curl magnitude, got {curl_magnitude_uniform}"
    
    assert torch.all(curl_magnitude_shear > 1e-6), \
        f"Shear field should have non-zero curl magnitude, got {curl_magnitude_shear}"
    
    # Check that vortex field has expected magnitude (curl = 2 for vortex field)
    # For 2D: curl = ∂f₂/∂x - ∂f₁/∂y = ∂x/∂x - ∂(-y)/∂y = 1 - (-1) = 2
    # But exterior derivative gives antisymmetric matrix, so magnitude is √2 * |curl|
    expected_vortex_magnitude = 2.0 * np.sqrt(2)
    expected_tensor = torch.tensor(expected_vortex_magnitude, dtype=curl_magnitude_vortex.dtype, device=curl_magnitude_vortex.device)
    assert torch.allclose(curl_magnitude_vortex, expected_tensor, atol=1e-5), \
        f"Vortex field should have magnitude {expected_vortex_magnitude}, got {curl_magnitude_vortex}"
    
    # Check that varying field has different magnitudes at different points
    assert not torch.allclose(curl_magnitude_varying[0], curl_magnitude_varying[1], atol=1e-6), \
        f"Varying field should have different magnitudes at different points, got {curl_magnitude_varying}"
    
    print("✓ Curl magnitude test passed\n")


def test_gradient_flow():
    """Test that the exterior derivative can be used in gradient-based optimization."""
    print("Testing gradient flow...")
    
    # Create a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleNN()
    x = torch.randn(5, 2, requires_grad=True)
    
    # Compute exterior derivative
    exterior_deriv = compute_exterior_derivative(x, model)
    
    # Compute loss based on exterior derivative
    loss = torch.mean(exterior_derivative_magnitude(exterior_deriv)**2)
    
    # Backward pass
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Model parameters have gradients: {all(p.grad is not None for p in model.parameters())}")
    print("✓ Gradient flow test passed\n")


def main():
    """Run all tests."""
    print("Running Exterior Derivative Tests")
    print("=" * 50)
    
    try:
        test_conservative_field()
        test_non_conservative_field()
        test_linear_function()
        test_rotation_function()
        test_3d_field()
        test_batch_processing()
        test_batch_independence()
        test_2d_tensors()
        test_3d_tensors()
        test_cnn_compatibility()
        test_curl_magnitude()
        test_gradient_flow()
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 