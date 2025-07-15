from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian


class ExteriorDerivative(nn.Module):
    """
    PyTorch transformation that computes the exterior derivative of a vector-valued function.
    
    For a function f: R^n -> R^n, the exterior derivative df is a 2-form (antisymmetric matrix)
    with components (df)_ij = ∂_i f_j - ∂_j f_i.
    
    This is computed using automatic differentiation to get the Jacobian matrix
    and then extracting the antisymmetric part.
    
    Supports both 1D vectors (batch_size, n) and higher-dimensional tensors
    (batch_size, channels, height, width, ...) by flattening and reshaping.
    """
    
    def __init__(self, function: Optional[Callable] = None):
        """
        Initialize the exterior derivative transformation.
        
        Args:
            function: Optional function to compute exterior derivative of.
                     If None, will be set during forward pass.
        """
        super().__init__()
        self.function = function
    
    def forward(self, x: torch.Tensor, function: Optional[Callable] = None) -> torch.Tensor:
        """
        Compute the exterior derivative of the function at point x.
        
        Args:
            x: Input tensor of shape (batch_size, n) where n is the dimension
            function: Function to differentiate. If None, uses self.function.
                     Should take x and return tensor of same shape as x.
        
        Returns:
            Tensor of shape (batch_size, n, n) containing the exterior derivative
            (antisymmetric matrix) at each point.
        """
        if function is None:
            function = self.function
        if function is None:
            raise ValueError("No function provided for exterior derivative computation")
        
        # Ensure x requires gradients
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        
        # Compute function values
        y = function(x)
        
        # Check that input and output dimensions match
        if x.shape != y.shape:
            raise ValueError(f"Function input shape {x.shape} must match output shape {y.shape}")
        
        batch_size, n = x.shape
        
        # Initialize exterior derivative tensor
        exterior_deriv = torch.zeros(batch_size, n, n, device=x.device, dtype=x.dtype)
        
        # Compute Jacobian matrix using PyTorch's jacobian function
        # For each batch element, compute the Jacobian
        jacobian_matrices = []
        for b in range(batch_size):
            # Define function for single batch element
            def single_func(x_single):
                return function(x_single.unsqueeze(0)).squeeze(0)
            
            # Compute Jacobian for this batch element
            jac = jacobian(single_func, x[b], create_graph=True)
            jacobian_matrices.append(jac)
        
        # Stack Jacobians and compute exterior derivative
        jacobian_stack = torch.stack(jacobian_matrices, dim=0)  # Shape: (batch_size, n, n)
        
        # Extract antisymmetric part: (df)_ij = ∂_i f_j - ∂_j f_i
        exterior_deriv = jacobian_stack - jacobian_stack.transpose(-2, -1)
        
        return exterior_deriv

    def forward_nd(self, x: torch.Tensor, function: Optional[Callable] = None) -> torch.Tensor:
        """
        Compute the exterior derivative for higher-dimensional tensors.
        
        Args:
            x: Input tensor of shape (batch_size, *dims) where dims can be any shape
            function: Function to differentiate. If None, uses self.function.
                     Should take x and return tensor of same shape as x.
        
        Returns:
            Tensor of shape (batch_size, *dims, *dims) containing the exterior derivative
            at each point, where the last two dimensions form antisymmetric matrices.
        """
        if function is None:
            function = self.function
        if function is None:
            raise ValueError("No function provided for exterior derivative computation")
        
        # Store original shape
        original_shape = x.shape
        batch_size = original_shape[0]
        
        # Flatten all dimensions except batch
        x_flat = x.view(batch_size, -1)  # Shape: (batch_size, total_elements)
        
        # Define a wrapper function that handles flattening/reshaping
        def flat_function(x_flat):
            # Reshape back to original shape (without batch dimension)
            x_reshaped = x_flat.view(original_shape[1:])
            # Apply the original function
            y = function(x_reshaped.unsqueeze(0))  # Add batch dimension back
            # Flatten the output (remove batch dimension first)
            return y.squeeze(0).view(-1)
        
        # Compute exterior derivative on flattened tensors using jacobian
        # For each batch element, compute the Jacobian
        jacobian_matrices = []
        for b in range(batch_size):
            # Define function for single batch element
            def single_func(x_single):
                return flat_function(x_single.unsqueeze(0)).squeeze(0)
            
            # Compute Jacobian for this batch element
            jac = jacobian(single_func, x_flat[b], create_graph=True)
            jacobian_matrices.append(jac)
        
        # Stack Jacobians and compute exterior derivative
        jacobian_stack = torch.stack(jacobian_matrices, dim=0)  # Shape: (batch_size, total_elements, total_elements)
        
        # Extract antisymmetric part
        exterior_deriv_flat = jacobian_stack - jacobian_stack.transpose(-2, -1)
        
        # Reshape back to higher dimensions
        total_elements = x_flat.shape[1]
        exterior_deriv = exterior_deriv_flat.view(batch_size, *original_shape[1:], *original_shape[1:])
        
        return exterior_deriv


class ExteriorDerivativeTransform:
    """
    A transform class that can be used in PyTorch data pipelines.
    """
    
    def __init__(self, function: Callable):
        """
        Initialize the transform.
        
        Args:
            function: Function to compute exterior derivative of.
        """
        self.exterior_deriv = ExteriorDerivative(function)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the exterior derivative transformation.
        
        Args:
            x: Input tensor
            
        Returns:
            Exterior derivative tensor
        """
        return self.exterior_deriv(x)


def compute_exterior_derivative(x: torch.Tensor, function: Callable) -> torch.Tensor:
    """
    Convenience function to compute exterior derivative without creating a class instance.
    
    Args:
        x: Input tensor of shape (batch_size, n)
        function: Function to differentiate
        
    Returns:
        Exterior derivative tensor of shape (batch_size, n, n)
    """
    exterior_deriv = ExteriorDerivative()
    return exterior_deriv(x, function)


def compute_exterior_derivative_nd(x: torch.Tensor, function: Callable) -> torch.Tensor:
    """
    Convenience function to compute exterior derivative for higher-dimensional tensors.
    
    Args:
        x: Input tensor of shape (batch_size, *dims)
        function: Function to differentiate
        
    Returns:
        Exterior derivative tensor of shape (batch_size, *dims, *dims)
    """
    exterior_deriv = ExteriorDerivative()
    return exterior_deriv.forward_nd(x, function)


# Example functions for testing
def linear_function(x: torch.Tensor) -> torch.Tensor:
    """Example linear function f(x) = Ax + b"""
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=x.device, dtype=x.dtype)
    b = torch.tensor([0.5, 1.0], device=x.device, dtype=x.dtype)
    return torch.matmul(x, A.T) + b


def quadratic_function(x: torch.Tensor) -> torch.Tensor:
    """Example quadratic function f(x) = x^2 + x"""
    return x**2 + x


def rotation_function(x: torch.Tensor) -> torch.Tensor:
    """Example rotation function (should have non-zero exterior derivative)"""
    theta = torch.tensor(torch.pi / 4, device=x.device, dtype=x.dtype)
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                     [torch.sin(theta), torch.cos(theta)]], 
                    device=x.device, dtype=x.dtype)
    return torch.matmul(x, R.T)


def cnn_like_function(x: torch.Tensor) -> torch.Tensor:
    """Example function that mimics CNN operations on 2D tensors"""
    # Apply convolution-like operations
    # For simplicity, we'll do element-wise operations that preserve shape
    batch_size, channels, height, width = x.shape
    
    # Create a simple transformation that varies spatially
    y = x.clone()
    
    # Add spatial variation based on position
    for h in range(height):
        for w in range(width):
            spatial_factor = torch.sin(torch.tensor(h + w, device=x.device, dtype=x.dtype) * 0.1)
            y[:, :, h, w] = x[:, :, h, w] * (1 + spatial_factor * 0.1)
    
    return y


def image_processing_function(x: torch.Tensor) -> torch.Tensor:
    """Example function that processes image-like tensors"""
    # Apply some image processing operations
    y = x.clone()
    
    # Add some non-linear transformations
    y = y + 0.1 * torch.sin(y)
    y = y * torch.exp(-0.01 * y**2)  # Gaussian-like modulation
    
    return y


# Utility functions for analysis
def is_closed(exterior_deriv: torch.Tensor, tol: float = 1e-6) -> bool:
    """
    Check if the exterior derivative is closed (d²f = 0).
    For 1-forms in R^n, this is always true by Poincaré's lemma.
    
    Args:
        exterior_deriv: Exterior derivative tensor
        tol: Tolerance for numerical comparison
        
    Returns:
        True if closed, False otherwise
    """
    # For 1-forms in R^n, d²f = 0 is always satisfied
    # This is a placeholder for more general cases
    return True


def is_exact(exterior_deriv: torch.Tensor, tol: float = 1e-6) -> bool:
    """
    Check if the exterior derivative is exact (df = 0).
    
    Args:
        exterior_deriv: Exterior derivative tensor
        tol: Tolerance for numerical comparison
        
    Returns:
        True if exact, False otherwise
    """
    return torch.allclose(exterior_deriv, torch.zeros_like(exterior_deriv), atol=tol)


def exterior_derivative_magnitude(exterior_deriv: torch.Tensor) -> torch.Tensor:
    """
    Compute the magnitude of the exterior derivative.
    
    Args:
        exterior_deriv: Exterior derivative tensor
        
    Returns:
        Magnitude tensor
    """
    return torch.norm(exterior_deriv, dim=(-2, -1))


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Exterior Derivative Implementation")
    print("=" * 50)
    
    # Create test data
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    
    # Test with different functions
    functions = {
        "Linear": linear_function,
        "Quadratic": quadratic_function,
        "Rotation": rotation_function
    }
    
    for name, func in functions.items():
        print(f"\nTesting {name} function:")
        
        # Compute exterior derivative
        exterior_deriv = compute_exterior_derivative(x, func)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {exterior_deriv.shape}")
        print(f"Exterior derivative:\n{exterior_deriv}")
        print(f"Is exact: {is_exact(exterior_deriv)}")
        print(f"Magnitude: {exterior_derivative_magnitude(exterior_deriv)}") 