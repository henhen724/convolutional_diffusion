# Exterior Derivative Transformation for PyTorch

This module provides a PyTorch transformation that computes the exterior derivative of vector-valued functions. The exterior derivative is a fundamental concept in differential geometry that generalizes the gradient, curl, and divergence operators.

## Mathematical Background

For a vector-valued function $f: \mathbb{R}^n \to \mathbb{R}^n$, the exterior derivative $df$ is a 2-form (antisymmetric matrix) with components:

$$(df)_{ij} = \frac{\partial f_j}{\partial x_i} - \frac{\partial f_i}{\partial x_j}$$

This measures the "rotation" or "non-conservative" nature of the vector field:
- If $df = 0$, the field is **conservative** (can be written as the gradient of a scalar potential)
- If $df \neq 0$, the field is **non-conservative** (has rotational components)

## Installation

Make sure you have PyTorch installed:
```bash
pip install torch
```

## Basic Usage

### Simple Function Call

```python
import torch
from exterior_derivative import compute_exterior_derivative

# Define a vector-valued function
def my_function(x):
    return torch.stack([x[:, 0] + x[:, 1], x[:, 1] - x[:, 0]], dim=1)

# Create input data
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# Compute exterior derivative
exterior_deriv = compute_exterior_derivative(x, my_function)
print(f"Exterior derivative shape: {exterior_deriv.shape}")
print(f"Exterior derivative:\n{exterior_deriv}")
```

### Using the Transform Class

```python
from exterior_derivative import ExteriorDerivativeTransform

# Create a transform
transform = ExteriorDerivativeTransform(my_function)

# Apply to data
result = transform(x)
```

### Using with Neural Networks

```python
import torch.nn as nn
from exterior_derivative import ExteriorDerivative

class VectorFieldNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# Create model and compute exterior derivative
model = VectorFieldNN()
x = torch.randn(10, 2, requires_grad=True)
exterior_deriv = compute_exterior_derivative(x, model)
```

## Examples

### Conservative Field (Zero Exterior Derivative)

```python
def conservative_field(x):
    """Conservative field: f(x,y) = (x, y)"""
    return x

x = torch.tensor([[1.0, 2.0]], requires_grad=True)
exterior_deriv = compute_exterior_derivative(x, conservative_field)
print(f"Exterior derivative:\n{exterior_deriv}")  # Should be zero
```

### Non-Conservative Field (Non-Zero Exterior Derivative)

```python
def non_conservative_field(x):
    """Non-conservative field: f(x,y) = (-y, x)"""
    return torch.stack([-x[:, 1], x[:, 0]], dim=1)

x = torch.tensor([[1.0, 2.0]], requires_grad=True)
exterior_deriv = compute_exterior_derivative(x, non_conservative_field)
print(f"Exterior derivative:\n{exterior_deriv}")  # Should be non-zero
```

## Loss Functions

You can incorporate the exterior derivative into loss functions for training neural networks:

```python
def exterior_derivative_loss(model, x, target_exterior_deriv=None, weight=1.0):
    """
    Loss function that incorporates the exterior derivative.
    """
    from exterior_derivative import compute_exterior_derivative, exterior_derivative_magnitude
    
    # Compute exterior derivative
    exterior_deriv = compute_exterior_derivative(x, model)
    
    if target_exterior_deriv is not None:
        # Match target exterior derivative
        loss = torch.mean((exterior_deriv - target_exterior_deriv)**2)
    else:
        # Minimize exterior derivative (make field more conservative)
        loss = torch.mean(exterior_derivative_magnitude(exterior_deriv)**2)
    
    return weight * loss

# Usage in training loop
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = exterior_derivative_loss(model, x_batch)
    loss.backward()
    optimizer.step()
```

## Utility Functions

The module provides several utility functions:

```python
from exterior_derivative import is_exact, exterior_derivative_magnitude

# Check if field is conservative (exact)
is_conservative = is_exact(exterior_deriv)

# Compute magnitude of exterior derivative
magnitude = exterior_derivative_magnitude(exterior_deriv)
```

## API Reference

### `ExteriorDerivative`

Main class for computing exterior derivatives.

**Methods:**
- `__init__(function=None)`: Initialize with optional function
- `forward(x, function=None)`: Compute exterior derivative

### `ExteriorDerivativeTransform`

Transform class for use in PyTorch pipelines.

**Methods:**
- `__init__(function)`: Initialize with function
- `__call__(x)`: Apply transformation

### `compute_exterior_derivative(x, function)`

Convenience function for computing exterior derivative.

**Parameters:**
- `x`: Input tensor of shape `(batch_size, n)`
- `function`: Function to differentiate

**Returns:**
- Tensor of shape `(batch_size, n, n)` containing exterior derivative

### `is_exact(exterior_deriv, tol=1e-6)`

Check if exterior derivative is zero (conservative field).

### `exterior_derivative_magnitude(exterior_deriv)`

Compute the magnitude of the exterior derivative.

## Testing

Run the test suite:

```bash
python test_exterior_derivative.py
```

## Applications

The exterior derivative transformation is useful for:

1. **Physics-Informed Neural Networks**: Enforcing physical constraints on learned vector fields
2. **Fluid Dynamics**: Analyzing velocity fields and their rotational properties
3. **Electromagnetic Fields**: Computing curl and divergence properties
4. **Geometric Deep Learning**: Incorporating differential geometric concepts
5. **Regularization**: Encouraging networks to learn more physically meaningful representations

## Limitations

- Currently supports only vector-valued functions with matching input and output dimensions
- Computationally intensive for large batch sizes or high dimensions
- Requires gradients to be enabled on input tensors

## Contributing

Feel free to submit issues and enhancement requests! 