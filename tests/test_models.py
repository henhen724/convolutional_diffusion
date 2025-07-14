import torch
import pytest
import numpy as np
from src.models import DDIM, MinimalUNet, MinimalResNet
from src.utils.noise_schedules import cosine_noise_schedule


class TestMinimalUNet:
    """Test cases for MinimalUNet model."""
    
    def test_minimal_unet_creation(self):
        """Test that MinimalUNet can be created with different parameters."""
        model = MinimalUNet(channels=3)
        assert model is not None
        assert hasattr(model, 'forward')
        
    def test_minimal_unet_forward_pass(self):
        """Test forward pass of MinimalUNet."""
        model = MinimalUNet(channels=3)
        batch_size = 4
        t = torch.rand(batch_size)
        x = torch.randn(batch_size, 3, 32, 32)
        
        output = model(t, x)
        assert output.shape == (batch_size, 3, 32, 32)
        
    def test_minimal_unet_conditional(self):
        """Test conditional forward pass of MinimalUNet."""
        model = MinimalUNet(channels=3, conditional=True, num_classes=10)
        batch_size = 4
        t = torch.rand(batch_size)
        x = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))
        
        output = model(t, x, label=labels)
        assert output.shape == (batch_size, 3, 32, 32)


class TestMinimalResNet:
    """Test cases for MinimalResNet model."""
    
    def test_minimal_resnet_creation(self):
        """Test that MinimalResNet can be created with different parameters."""
        model = MinimalResNet(channels=3)
        assert model is not None
        assert hasattr(model, 'forward')
        
    def test_minimal_resnet_forward_pass(self):
        """Test forward pass of MinimalResNet."""
        model = MinimalResNet(channels=3)
        batch_size = 4
        t = torch.rand(batch_size)
        x = torch.randn(batch_size, 3, 32, 32)
        
        output = model(t, x)
        assert output.shape == (batch_size, 3, 32, 32)
        
    def test_minimal_resnet_conditional(self):
        """Test conditional forward pass of MinimalResNet."""
        model = MinimalResNet(channels=3, conditional=True, num_classes=10)
        batch_size = 4
        t = torch.rand(batch_size)
        x = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))
        
        output = model(t, x, label=labels)
        assert output.shape == (batch_size, 3, 32, 32)


class TestDDIM:
    """Test cases for DDIM model."""
    
    def test_ddim_creation(self):
        """Test that DDIM can be created with different parameters."""
        backbone = MinimalUNet(channels=3)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'sample')
        
    def test_ddim_forward_pass(self):
        """Test forward pass of DDIM."""
        backbone = MinimalUNet(channels=3)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        batch_size = 4
        t = torch.rand(batch_size)
        x = torch.randn(batch_size, 3, 32, 32)
        
        output = model(t, x)
        assert output.shape == (batch_size, 3, 32, 32)
        
    def test_ddim_conditional_forward(self):
        """Test conditional forward pass of DDIM."""
        backbone = MinimalUNet(channels=3, conditional=True, num_classes=10)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        batch_size = 4
        t = torch.rand(batch_size)
        x = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))
        
        output = model(t, x, label=labels)
        assert output.shape == (batch_size, 3, 32, 32)
        
    def test_ddim_sample(self):
        """Test sampling from DDIM."""
        backbone = MinimalUNet(channels=3)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        batch_size = 4
        
        # Test sampling with default parameters
        samples = model.sample(batch_size=batch_size, nsteps=10)
        assert samples.shape == (batch_size, 3, 32, 32)
        
        # Test sampling with provided noise
        x = torch.randn(batch_size, 3, 32, 32)
        samples = model.sample(batch_size=batch_size, x=x, nsteps=10)
        assert samples.shape == (batch_size, 3, 32, 32)
        
    def test_ddim_sample_conditional(self):
        """Test conditional sampling from DDIM."""
        backbone = MinimalUNet(channels=3, conditional=True, num_classes=10)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        batch_size = 4
        labels = torch.randint(0, 10, (batch_size,))
        
        samples = model.sample(batch_size=batch_size, nsteps=10, label=labels)
        assert samples.shape == (batch_size, 3, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__]) 