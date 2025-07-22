import os
import tempfile

import numpy as np
import pytest
import torch
from torch.serialization import add_safe_globals

from src.models import DDIM, MinimalResNet, MinimalUNet
from src.utils.data import get_dataset
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.train import train_diffusion

# Register model classes as safe globals for model saving/loading
add_safe_globals(safe_globals=[DDIM, MinimalUNet, MinimalResNet])


class TestFullPipeline:
    """Integration tests for the full diffusion pipeline."""
    
    def test_unet_training_pipeline(self):
        """Test complete UNet training pipeline with small dataset."""
        # Create a small dataset for testing
        dataset, metadata = get_dataset('cifar10', root='./data')
        subset = torch.utils.data.Subset(dataset, range(min(50, len(dataset))))
        train_loader = torch.utils.data.DataLoader(subset, batch_size=4, shuffle=True)
        
        # Create model
        backbone = MinimalUNet(channels=3)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        
        # Test that model can process a batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= 2:  # Only test first 2 batches
                break
                
            batch_size = images.shape[0]
            t = torch.rand(batch_size)
            
            # Test forward pass
            output = model(t, images)
            assert output.shape == images.shape
            
            # Test that gradients can be computed
            loss = torch.nn.functional.mse_loss(output, images)
            loss.backward()
            
            # Check that gradients exist
            for param in model.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()
                    break
    
    def test_resnet_training_pipeline(self):
        """Test complete ResNet training pipeline with small dataset."""
        # Create a small dataset for testing
        dataset, metadata = get_dataset('cifar10', root='./data')
        subset = torch.utils.data.Subset(dataset, range(min(50, len(dataset))))
        train_loader = torch.utils.data.DataLoader(subset, batch_size=4, shuffle=True)
        
        # Create model
        backbone = MinimalResNet(channels=3)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        
        # Test that model can process a batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= 2:  # Only test first 2 batches
                break
                
            batch_size = images.shape[0]
            t = torch.rand(batch_size)
            
            # Test forward pass
            output = model(t, images)
            assert output.shape == images.shape
            
            # Test that gradients can be computed
            loss = torch.nn.functional.mse_loss(output, images)
            loss.backward()
            
            # Check that gradients exist
            for param in model.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()
                    break
    
    def test_conditional_training_pipeline(self):
        """Test conditional training pipeline."""
        # Create a small dataset for testing
        dataset, metadata = get_dataset('cifar10', root='./data')
        subset = torch.utils.data.Subset(dataset, range(min(50, len(dataset))))
        train_loader = torch.utils.data.DataLoader(subset, batch_size=4, shuffle=True)
        
        # Create conditional model
        backbone = MinimalUNet(channels=3, conditional=True, num_classes=10)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        
        # Test that model can process a batch with labels
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= 2:  # Only test first 2 batches
                break
                
            batch_size = images.shape[0]
            t = torch.rand(batch_size)
            
            # Test forward pass with labels
            output = model(t, images, label=labels)
            assert output.shape == images.shape
            
            # Test that gradients can be computed
            loss = torch.nn.functional.mse_loss(output, images)
            loss.backward()
            
            # Check that gradients exist
            for param in model.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()
                    break
    
    def test_sampling_pipeline(self):
        """Test complete sampling pipeline."""
        # Create model
        backbone = MinimalUNet(channels=3)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        
        # Test sampling
        batch_size = 4
        samples = model.sample(batch_size=batch_size, nsteps=5)  # Use fewer steps for testing
        
        assert samples.shape == (batch_size, 3, 32, 32)
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()
    
    def test_conditional_sampling_pipeline(self):
        """Test conditional sampling pipeline."""
        # Create conditional model
        backbone = MinimalUNet(channels=3, conditional=True, num_classes=10)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        
        # Test conditional sampling
        batch_size = 4
        labels = torch.randint(0, 10, (batch_size,))
        samples = model.sample(batch_size=batch_size, nsteps=5, label=labels)  # Use fewer steps for testing
        
        assert samples.shape == (batch_size, 3, 32, 32)
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()


class TestModelSaving:
    """Test model saving and loading functionality."""
    
    def test_model_save_load(self):
        """Test that models can be saved and loaded correctly."""
        # Create model
        backbone = MinimalUNet(channels=3)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        
        # Create test input
        batch_size = 2
        t = torch.rand(batch_size)
        x = torch.randn(batch_size, 3, 32, 32)
        
        # Get original output
        original_output = model(t, x)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            torch.save(model, tmp_file.name)
            tmp_file_path = tmp_file.name
        
        try:
            # Load model
            loaded_model = torch.load(tmp_file_path, weights_only=False)
            
            # Get output from loaded model
            loaded_output = loaded_model(t, x)
            
            # Check that outputs are the same
            assert torch.allclose(original_output, loaded_output, atol=1e-6)
        finally:
            # Clean up
            try:
                os.unlink(tmp_file_path)
            except (OSError, PermissionError):
                # On Windows, file might still be in use, ignore cleanup errors
                pass


class TestDataConsistency:
    """Test data consistency across different operations."""
    
    def test_noise_schedule_consistency(self):
        """Test that noise schedule is consistent across different time steps."""
        t1 = torch.tensor([0.5])
        t2 = torch.tensor([0.5])
        
        beta1 = cosine_noise_schedule(t1)
        beta2 = cosine_noise_schedule(t2)
        
        assert torch.allclose(beta1, beta2)
    
    def test_model_output_consistency(self):
        """Test that model outputs are consistent for same inputs."""
        backbone = MinimalUNet(channels=3)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        model.eval()  # Set to evaluation mode
        
        batch_size = 2
        t = torch.rand(batch_size)
        x = torch.randn(batch_size, 3, 32, 32)
        
        # Get outputs twice
        output1 = model(t, x)
        output2 = model(t, x)
        
        # Check that outputs are the same
        assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__]) 