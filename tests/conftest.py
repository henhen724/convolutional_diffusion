import os
import tempfile

import numpy as np
import pytest
import torch

from src.models import DDIM, MinimalResNet, MinimalUNet
from src.utils.noise_schedules import cosine_noise_schedule


@pytest.fixture(scope="session")
def device():
    """Get the device to run tests on."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


@pytest.fixture
def small_batch():
    """Create a small batch of test data."""
    batch_size = 4
    channels = 3
    image_size = 32
    return torch.randn(batch_size, channels, image_size, image_size)


@pytest.fixture
def small_batch_mnist():
    """Create a small batch of MNIST test data."""
    batch_size = 4
    channels = 1
    image_size = 28
    return torch.randn(batch_size, channels, image_size, image_size)


@pytest.fixture
def time_steps():
    """Create time steps for testing."""
    batch_size = 4
    return torch.rand(batch_size)


@pytest.fixture
def labels():
    """Create labels for conditional testing."""
    batch_size = 4
    return torch.randint(0, 10, (batch_size,))


@pytest.fixture
def unet_model():
    """Create a MinimalUNet model for testing."""
    return MinimalUNet(channels=3)


@pytest.fixture
def resnet_model():
    """Create a MinimalResNet model for testing."""
    return MinimalResNet(channels=3)


@pytest.fixture
def conditional_unet_model():
    """Create a conditional MinimalUNet model for testing."""
    return MinimalUNet(channels=3, conditional=True, num_classes=10)


@pytest.fixture
def conditional_resnet_model():
    """Create a conditional MinimalResNet model for testing."""
    return MinimalResNet(channels=3, conditional=True, num_classes=10)


@pytest.fixture
def ddim_model(unet_model):
    """Create a DDIM model with UNet backbone for testing."""
    return DDIM(backbone=unet_model, in_channels=3, default_imsize=32)


@pytest.fixture
def conditional_ddim_model(conditional_unet_model):
    """Create a conditional DDIM model for testing."""
    return DDIM(backbone=conditional_unet_model, in_channels=3, default_imsize=32)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    class MockDataset:
        def __init__(self, size=100):
            self.size = size
            self.data = torch.randn(size, 3, 32, 32)
            self.targets = torch.randint(0, 10, (size,))
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    return MockDataset()


@pytest.fixture
def mock_mnist_dataset():
    """Create a mock MNIST dataset for testing."""
    class MockMNISTDataset:
        def __init__(self, size=100):
            self.size = size
            self.data = torch.randn(size, 1, 28, 28)
            self.targets = torch.randint(0, 10, (size,))
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    return MockMNISTDataset()


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names."""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_" in item.nodeid:
            item.add_marker(pytest.mark.unit) 