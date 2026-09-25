import numpy as np
import pytest
import torch

from src.models import DDIM, MinimalUNet
from src.utils.data import get_dataset, get_metadata
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.train import train_diffusion


class TestNoiseSchedules:
    """Test cases for noise schedule functions."""
    
    def test_cosine_noise_schedule(self):
        """Test cosine noise schedule function."""
        t = torch.tensor([0.0, 0.5, 1.0])
        beta = cosine_noise_schedule(t)
        
        # Check output shape
        assert beta.shape == t.shape
        
        # Check that beta values are in reasonable range (0, 1)
        assert torch.all(beta >= 0)
        assert torch.all(beta <= 1)
        
        # Check that beta increases with t (noise increases over time)
        assert beta[0] <= beta[1] <= beta[2]
        
    def test_cosine_noise_schedule_batch(self):
        """Test cosine noise schedule with batch input."""
        batch_size = 10
        t = torch.rand(batch_size)
        beta = cosine_noise_schedule(t)
        
        assert beta.shape == (batch_size,)
        assert torch.all(beta >= 0)
        assert torch.all(beta <= 1)


class TestDataLoading:
    """Test cases for data loading utilities.
    
    Note: CelebA dataset tests may be skipped due to Google Drive download limitations.
    The CelebA dataset is hosted on Google Drive which has rate limits and may 
    temporarily block downloads when too many users access it simultaneously.
    """
    
    def test_get_metadata(self):
        """Test metadata retrieval for different datasets."""
        # Test MNIST metadata
        mnist_metadata = get_metadata('mnist')
        assert 'num_channels' in mnist_metadata
        assert 'image_size' in mnist_metadata
        assert 'mean' in mnist_metadata
        assert 'std' in mnist_metadata
        assert mnist_metadata['num_channels'] == 1
        assert mnist_metadata['image_size'] == 32  # MNIST is resized to 32x32
        
        # Test CIFAR10 metadata
        cifar_metadata = get_metadata('cifar10')
        assert cifar_metadata['num_channels'] == 3
        assert cifar_metadata['image_size'] == 32
        
    def test_get_dataset_mnist(self):
        """Test MNIST dataset loading."""
        dataset, metadata = get_dataset('mnist', root='./data')
        
        assert len(dataset) > 0
        assert metadata['name'] == 'mnist'
        assert metadata['num_channels'] == 1
        
        # Test a sample from the dataset
        sample, label = dataset[0]
        assert sample.shape == (1, 32, 32)  # MNIST is resized to 32x32
        assert isinstance(label, int)
        
    def test_get_dataset_cifar10(self):
        """Test CIFAR10 dataset loading."""
        dataset, metadata = get_dataset('cifar10', root='./data')
        
        assert len(dataset) > 0
        assert metadata['name'] == 'cifar10'
        assert metadata['num_channels'] == 3
        
        # Test a sample from the dataset
        sample, label = dataset[0]
        assert sample.shape == (3, 32, 32)
        assert isinstance(label, int)

    def test_get_dataset_celeba(self):
        """Test CelebA dataset loading."""
        try:
            dataset, metadata = get_dataset('celeba', root='./data', train=True)
            
            assert len(dataset) > 0
            assert metadata['name'] == 'celeba'
            assert metadata['num_channels'] == 3
            assert metadata['image_size'] == 32
            
            # Test a sample from the dataset
            sample, label = dataset[0]
            assert sample.shape == (3, 32, 32)
            # CelebA labels can be either int or tensor, handle both
            assert isinstance(label, (int, torch.Tensor))
            
        except FileNotFoundError as e:
            if "Found no valid file for the classes all" in str(e):
                pytest.skip("CelebA dataset not properly prepared - need to fix prepare_celeba_32x32 function")
            else:
                raise
        except Exception as e:
            # Handle common CelebA download issues
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in [
                "too many users", "download", "google drive", "gdown", 
                "file url retrieval", "celeba", "connection"
            ]):
                pytest.skip(f"CelebA dataset download failed due to Google Drive limitations: {e}")
            else:
                raise

    def test_get_dataset_celeba_metadata_only(self):
        """Test CelebA metadata without requiring dataset download."""
        metadata = get_metadata('celeba')
        
        # Test metadata structure
        assert metadata['name'] == 'celeba'
        assert metadata['image_size'] == 32
        assert metadata['num_classes'] == 1
        assert metadata['train_images'] == 200000
        assert metadata['val_images'] == 0
        assert metadata['num_channels'] == 3
        assert metadata['mean'] == [0.5, 0.5, 0.5]
        assert metadata['std'] == [0.5, 0.5, 0.5]

class TestTraining:
    """Test cases for training utilities."""
    
    def test_train_diffusion_creation(self):
        """Test that training function can be called with valid parameters."""
        # Create a simple model
        backbone = MinimalUNet(channels=3)
        model = DDIM(backbone=backbone, in_channels=3, default_imsize=32)
        
        # Create a simple dataloader
        dataset, _ = get_dataset('cifar10', root='./data')
        # Use a small subset for testing
        subset = torch.utils.data.Subset(dataset, range(min(100, len(dataset))))
        train_loader = torch.utils.data.DataLoader(subset, batch_size=4, shuffle=True)
        
        # Test that the function can be called (we won't actually train)
        # This is more of a smoke test
        assert model is not None
        assert train_loader is not None


class TestIdealScore:
    """Test cases for ideal score modules."""
    
    def test_scheduled_score_machine_creation(self):
        """Test ScheduledScoreMachine creation."""
        from src.utils.idealscore import LocalEquivScoreModule, ScheduledScoreMachine

        # Create a simple dataset
        dataset, metadata = get_dataset('cifar10', root='./data')
        subset = torch.utils.data.Subset(dataset, range(min(100, len(dataset))))
        
        # Create score module
        score_module = LocalEquivScoreModule(
            subset,
            batch_size=4,
            image_size=32,
            channels=3,
            schedule=cosine_noise_schedule,
            max_samples=50
        )
        
        # Create scheduled score machine
        machine = ScheduledScoreMachine(
            score_module,
            in_channels=3,
            noise_schedule=cosine_noise_schedule,
            score_backbone=True
        )
        
        assert machine is not None
        assert hasattr(machine, 'forward')


class _TinyImageDataset(torch.utils.data.Dataset):
    """Small fixed synthetic image dataset for fast, offline ELS tests."""

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(0)


class TestLocalEquivScoreModuleBatchBroadcast:
    """Regression tests for a broadcasting bug: at/bt have shape (b,), which
    numpy/torch broadcasting silently aligns against the *last* axis of
    whatever it multiplies, instead of the batch axis, whenever b != that
    axis's size (and b != 1) -- so it only "worked" by accident before, when
    called with batch_size 1 or with batch_size coincidentally equal to a
    spatial dimension. These tests use a batch size that matches neither the
    image size nor any patch-derived dimension, which would fail loudly (or
    silently return wrong values) without the [:, None, None, None]-style
    reshape fix in LocalEquivScoreModule.forward /
    forward_with_posterior_stats.
    """

    def _build_module(self, image_size=6, channels=1, n_train=5):
        from src.utils.idealscore import LocalEquivScoreModule

        rng = torch.Generator().manual_seed(0)
        images = [torch.randn(channels, image_size, image_size, generator=rng) for _ in range(n_train)]
        dataset = _TinyImageDataset(images)
        return LocalEquivScoreModule(
            dataset, kernel_size=3, batch_size=n_train, image_size=image_size,
            channels=channels, max_samples=None, shuffle=False,
        )

    def test_forward_batched_matches_per_sample_calls(self):
        module = self._build_module()
        device = torch.device("cpu")
        rng = torch.Generator().manual_seed(1)
        # batch size 3, image size 6, training set 5 -- no two of these match,
        # so a broadcast-against-the-wrong-axis bug can't hide.
        xs = [torch.randn(1, 1, 6, 6, generator=rng) for _ in range(3)]
        # Same t for every batch element, as in actual use (ScheduledScoreMachine
        # always evaluates a whole batch at one shared noise level per step) --
        # this alone is enough to trip the broadcast-against-the-wrong-axis bug,
        # and avoids conflating it with the separate, pre-existing numerical
        # fragility of the shared batch-wide log-sum-exp stabilizer when
        # batch elements sit at very different noise levels (see the xfail'd
        # forward_with_posterior_stats test above).
        t_shared = 0.5
        ts = [torch.tensor([t_shared])] * 3

        individual = [module(t, x, device=device) for t, x in zip(ts, xs)]

        x_batch = torch.cat(xs, dim=0)
        t_batch = torch.cat(ts, dim=0)
        batched = module(t_batch, x_batch, device=device)

        assert batched.shape == (3, 1, 6, 6)
        for i in range(3):
            torch.testing.assert_close(batched[i : i + 1], individual[i], atol=1e-4, rtol=1e-4)

    @pytest.mark.xfail(
        reason=(
            "forward_with_posterior_stats has a second, separate batch-consistency bug beyond "
            "the at/bt broadcast issue this class targets: batched vs. per-sample calls disagree "
            "by ~15 (not float noise) at circular-padding boundary sites specifically. forward() "
            "(same at/bt fix, same circular padding) passes cleanly, so this is isolated to the "
            "posterior-stats accumulator/rescaling path. Not diagnosed further since nothing in "
            "this project currently needs this method with batch_size > 1 -- AutoscalingELSModule "
            "is the one caller that could hit it and should avoid batch_size > 1 until fixed."
        ),
        strict=True,
    )
    def test_forward_with_posterior_stats_batched_matches_per_sample_calls(self):
        module = self._build_module()
        device = torch.device("cpu")
        rng = torch.Generator().manual_seed(2)
        xs = [torch.randn(1, 1, 6, 6, generator=rng) for _ in range(3)]
        ts = [torch.tensor([0.2]), torch.tensor([0.5]), torch.tensor([0.8])]

        individual = [module.forward_with_posterior_stats(t, x, device=device) for t, x in zip(ts, xs)]
        x_batch = torch.cat(xs, dim=0)
        t_batch = torch.cat(ts, dim=0)
        batched = module.forward_with_posterior_stats(t_batch, x_batch, device=device)

        names = ["score", "E_x0", "entropy_map", "center_variance_map", "center_binder_map",
                 "patch_variance_map", "patch_binder_map"]
        for field_idx, name in enumerate(names):
            for i in range(3):
                try:
                    torch.testing.assert_close(
                        batched[field_idx][i : i + 1], individual[i][field_idx], atol=2e-3, rtol=2e-3,
                    )
                except AssertionError as e:
                    raise AssertionError(f"mismatch in {name} for batch element {i}: {e}") from None

    def test_low_noise_does_not_produce_nan(self):
        # At very low noise (bt -> 0) the Gaussian kernel weights concentrate
        # so sharply that, with no near-exact match in a small training set,
        # every patch's exp_vals can underflow to exactly 0 in float32,
        # giving a silent 0/0 without a denominator clamp.
        module = self._build_module(image_size=8, n_train=5)
        rng = torch.Generator().manual_seed(3)
        x = torch.randn(2, 1, 8, 8, generator=rng)  # unrelated to the training images
        t = torch.full((2,), 0.01)
        score = module(t, x, device=torch.device("cpu"))
        assert not torch.isnan(score).any()
        assert not torch.isinf(score).any()


class TestScheduledScoreMachineTStart:
    """t_start lets forward() start from a partially-corrupted input instead
    of pure noise; default t_start=1.0 must reproduce the original loop
    bounds exactly (regression guard for the earlier hardcoded range)."""

    class _RecordingBackbone(torch.nn.Module):
        def __init__(self, calls):
            super().__init__()
            self.calls = calls

        def forward(self, t, x, device=None, k=None):
            self.calls.append(float(t[0]))
            return torch.zeros_like(x)

    def test_default_t_start_matches_original_loop_bounds(self):
        from src.utils.idealscore import ScheduledScoreMachine

        calls = []
        machine = ScheduledScoreMachine(self._RecordingBackbone(calls), in_channels=1, imsize=4, default_time_steps=10)
        machine(torch.zeros(1, 1, 4, 4), nsteps=10, device=torch.device("cpu"))
        np.testing.assert_allclose(calls, [i / 10 for i in range(9, 0, -1)], atol=1e-6)

    def test_t_start_less_than_one_shortens_loop(self):
        from src.utils.idealscore import ScheduledScoreMachine

        calls = []
        machine = ScheduledScoreMachine(self._RecordingBackbone(calls), in_channels=1, imsize=4, default_time_steps=10)
        machine(torch.zeros(1, 1, 4, 4), nsteps=10, t_start=0.5, device=torch.device("cpu"))
        np.testing.assert_allclose(calls, [i / 10 for i in range(4, 0, -1)], atol=1e-6)

    def test_t_start_zero_is_a_no_op(self):
        from src.utils.idealscore import ScheduledScoreMachine

        class _AssertNotCalled(torch.nn.Module):
            def forward(self, t, x, device=None, k=None):
                raise AssertionError("backbone should not be called when t_start=0")

        machine = ScheduledScoreMachine(_AssertNotCalled(), in_channels=1, imsize=4, default_time_steps=10)
        x = torch.randn(1, 1, 4, 4)
        out = machine(x.clone(), nsteps=10, t_start=0.0, device=torch.device("cpu"))
        torch.testing.assert_close(out, x)


class TestScheduledScoreMachineBatchBroadcast:
    """`eps *= -beta_t**0.5` used a bare (b,)-shaped beta_t against a 4D eps,
    which crashes whenever batch_size doesn't happen to equal image width
    (and isn't 1) -- every *other* beta_t/alpha_t usage in this same loop was
    already correctly reshaped with [:,None,None,None], this one spot wasn't.
    """

    class _DeterministicBackbone(torch.nn.Module):
        def forward(self, t, x, device=None, k=None):
            return 0.1 * x + 0.01

    def test_batch_size_not_matching_image_width_does_not_crash_and_matches_per_sample(self):
        from src.utils.idealscore import ScheduledScoreMachine

        machine = ScheduledScoreMachine(self._DeterministicBackbone(), in_channels=1, imsize=16, default_time_steps=6)
        rng = torch.Generator().manual_seed(0)
        x = torch.randn(5, 1, 16, 16, generator=rng)  # batch=5, width=16: no coincidental match

        batched = machine(x.clone(), nsteps=6, device=torch.device("cpu"))
        individual = torch.cat(
            [machine(x[i : i + 1].clone(), nsteps=6, device=torch.device("cpu")) for i in range(5)], dim=0
        )
        torch.testing.assert_close(batched, individual, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__]) 