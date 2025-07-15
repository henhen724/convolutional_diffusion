import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestScriptImports:
    """Test that all scripts can be imported without errors."""
    
    def test_training_script_import(self):
        """Test that training_script.py can be imported."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        try:
            import training_script
            assert hasattr(training_script, 'main')
        finally:
            sys.path.pop(0)
    
    def test_eval_script_import(self):
        """Test that eval_script.py can be imported."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        try:
            import eval_script
            assert hasattr(eval_script, 'main')
        finally:
            sys.path.pop(0)
    
    def test_els_script_import(self):
        """Test that els_script.py can be imported."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        try:
            import els_script
            assert hasattr(els_script, 'main')
        finally:
            sys.path.pop(0)


class TestScriptArguments:
    """Test that scripts handle arguments correctly."""
    
    def test_training_script_help(self):
        """Test that training_script.py shows help without error."""
        script_path = Path(__file__).parent.parent / "scripts" / "training_script.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
    
    def test_eval_script_help(self):
        """Test that eval_script.py shows help without error."""
        script_path = Path(__file__).parent.parent / "scripts" / "eval_script.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
    
    def test_els_script_help(self):
        """Test that els_script.py shows help without error."""
        script_path = Path(__file__).parent.parent / "scripts" / "els_script.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()


class TestScalesCalibration:
    """Test scales_calibration.py functionality."""
    
    def test_scales_calibration_import(self):
        """Test that scales_calibration.py can be imported."""
        from scripts.scales_calibration import calibrate
        assert callable(calibrate)
    
    def test_scales_calibration_help(self):
        """Test that scales_calibration.py shows help without error."""
        script_path = Path(__file__).parent.parent / "scripts" / "scales_calibration.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()


class TestScriptDependencies:
    """Test that scripts have all required dependencies."""
    
    def test_script_dependencies_available(self):
        """Test that all required modules are available for scripts."""
        required_modules = [
            'torch',
            'torchvision',
            'numpy',
            'matplotlib',
            'argparse',
            'src.models',
            'src.utils.data',
            'src.utils.noise_schedules',
            'src.utils.idealscore',
            'src.utils.train'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError as e:
                pytest.fail(f"Required module {module} not available: {e}")


class TestScriptPaths:
    """Test that scripts can find their dependencies."""
    
    def test_script_paths(self):
        """Test that scripts can import their dependencies."""
        # Add the project root to Python path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        try:
            # Test importing main modules
            from src.models import DDIM, MinimalResNet, MinimalUNet
            from src.utils.data import get_dataset
            from src.utils.idealscore import ScheduledScoreMachine
            from src.utils.noise_schedules import cosine_noise_schedule
            from src.utils.train import train_diffusion
            
            assert all([DDIM, MinimalUNet, MinimalResNet, get_dataset, 
                       cosine_noise_schedule, ScheduledScoreMachine, train_diffusion])
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    pytest.main([__file__]) 