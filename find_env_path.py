#!/usr/bin/env python3
"""
Script to find the correct path to the diffusion_env environment.
This helps configure Pylance to use the right Python interpreter.
"""

import os
import subprocess
import sys
from pathlib import Path


def find_conda_env_path(env_name):
    """Find the path to a conda environment."""
    try:
        # Run conda info to get environment paths
        result = subprocess.run(
            ["conda", "info", "--envs"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if env_name in line:
                # Extract the path from the line
                parts = line.split()
                if len(parts) >= 2:
                    env_path = parts[-1]
                    return Path(env_path)
        
        return None
        
    except subprocess.CalledProcessError:
        print("Error: Could not run 'conda info --envs'")
        return None
    except FileNotFoundError:
        print("Error: conda command not found")
        return None

def find_python_in_env(env_path):
    """Find the Python executable in the environment."""
    if not env_path or not env_path.exists():
        return None
    
    # Try different possible Python paths
    possible_paths = [
        env_path / "python.exe",  # Windows
        env_path / "bin" / "python",  # Unix/Linux/Mac
        env_path / "Scripts" / "python.exe",  # Windows alternative
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

def main():
    """Main function to find and display environment information."""
    print("🔍 Finding diffusion_env environment...")
    print("-" * 50)
    
    # Find the environment path
    env_path = find_conda_env_path("diffusion_env")
    
    if not env_path:
        print("❌ Could not find diffusion_env environment")
        print("\nPossible solutions:")
        print("1. Make sure the environment exists:")
        print("   conda env list")
        print("2. Create the environment if it doesn't exist:")
        print("   conda env create -f environment.yml")
        return 1
    
    print(f"✅ Found environment at: {env_path}")
    
    # Find Python executable
    python_path = find_python_in_env(env_path)
    
    if not python_path:
        print("❌ Could not find Python executable in the environment")
        return 1
    
    print(f"✅ Found Python at: {python_path}")
    
    # Get Python version
    try:
        result = subprocess.run(
            [str(python_path), "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        python_version = result.stdout.strip()
        print(f"✅ Python version: {python_version}")
    except subprocess.CalledProcessError:
        print("⚠️  Could not determine Python version")
    
    # Check if key packages are available
    print("\n📦 Checking key packages...")
    packages_to_check = ["torch", "numpy", "matplotlib", "scipy"]
    
    for package in packages_to_check:
        try:
            result = subprocess.run(
                [str(python_path), "-c", f"import {package}; print(f'{package}: OK')"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"   ✅ {result.stdout.strip()}")
        except subprocess.CalledProcessError:
            print(f"   ❌ {package}: Not found")
    
    # Display configuration information
    print("\n📋 Configuration Information:")
    print("-" * 50)
    
    # For VS Code settings
    print("\nFor .vscode/settings.json:")
    print(f'"python.defaultInterpreterPath": "{python_path}",')
    
    # For pyrightconfig.json
    print("\nFor pyrightconfig.json:")
    print(f'"venvPath": "{env_path.parent}",')
    print(f'"venv": "{env_path.name}",')
    
    # For manual activation
    print("\nFor manual activation:")
    print(f"conda activate {env_path.name}")
    print(f"# or")
    print(f"source {env_path}/bin/activate  # Unix/Linux/Mac")
    print(f"# or")
    print(f"{env_path}/Scripts/activate.bat  # Windows")
    
    # Check if we're in the right environment
    current_env = os.environ.get('CONDA_DEFAULT_ENV')
    if current_env == 'diffusion_env':
        print(f"\n✅ Currently in correct environment: {current_env}")
    else:
        print(f"\n⚠️  Currently in environment: {current_env}")
        print("   Consider activating diffusion_env for development")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 