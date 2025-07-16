#!/bin/bash
# Script to activate the diffusion_env conda environment in Cursor terminal
# Usage: source activate_env.sh

echo "Sourcing bash profile to load conda..."
source ~/.bash_profile

echo "Activating diffusion_env..."
conda activate diffusion_env

echo "Environment activated! Current environment:"
conda info --envs | grep "*"

echo "Python version:"
python --version

echo "Ready to work with convolutional diffusion project!" 