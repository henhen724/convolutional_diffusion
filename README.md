# An analytic theory of creativity in convolutional diffusion models

Code for the paper "An Analytic Theory of Creativity" (https://arxiv.org/abs/2412.20292)

## Setup

```bash
conda env create -f environment.yml
conda activate diffusion_env
```

## Training

```bash
python scripts/training_script.py --epochs 300 --dataset cifar10 --conditional --mode zeros --layers 8 --resnet --homedir [directory for model checkpoints]
```

## Scale Calibration

```bash
python scripts/scales_calibration.py --dataset cifar10 --modelfile [model file name] --kfilename scalesfile --tld [directory for scales files] --scoremoduletype [one of LS, ELS, bbELS] --conditional --kernelsizes 3 5 7 9 11 13 15 17 --nsteps 20 --nsamps 10
```

## Generate ELS Samples

```bash
python scripts/els_script.py --expname cifar10_resnet_els --dataset cifar10 --scoremoduletype [one of LS, ELS, bbELS] --conditional --scalesfile [path to scales] --numiters 100
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run tests with coverage
python -m pytest --cov=src --cov-report=term-missing tests/

# Or use the coverage script
python run_coverage.py
```

### Coverage Reports

The project includes code coverage checking to ensure comprehensive testing:

- **Terminal Report**: Shows missing lines in the terminal output
- **HTML Report**: Detailed coverage report in `htmlcov/` directory
- **XML Report**: Coverage data in `coverage.xml` for CI/CD integration

To view the HTML coverage report:

```bash
# After running coverage, open in browser
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
xdg-open htmlcov/index.html  # Linux
```
