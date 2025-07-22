import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import DDIM, MinimalResNet, MinimalUNet
from src.utils.data import get_dataset
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.idealscore import LocalEquivBordersScoreModule, ScheduledScoreMachine
from src.utils.exterior_derivative import compute_exterior_derivative, exterior_derivative_magnitude


class ExteriorDerivativeAnalyzer:
    """
    Analyze exterior derivative magnitudes of score function models 
    during the reverse diffusion process.
    """
    
    def __init__(self, dataset_name='mnist', device=None, nsteps=20):
        """
        Initialize the analyzer.
        
        Args:
            dataset_name: Name of dataset (mnist, cifar10, fashionmnist)
            device: Computing device
            nsteps: Number of diffusion steps
        """
        self.dataset_name = dataset_name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nsteps = nsteps
        self.noise_schedule = cosine_noise_schedule
        
        # Load dataset
        self.dataset, self.metadata = get_dataset(dataset_name, root='./data')
        self.in_channels = self.metadata['num_channels']
        self.image_size = self.metadata['image_size']
        
        print(f"Initialized analyzer for {dataset_name}")
        print(f"Image size: {self.image_size}, Channels: {self.in_channels}")
        print(f"Device: {self.device}")
    
    def load_models(self, unet_path=None, resnet_path=None, scales_path=None):
        """
        Load UNet, ResNet, and ELS models.
        
        Args:
            unet_path: Path to UNet checkpoint
            resnet_path: Path to ResNet checkpoint  
            scales_path: Path to scales file for ELS
        """
        self.models = {}
        
        # Auto-detect paths if not provided
        checkpoints_dir = Path('./checkpoints')
        
        if unet_path is None:
            # Find UNet model for this dataset
            pattern = f"backbone_{self.dataset_name.upper()}_UNet_*.pt"
            unet_files = list(checkpoints_dir.glob(pattern))
            if unet_files:
                unet_path = unet_files[0]
        
        if resnet_path is None:
            # Find ResNet model for this dataset
            pattern = f"backbone_{self.dataset_name.upper()}_ResNet_*.pt"
            resnet_files = list(checkpoints_dir.glob(pattern))
            if resnet_files:
                resnet_path = resnet_files[0]
        
        if scales_path is None:
            # Find scales file for this dataset
            pattern = f"scales_{self.dataset_name.upper()}_*.pt"
            scales_files = list(checkpoints_dir.glob(pattern))
            if scales_files:
                scales_path = scales_files[0]
        
        # Load UNet model
        if unet_path and os.path.exists(unet_path):
            print(f"Loading UNet from {unet_path}")
            self.models['unet'] = torch.load(unet_path, map_location=self.device, weights_only=False)
            self.models['unet'].eval()
            self.models['unet'].to(self.device)
        else:
            print("UNet model not found, creating default")
            unet = MinimalUNet(channels=self.in_channels, conditional=False)
            self.models['unet'] = DDIM(backbone=unet, in_channels=self.in_channels, 
                                     default_imsize=self.image_size)
            self.models['unet'].to(self.device)
        
        # Load ResNet model
        if resnet_path and os.path.exists(resnet_path):
            print(f"Loading ResNet from {resnet_path}")
            self.models['resnet'] = torch.load(resnet_path, map_location=self.device, weights_only=False)
            self.models['resnet'].eval()
            self.models['resnet'].to(self.device)
        else:
            print("ResNet model not found, creating default")
            resnet = MinimalResNet(channels=self.in_channels, conditional=False)
            self.models['resnet'] = DDIM(backbone=resnet, in_channels=self.in_channels,
                                       default_imsize=self.image_size)
            self.models['resnet'].to(self.device)
        
        # Load scales and create ELS model
        scales = None
        if scales_path and os.path.exists(scales_path):
            print(f"Loading scales from {scales_path}")
            scales = torch.load(scales_path, map_location=self.device, weights_only=False)
        
        print("Creating ELS model")
        els_backbone = LocalEquivBordersScoreModule(
            self.dataset,
            batch_size=64,
            image_size=self.image_size,
            channels=self.in_channels,
            schedule=self.noise_schedule,
            max_samples=1000
        )
        
        self.models['els'] = ScheduledScoreMachine(
            els_backbone,
            in_channels=self.in_channels,
            imsize=self.image_size,
            noise_schedule=self.noise_schedule,
            score_backbone=True,
            scales=scales
        )
        self.models['els'].to(self.device)
        
        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def analyze_single_realization(self, noise_seed=None, save_path=None):
        """
        Analyze exterior derivative for a single noise realization.
        
        Args:
            noise_seed: Random seed for reproducibility
            save_path: Path to save results
            
        Returns:
            Dictionary with results for each model
        """
        if noise_seed is not None:
            torch.manual_seed(noise_seed)
            
        # Generate initial white noise
        x_init = torch.randn(1, self.in_channels, self.image_size, self.image_size, device=self.device)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nAnalyzing {model_name}...")
            
            # Track exterior derivative during reverse process
            ed_magnitudes = []
            timesteps = []
            generated_images = []
            
            # Clone initial noise for this model
            x = x_init.clone()
            
            # Run reverse diffusion process
            for i in range(self.nsteps, 0, -1):
                t = i * torch.ones(1, device=self.device) / self.nsteps
                timesteps.append(i)
                
                # Get score function prediction
                if model_name == 'els':
                    # ELS uses different interface
                    def score_function(x_flat):
                        # Reshape flattened input back to image shape
                        x_img = x_flat.view(1, self.in_channels, self.image_size, self.image_size)
                        return model.backbone(t, x_img, device=self.device).flatten()
                    
                    # Flatten x for exterior derivative computation
                    x_flat = x.view(1, -1)
                    
                    # Compute exterior derivative
                    try:
                        exterior_deriv = compute_exterior_derivative(x_flat, score_function)
                        ed_magnitude = exterior_derivative_magnitude(exterior_deriv)
                        ed_magnitudes.append(ed_magnitude.item())
                    except Exception as e:
                        print(f"Error computing exterior derivative for {model_name} at step {i}: {e}")
                        ed_magnitudes.append(0.0)
                        
                    # Update x using ELS forward pass
                    x = model.backbone(t, x, device=self.device)
                    
                else:
                    # UNet/ResNet models
                    def score_function(x_flat):
                        # Reshape flattened input back to image shape  
                        x_img = x_flat.view(1, self.in_channels, self.image_size, self.image_size)
                        return model.backbone(t, x_img).flatten()
                    
                    # Flatten x for exterior derivative computation
                    x_flat = x.view(1, -1)
                    
                    # Compute exterior derivative
                    try:
                        exterior_deriv = compute_exterior_derivative(x_flat, score_function)
                        ed_magnitude = exterior_derivative_magnitude(exterior_deriv)
                        ed_magnitudes.append(ed_magnitude.item())
                    except Exception as e:
                        print(f"Error computing exterior derivative for {model_name} at step {i}: {e}")
                        ed_magnitudes.append(0.0)
                        
                    # Update x using DDIM step
                    beta_t = self.noise_schedule(t).to(self.device)
                    score = model.backbone(t, x)
                    
                    alpha_t = 1 - beta_t
                    beta_t_prev = self.noise_schedule(t - 1/self.nsteps).to(self.device)
                    alpha_t_prev = 1 - beta_t_prev
                    
                    x *= torch.sqrt(alpha_t_prev/alpha_t)[:, None, None, None]
                    score_correction = (torch.sqrt(beta_t_prev)[:, None, None, None] - 
                                      torch.sqrt(alpha_t_prev/alpha_t)[:, None, None, None] * 
                                      torch.sqrt(beta_t)[:, None, None, None]) * score
                    x += score_correction
                
                # Store intermediate result
                generated_images.append(x.clone().detach().cpu())
            
            results[model_name] = {
                'timesteps': timesteps,
                'ed_magnitudes': ed_magnitudes,
                'final_image': x.detach().cpu(),
                'intermediate_images': generated_images,
                'initial_noise': x_init.detach().cpu()
            }
            
            print(f"Completed {model_name}: mean ED magnitude = {np.mean(ed_magnitudes):.4f}")
        
        # Save results if path provided
        if save_path:
            self._save_results(results, save_path)
            
        return results
    
    def analyze_multiple_realizations(self, num_realizations=10, save_dir=None):
        """
        Analyze exterior derivative for multiple noise realizations.
        
        Args:
            num_realizations: Number of different noise seeds to analyze
            save_dir: Directory to save results
            
        Returns:
            Dictionary with aggregated results
        """
        all_results = []
        
        for realization in range(num_realizations):
            print(f"\n{'='*50}")
            print(f"REALIZATION {realization + 1}/{num_realizations}")
            print(f"{'='*50}")
            
            results = self.analyze_single_realization(noise_seed=realization)
            all_results.append(results)
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        # Save aggregated results
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self._save_aggregated_results(aggregated, save_dir)
            self._plot_aggregated_results(aggregated, save_dir)
        
        return aggregated
    
    def _save_results(self, results, save_path):
        """Save single realization results."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save numerical data
        data_to_save = {}
        for model_name, result in results.items():
            data_to_save[model_name] = {
                'timesteps': result['timesteps'],
                'ed_magnitudes': result['ed_magnitudes']
            }
        
        torch.save(data_to_save, save_path / 'ed_analysis_single.pt')
        
        # Create plots
        self._plot_single_results(results, save_path)
    
    def _aggregate_results(self, all_results):
        """Aggregate results across multiple realizations."""
        model_names = list(all_results[0].keys())
        num_realizations = len(all_results)
        
        aggregated = {}
        
        for model_name in model_names:
            # Collect all ED magnitudes
            all_ed_magnitudes = []
            timesteps = all_results[0][model_name]['timesteps']
            
            for results in all_results:
                all_ed_magnitudes.append(results[model_name]['ed_magnitudes'])
            
            # Convert to numpy array for easier manipulation
            all_ed_magnitudes = np.array(all_ed_magnitudes)  # Shape: (num_realizations, num_timesteps)
            
            aggregated[model_name] = {
                'timesteps': timesteps,
                'ed_magnitudes_mean': np.mean(all_ed_magnitudes, axis=0),
                'ed_magnitudes_std': np.std(all_ed_magnitudes, axis=0),
                'ed_magnitudes_all': all_ed_magnitudes,
                'num_realizations': num_realizations
            }
        
        return aggregated
    
    def _save_aggregated_results(self, aggregated, save_dir):
        """Save aggregated results."""
        # Save numerical data
        torch.save(aggregated, save_dir / 'ed_analysis_aggregated.pt')
        
        # Save summary statistics
        summary = {}
        for model_name, result in aggregated.items():
            summary[model_name] = {
                'mean_ed_magnitude': float(np.mean(result['ed_magnitudes_mean'])),
                'std_ed_magnitude': float(np.mean(result['ed_magnitudes_std'])),
                'max_ed_magnitude': float(np.max(result['ed_magnitudes_mean'])),
                'num_realizations': result['num_realizations']
            }
        
        import json
        with open(save_dir / 'summary_stats.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _plot_single_results(self, results, save_path):
        """Create plots for single realization results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot ED magnitudes over time
        ax = axes[0, 0]
        for model_name, result in results.items():
            ax.plot(result['timesteps'], result['ed_magnitudes'], 
                   label=model_name, marker='o', linewidth=2)
        ax.set_xlabel('Diffusion Timestep')
        ax.set_ylabel('Exterior Derivative Magnitude')
        ax.set_title('Exterior Derivative During Reverse Process')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot log scale
        ax = axes[0, 1]
        for model_name, result in results.items():
            ed_mags = np.array(result['ed_magnitudes'])
            ed_mags = ed_mags[ed_mags > 0]  # Remove zeros for log plot
            timesteps = np.array(result['timesteps'])[:len(ed_mags)]
            ax.semilogy(timesteps, ed_mags, label=model_name, marker='o', linewidth=2)
        ax.set_xlabel('Diffusion Timestep')
        ax.set_ylabel('Exterior Derivative Magnitude (log)')
        ax.set_title('Exterior Derivative (Log Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Show final generated images
        ax = axes[1, 0]
        model_names = list(results.keys())
        if len(model_names) >= 1:
            img = results[model_names[0]]['final_image'].squeeze()
            if img.dim() == 3 and img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(f'Generated Image ({model_names[0]})')
            ax.axis('off')
        
        # Show comparison of mean ED magnitudes
        ax = axes[1, 1]
        means = [np.mean(result['ed_magnitudes']) for result in results.values()]
        model_names = list(results.keys())
        bars = ax.bar(model_names, means, alpha=0.7)
        ax.set_ylabel('Mean Exterior Derivative Magnitude')
        ax.set_title('Mean ED Magnitude Comparison')
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.01,
                   f'{mean_val:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path / 'ed_analysis_single.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_aggregated_results(self, aggregated, save_dir):
        """Create plots for aggregated results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot mean ED magnitudes with error bars
        ax = axes[0, 0]
        for model_name, result in aggregated.items():
            timesteps = result['timesteps']
            means = result['ed_magnitudes_mean']
            stds = result['ed_magnitudes_std']
            
            ax.errorbar(timesteps, means, yerr=stds, 
                       label=f"{model_name} (n={result['num_realizations']})", 
                       marker='o', linewidth=2, capsize=5)
        
        ax.set_xlabel('Diffusion Timestep')
        ax.set_ylabel('Exterior Derivative Magnitude')
        ax.set_title('Mean Exterior Derivative During Reverse Process')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot coefficient of variation
        ax = axes[0, 1]
        for model_name, result in aggregated.items():
            timesteps = result['timesteps']
            means = result['ed_magnitudes_mean']
            stds = result['ed_magnitudes_std']
            cv = stds / (means + 1e-8)  # Add small epsilon to avoid division by zero
            
            ax.plot(timesteps, cv, label=model_name, marker='o', linewidth=2)
        
        ax.set_xlabel('Diffusion Timestep')
        ax.set_ylabel('Coefficient of Variation (std/mean)')
        ax.set_title('Variability of Exterior Derivative')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Heatmap of all realizations
        ax = axes[1, 0]
        model_names = list(aggregated.keys())
        if len(model_names) >= 1:
            first_model = model_names[0]
            ed_data = aggregated[first_model]['ed_magnitudes_all']
            
            im = ax.imshow(ed_data, aspect='auto', cmap='viridis')
            ax.set_xlabel('Diffusion Timestep')
            ax.set_ylabel('Realization')
            ax.set_title(f'ED Magnitudes Across Realizations ({first_model})')
            plt.colorbar(im, ax=ax)
        
        # Summary statistics comparison
        ax = axes[1, 1]
        model_names = list(aggregated.keys())
        mean_eds = [np.mean(result['ed_magnitudes_mean']) for result in aggregated.values()]
        std_eds = [np.mean(result['ed_magnitudes_std']) for result in aggregated.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, mean_eds, width, label='Mean ED', alpha=0.7)
        ax.bar(x + width/2, std_eds, width, label='Mean Std', alpha=0.7)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Magnitude')
        ax.set_title('Summary Statistics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'ed_analysis_aggregated.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze exterior derivative of score function models')
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'cifar10', 'fashionmnist'],
                       help='Dataset to analyze')
    parser.add_argument('--single', action='store_true', 
                       help='Run single realization analysis')
    parser.add_argument('--multiple', type=int, default=0,
                       help='Number of realizations for multiple analysis')
    parser.add_argument('--nsteps', type=int, default=20,
                       help='Number of diffusion steps')
    parser.add_argument('--output_dir', type=str, default='./results/exterior_derivative',
                       help='Output directory for results')
    parser.add_argument('--unet_path', type=str, default=None,
                       help='Path to UNet checkpoint')
    parser.add_argument('--resnet_path', type=str, default=None,
                       help='Path to ResNet checkpoint')
    parser.add_argument('--scales_path', type=str, default=None,
                       help='Path to scales file for ELS')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ExteriorDerivativeAnalyzer(
        dataset_name=args.dataset,
        nsteps=args.nsteps
    )
    
    # Load models
    analyzer.load_models(
        unet_path=args.unet_path,
        resnet_path=args.resnet_path,
        scales_path=args.scales_path
    )
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    if args.single:
        print("\nRunning single realization analysis...")
        results = analyzer.analyze_single_realization(
            noise_seed=42,
            save_path=output_dir / 'single_realization'
        )
        print(f"Results saved to {output_dir / 'single_realization'}")
    
    if args.multiple > 0:
        print(f"\nRunning multiple realization analysis ({args.multiple} realizations)...")
        results = analyzer.analyze_multiple_realizations(
            num_realizations=args.multiple,
            save_dir=output_dir / 'multiple_realizations'
        )
        print(f"Results saved to {output_dir / 'multiple_realizations'}")
    
    if not args.single and args.multiple == 0:
        print("No analysis specified. Use --single or --multiple N")


if __name__ == '__main__':
    main() 