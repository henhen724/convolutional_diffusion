#!/usr/bin/env python
"""
Calibrate ELS (and optionally LS) kernel sizes by maximizing
cosine similarity with neural network scores at each diffusion timestep.

For each step i (t = i/nsteps):
  1. Create noised calibration samples
  2. Compute NN noise prediction (UNet and ResNet)
  3. For each candidate kernel size k:
       - Compute ELS score, convert to noise prediction
       - Measure cos(ELS_eps, NN_eps) averaged over calibration samples
  4. Pick k that maximizes cosine similarity

Saves a .pt dict with calibrated scale lists and diagnostic curves.
"""

import argparse
import os
import sys
import time

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.idealscore import LocalEquivScoreModule, LocalScoreModule
from src.utils.latent_data import LatentDataset
from src.utils.noise_schedules import cosine_noise_schedule


def cosine_sim(a, b):
    """Cosine similarity between two flat tensors."""
    return (a @ b) / (a.norm() * b.norm() + 1e-8)


def calibrate_one_step(step, nsteps, calib_clean, nn_model, backbone,
                       candidate_ks, device, n_calib):
    """
    For a single diffusion timestep, compute cosine similarity between
    the analytical backbone and the NN model for each candidate kernel size.
    Returns list of mean cosine similarities (one per candidate k).
    """
    t_val = step / nsteps
    t_tensor = torch.tensor([t_val], device=device)

    beta_t = cosine_noise_schedule(t_tensor)
    alpha_t = 1 - beta_t

    cos_by_k = [0.0] * len(candidate_ks)

    for s in range(n_calib):
        x_clean = calib_clean[s:s+1].to(device)
        eps_true = torch.randn_like(x_clean)
        x_noised = alpha_t.sqrt() * x_clean + beta_t.sqrt() * eps_true

        # NN noise prediction
        with torch.no_grad():
            nn_eps = nn_model(t_tensor, x_noised)

        for ki, k in enumerate(candidate_ks):
            with torch.no_grad():
                score = backbone(t_tensor, x_noised, device=device, k=k)
            # Convert score to noise prediction: eps = -score * sqrt(beta_t)
            analytic_eps = -score * beta_t.sqrt()

            cs = cosine_sim(analytic_eps.flatten(), nn_eps.flatten()).item()
            cos_by_k[ki] += cs

    # Average over calibration samples
    cos_by_k = [c / n_calib for c in cos_by_k]
    return cos_by_k


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate ELS/LS kernel sizes via cosine similarity with NN')
    parser.add_argument('--unet_path', type=str,
                        default='checkpoints/backbone_CelebAHQ_latent_UNet_zeros_nonorm_final.pt')
    parser.add_argument('--resnet_path', type=str,
                        default='checkpoints/backbone_CelebAHQ_latent_ResNet_zeros_nonorm_final.pt')
    parser.add_argument('--latent_path', type=str,
                        default='/scratch/users/hshunt/celeba_hq_latents/celeba_hq_latents.pt')
    parser.add_argument('--stats_path', type=str,
                        default='/scratch/users/hshunt/celeba_hq_latents/celeba_hq_latent_stats.pt')
    parser.add_argument('--output', type=str,
                        default='checkpoints/els_calibrated_scales.pt')
    parser.add_argument('--nsteps', type=int, default=50,
                        help='Number of diffusion steps (must match sampling)')
    parser.add_argument('--n_calib', type=int, default=4,
                        help='Number of noised samples per timestep for averaging')
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='Max training images used per ELS/LS evaluation')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for iterating over training data')
    parser.add_argument('--candidate_ks', type=str,
                        default='3,5,7,9,11,15,19,23,27,31',
                        help='Comma-separated candidate kernel sizes')
    parser.add_argument('--skip_ls', action='store_true',
                        help='Skip LS calibration (only do ELS)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    candidate_ks = [int(k) for k in args.candidate_ks.split(',')]
    nsteps = args.nsteps

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print(f'\nLoading UNet from {args.unet_path}')
    unet_model = torch.load(args.unet_path, map_location=device, weights_only=False)
    unet_model.eval()

    print(f'Loading ResNet from {args.resnet_path}')
    resnet_model = torch.load(args.resnet_path, map_location=device, weights_only=False)
    resnet_model.eval()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f'Loading latent dataset from {args.latent_path}')
    latent_dataset = LatentDataset(
        args.latent_path, stats_path=args.stats_path, normalize=True)

    # ------------------------------------------------------------------
    # Create analytical backbones
    # ------------------------------------------------------------------
    els_backbone = LocalEquivScoreModule(
        dataset=latent_dataset,
        kernel_size=3,
        batch_size=args.batch_size,
        image_size=32,
        channels=4,
        schedule=cosine_noise_schedule,
        max_samples=args.max_samples,
    )

    ls_backbone = None
    if not args.skip_ls:
        ls_backbone = LocalScoreModule(
            dataset=latent_dataset,
            kernel_size=3,
            image_size=32,
            batch_size=args.batch_size,
            schedule=cosine_noise_schedule,
            max_samples=args.max_samples,
        )

    # ------------------------------------------------------------------
    # Calibration samples
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    calib_idx = torch.randperm(len(latent_dataset))[:args.n_calib]
    calib_clean = latent_dataset.latents[calib_idx]  # (n_calib, 4, 32, 32)
    print(f'\nCalibration samples: {args.n_calib}')
    print(f'Candidate kernel sizes: {candidate_ks}')
    print(f'Diffusion steps: {nsteps}')
    print(f'Max training samples per eval: {args.max_samples}')

    backbones_to_calibrate = [('ELS', els_backbone)]
    if ls_backbone is not None:
        backbones_to_calibrate.append(('LS', ls_backbone))

    nn_models = [('UNet', unet_model), ('ResNet', resnet_model)]

    # ------------------------------------------------------------------
    # Main calibration loop
    # ------------------------------------------------------------------
    n_combos = len(backbones_to_calibrate) * len(nn_models)
    total_evals = (nsteps - 1) * len(candidate_ks) * args.n_calib * n_combos
    print(f'\nTotal analytical evaluations: {total_evals}')
    print(f'Estimated time: ~{total_evals * 3 / 60:.0f}–{total_evals * 8 / 60:.0f} min')
    print('=' * 70)

    results = {
        'candidate_ks': candidate_ks,
        'nsteps': nsteps,
        'n_calib': args.n_calib,
        'max_samples': args.max_samples,
    }

    t0 = time.time()

    for bb_name, backbone in backbones_to_calibrate:
        for nn_name, nn_model in nn_models:
            key = f'{nn_name.lower()}_{bb_name.lower()}'
            scales = [3] * nsteps
            cos_curves = []

            print(f'\n--- Calibrating {bb_name} against {nn_name} ---')

            for step in range(1, nsteps):
                t_val = step / nsteps

                cos_by_k = calibrate_one_step(
                    step, nsteps, calib_clean, nn_model, backbone,
                    candidate_ks, device, args.n_calib)

                best_idx = max(range(len(candidate_ks)), key=lambda i: cos_by_k[i])
                best_k = candidate_ks[best_idx]
                best_cos = cos_by_k[best_idx]

                scales[step] = best_k
                cos_curves.append(cos_by_k)

                elapsed = time.time() - t0
                steps_done = step  # within this bb/nn combo
                # rough ETA for this combo
                combo_eta = elapsed / max(steps_done, 1) * (nsteps - 1 - step)

                if step % 5 == 0 or step == nsteps - 1:
                    print(f'  step {step:3d}/{nsteps}  t={t_val:.3f}  '
                          f'best k={best_k:2d} (cos={best_cos:.4f})  '
                          f'[{elapsed:.0f}s elapsed, ~{combo_eta:.0f}s ETA this combo]')

            results[f'{key}_scales'] = scales
            results[f'{key}_cos_curves'] = cos_curves
            print(f'  Scales: {scales}')

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(results, args.output)
    total_time = time.time() - t0
    print(f'\n{"=" * 70}')
    print(f'Results saved to {args.output}')
    print(f'Total time: {total_time:.1f}s ({total_time/60:.1f} min)')

    # Summary
    for bb_name, _ in backbones_to_calibrate:
        for nn_name, _ in nn_models:
            key = f'{nn_name.lower()}_{bb_name.lower()}'
            sc = results[f'{key}_scales']
            print(f'  {nn_name} {bb_name}: k range [{min(sc[1:])}, {max(sc[1:])}]')


if __name__ == '__main__':
    main()
