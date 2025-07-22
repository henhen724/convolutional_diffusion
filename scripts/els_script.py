import argparse
import os
import shutil
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.models import DDIM, MinimalResNet, MinimalUNet
from src.utils.data import get_dataset
from src.utils.idealscore import (
    IdealScoreModule,
    LocalEquivBordersScoreModule,
    LocalEquivScoreModule,
    LocalScoreModule,
    ScheduledScoreMachine,
)
from src.utils.noise_schedules import cosine_noise_schedule

## GENERATES ELS MACHINE OUTPUTS
##
## Updated file structure:
## - Scales files: ./checkpoints/scales_DATASET_MODEL_*.pt
## - Model files: ./checkpoints/backbone_DATASET_MODEL_*.pt  
## - Output results: ./results/EXPNAME/
##
## Auto-detects scales files from checkpoints directory if --scalesfile not specified
##
## Example usage:
## python scripts/els_script.py --dataset cifar10 --conditional --scoremoduletype bbELS --numiters 50

def main():

	parser = argparse.ArgumentParser(description='Generate_Data')
	parser.add_argument('--expname', type=str, default=None)
	parser.add_argument('--idealname', type=str, default='els_outputs')
	parser.add_argument('--dataset', type=str, default='mnist')
	parser.add_argument('--scoremoduletype', type=str, default='bbELS') # options: boundary broken ELS (bbELS), ELS, LS, and IS
	parser.add_argument('--conditional', action="store_true", default=False)
	parser.add_argument('--scalesfile', type=str, default=None) # file with scales (auto-detected from checkpoints if None)
	parser.add_argument('--scorebatchsize', type=int, default=256)
	parser.add_argument('--fill', action="store_true", default=False)
	parser.add_argument('--numiters', type=int, default=100)
	parser.add_argument('--nsteps', type=int, default=20)
	parser.add_argument('--nlabels', type=int, default=10)
	parser.add_argument('--force_overwrite', action="store_true", default=False)
	parser.add_argument('--cpu', action="store_true", default=False)
	parser.add_argument('--max_samples', type=int, default=100000)
	parser.add_argument('--shuffle', action="store_true", default=False)

	args = parser.parse_args()

	dataset, metadata = get_dataset(args.dataset, root='./data')
	in_channels = metadata['num_channels']
	image_size = metadata['image_size']

	if args.expname is None:
		expname = 'dataset_%s_option_%1d' %(metadata['name'], args.scoremoduletype)
		if args.conditional:
			expname += '_conditional'
	else:
		expname = args.expname

	# Load ideal score modules
	schedule = cosine_noise_schedule
	max_samples = args.max_samples
	
	if args.scoremoduletype == 'ELS':
		mod = LocalEquivScoreModule(dataset,
					batch_size=args.scorebatchsize,
					image_size=image_size,
					channels=in_channels,
					schedule=schedule,
					shuffle=args.shuffle,
					max_samples=max_samples)
	elif args.scoremoduletype == 'bbELS': # ELS with Borders 
		mod = LocalEquivBordersScoreModule(dataset,
					batch_size=args.scorebatchsize,
					image_size=image_size,
					channels=in_channels,
					schedule=schedule,
					max_samples=max_samples)
	elif args.scoremoduletype == 'LS': # LS
		mod = LocalScoreModule(dataset,
					image_size=image_size,
					batch_size=len(dataset),
					show_plots=False,
					schedule=schedule)
	elif args.scoremoduletype == 'IS': # Ideal score
		mod = IdealScoreModule(dataset,
							image_size=image_size,
							batch_size=len(dataset),
							schedule=schedule)
	else:
		raise ValueError(f"Unknown scoremoduletype: {args.scoremoduletype}")

	# Auto-detect scales file if not provided
	if args.scalesfile is None:
		dataset_name_upper = metadata['name'].upper()
		possible_scales_files = [
			f'./checkpoints/scales_{dataset_name_upper}_ResNet_zeros_conditional.pt',
			f'./checkpoints/scales_{dataset_name_upper}_ResNet_zeros.pt',
			f'./checkpoints/scales_{dataset_name_upper}_UNet_zeros_conditional.pt',
			f'./checkpoints/scales_{dataset_name_upper}_UNet_zeros.pt'
		]
		
		scales_file = None
		for candidate in possible_scales_files:
			if os.path.exists(candidate):
				scales_file = candidate
				print(f"Auto-detected scales file: {scales_file}")
				break
		
		if scales_file is None:
			raise FileNotFoundError(f"No scales file found. Please specify --scalesfile or ensure scales exist in ./checkpoints/")
		args.scalesfile = scales_file
	
	scales_data = torch.load(args.scalesfile, weights_only=False)
	# Handle both tensor and list formats
	if isinstance(scales_data, torch.Tensor):
		scales = list(scales_data.int().numpy())
		scales = [int(s) for s in scales]
	elif isinstance(scales_data, list):
		scales = [int(s) for s in scales_data]
	else:
		scales = scales_data

	machine = ScheduledScoreMachine(mod, in_channels=in_channels, noise_schedule=schedule, score_backbone=True, scales=scales)


	DPATH = os.path.join('./results', expname)
	SEEDPATH = os.path.join(DPATH, 'seeds')
	SPATH = os.path.join(DPATH, args.idealname)
	if args.conditional:
		LPATH = os.path.join(DPATH, 'labels')

	if args.cpu:
		device = torch.device('cpu')
	else:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	machine.to(device)

	if args.fill: # Assumes seeds have already been generated, e.g. if this script
					# has been run before with a different scoremoduletype
		
		if not os.path.isdir(DPATH) or not os.path.isdir(SEEDPATH):
			raise FileNotFoundError(f"Required directories not found: {DPATH} or {SEEDPATH}")

		if not os.path.isdir(SPATH):
			os.makedirs(SPATH)

		i = 0
		while os.path.exists(os.path.join(SEEDPATH, f'{i:04d}.pt')):
			seed = torch.load(os.path.join(SEEDPATH, f'{i:04d}.pt'), weights_only=False)
			if args.conditional:
				label = torch.load(os.path.join(LPATH, f'{i:04d}.pt'), weights_only=False)
			else:
				label = None

			if not os.path.exists(os.path.join(SPATH, f'{i:04d}.pt')):
				score_output = machine(seed.clone(), label=label, device=device)
				torch.save(score_output, os.path.join(SPATH, f'{i:04d}.pt'))
			
			i += 1

	else:

		min_iter = 0
		if os.path.isdir(DPATH) and not args.force_overwrite:
			for i in range(args.numiters):
				check = os.path.exists(os.path.join(SEEDPATH, f'{i:04d}.pt'))
				check = check and os.path.exists(os.path.join(SPATH, f'{i:04d}.pt'))

				if check:
					continue
				else:
					min_iter = i
					break
		else:
			if os.path.isdir(DPATH):
				shutil.rmtree(DPATH)

			os.makedirs(DPATH)
			os.makedirs(SEEDPATH)
			os.makedirs(SPATH)
			if args.conditional:
				os.makedirs(LPATH)

		for i in range(min_iter, args.numiters):
			seed = torch.randn(1,in_channels,image_size,image_size,device=device)
			if args.conditional:
				label = torch.randint(0,args.nlabels,(1,))
			else:
				label = None
			
			score_output = machine(seed.clone(), label=label, device=device)

			torch.save(seed, os.path.join(SEEDPATH, f'{i:04d}.pt'))
			torch.save(score_output, os.path.join(SPATH, f'{i:04d}.pt'))

			if args.conditional:
				torch.save(label, os.path.join(LPATH, f'{i:04d}.pt'))



if __name__ == "__main__":
	main()