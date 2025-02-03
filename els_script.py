import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import argparse

from utils.noise_schedules import cosine_noise_schedule
from utils.data import get_dataset
from utils.idealscore import ScheduledScoreMachine, LocalEquivBordersScoreModule, LocalEquivScoreModule, LocalScoreModule, IdealScoreModule
from models import DDIM, MinimalUNet, MinimalResNet
import shutil

import sys
import os

## GENERATES ELS MACHINE OUTPUTS

def main():

	parser = argparse.ArgumentParser(description='Generate_Data')
	parser.add_argument('--expname', type=str, default=None)
	parser.add_argument('--idealname', type=str, default='els_outputs')
	parser.add_argument('--dataset', type=str, default='mnist')
	parser.add_argument('--scoremoduletype', type=str, default='bbELS') # options: boundary broken ELS (bbELS), ELS, LS, and IS
	parser.add_argument('--conditional', action="store_true", default=False)
	parser.add_argument('--scalesfile', type=str, default='scales') # file with scales
	parser.add_argument('--scorebatchsize', type=int, default=256)
	parser.add_argument('--fill', action="store_true", default=False)
	parser.add_argument('--numiters', type=int, default=100)
	parser.add_argument('--nsteps', type=int, default=20)
	parser.add_argument('--nlabels', type=int, default=10)
	parser.add_argument('--force_overwrite', action="store_true", default=False)
	parser.add_argument('--cpu', action="store_true", default=False)
	parser.add_argument('--reduce_size', action="store_true", default=False)
	parser.add_argument('--max_samples', type=int, default=1000)
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
	max_samples = args.max_samples if args.reduce_size else None
	
	if args.scoremoduletype == 'ELS':
		mod = LocalEquivScoreModule(1, dataset,
					batch_size=args.scorebatchsize,
					image_size=image_size,
					channels=in_channels,
					mode='circular',
					schedule=schedule,
					shuffle=args.shuffle,
					max_samples=max_samples)
	elif args.scoremoduletype == 'bbELS': # ELS with Borders 
		mod = LocalEquivBordersScoreModule(1, dataset,
					batch_size=args.scorebatchsize,
					image_size=image_size,
					channels=in_channels,
					schedule=schedule,
					max_samples=max_samples)
	elif args.scoremoduletype == 'LS': # LS
		mod = LocalScoreModule(1, dataset,
					image_size=image_size,
					batch_size=len(dataset),
					mode='zeros',
					show_plots=False,
					schedule=schedule)
	elif args.scoremoduletype == 'IS': # Ideal score
		mod = IdealScoreModule(dataset,
							image_size=image_size,
							batch_size=len(dataset),
							schedule=schedule)
	else:
		raise()

	scales = list(torch.load(args.sfile + '.pt').int().numpy())
	scales = [int(s) for s in scales]

	machine = ScheduledScoreMachine(mod, in_channels=in_channels, noise_schedule=schedule, score_backbone=True, scales=scales)


	DPATH = './experiments/' + expname
	SEEDPATH = DPATH + '/seeds/'
	SPATH = DPATH + '/' + args.idealname + '/'
	if args.conditional:
		LPATH = DPATH + '/labels/'

	if args.cpu:
		device = torch.device('cpu')
	else:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	machine.to(device)

	if args.fill: # Assumes seeds have already been generated, e.g. if this script
					# has been run before with a different scoremoduletype
		
		if not os.path.isdir(DPATH) or not os.path.isdir(SEEDPATH):
			raise()

		if not os.path.isdir(SPATH):
			os.makedirs(SPATH)

		i = 0
		while os.path.exists(SEEDPATH + '%04d.pt' % (i)):
			seed = torch.load(SEEDPATH + '%04d.pt' % (i))
			if args.conditional:
				label = torch.load(LPATH + '%04d.pt' % (i))
			else:
				label = None

			if not os.path.exists(SPATH + '%04d.pt' %(i)):
				score_output = machine(seed.clone(), label=label, device=device)
				torch.save(score_output, SPATH + '%04d.pt' %(i))
			
			i += 1

	else:

		min_iter = 0
		if os.path.isdir(DPATH) and not args.force_overwrite:
			for i in range(args.numiters):
				check = os.path.exists(SEEDPATH + '%04d.pt' %(i))
				check = check and os.path.exists(SPATH + '%04d.pt' %(i))

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

			torch.save(seed, SEEDPATH + '%04d.pt' % (i))
			torch.save(score_output, SPATH + '%04d.pt' % (i))

			if args.conditional:
				torch.save(label, LPATH + '%04d.pt' % (i))



if __name__ == "__main__":
	main()