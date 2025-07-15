import argparse
import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.data import get_dataset
from src.utils.idealscore import denormalize
from src.utils.noise_schedules import cosine_noise_schedule


def main():
	# Accepts a directory formatted according to outputs of els_script.py,
	# with one subdirectory for the outputs of the ELS machine and one for the
	# outputs of the IS machine

	parser = argparse.ArgumentParser(description='Evaluation Script')
	parser.add_argument('--exp_fname', type=str, default='./') # top level directory
	parser.add_argument('--model_fname', type=str, default='./model_checkpoints/test.pt') # file name of model
	parser.add_argument('--outputname', type=str, default='els_outputs/') # subdirectory with ELS outputs
	parser.add_argument('--dsname', type=str, default='cifar10') # name of dataset
	parser.add_argument('--conditional', action="store_true", default=False)
	parser.add_argument('--title', type=str, default="")
	parser.add_argument('--figname', type=str, default='corrs.png')

	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = torch.load(args.model_fname, map_location=device)
	model.noise_schedule = cosine_noise_schedule
	model.eval()

	_, metadata = get_dataset(args.dsname)

	SEEDPATH = os.path.join(args.exp_fname, 'seeds') # seeds directory
	OUTPATH = os.path.join(args.exp_fname, args.outputname) # ELS output directory
	LPATH = os.path.join(args.exp_fname, 'labels') # labels (if conditional)
	IPATH = os.path.join(args.exp_fname, 'ideal') # ideal score outputs for comparison


	ideal_corrs = []
	target_corrs = []

	n = 0
	while os.path.exists(os.path.join(SEEDPATH, f'{n:04d}.pt')):
		seed = torch.load(os.path.join(SEEDPATH, f'{n:04d}.pt'), map_location=device)
		
		if args.conditional:
			label = torch.load(os.path.join(LPATH, f'{n:04d}.pt'), map_location=device)
		
		output = model.sample(x=seed.clone(), nsteps=20, label=torch.tensor(label) if args.conditional else None)
		norm_output = output - torch.mean(output)
		norm_output = norm_output / torch.norm(norm_output)

		theoretical = torch.load(os.path.join(OUTPATH, f'{n:04d}.pt'), map_location=device)
		norm_theoretical = theoretical - torch.mean(theoretical)
		norm_theoretical = norm_theoretical / torch.norm(norm_theoretical)

		ideal = torch.load(os.path.join(IPATH, f'{n:04d}.pt'), map_location=device)
		norm_ideal = ideal - torch.mean(ideal)
		norm_ideal = norm_ideal / torch.norm(norm_ideal)

		ideal_corrs.append(torch.sum(norm_ideal*norm_output).detach().item())
		target_corrs.append(torch.sum(norm_theoretical*norm_output).detach().item())

		n += 1

	print(np.median(ideal_corrs))
	print(np.median(target_corrs))
	print(np.sum([1.0*(target_corrs[i]>ideal_corrs[i]) for i in range(len(ideal_corrs))])/len(ideal_corrs))

	fig, ax = plt.subplots()
	
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	ax.set_xlabel(r'$r^2$, IS Machine')
	ax.set_ylabel(r'$r^2$, ELS Machine')
	ax.scatter(ideal_corrs, target_corrs)
	ax.plot([0,1], [0,1], color='orange')
	ax.set_title(args.title)
	figname = 'scatter_' + args.figname

	fig.savefig(figname, bbox_inches='tight', pad_inches=0)
	plt.show()

if __name__ == "__main__":
	main()