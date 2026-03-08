import argparse
import json
import os

import numpy as np


def default_amplitude_from_measure(image_size, alpha, num_points=4096):
	"""Set A so integral dmu(k) * A * k^{-alpha} = 1 with normalized dmu."""
	ky = np.fft.fftfreq(image_size, d=1.0)[:, None]
	kx = np.fft.rfftfreq(image_size, d=1.0)[None, :]
	k = np.sqrt(kx**2 + ky**2)
	k_nonzero = k[k > 0]
	lambda_ir = float(np.min(k_nonzero))
	lambda_uv = float(np.max(k_nonzero))
	norm = 0.5 * (lambda_uv**2 - lambda_ir**2)
	k_grid = np.linspace(lambda_ir, lambda_uv, num_points, dtype=np.float64)
	integrand = (k_grid / norm) * np.power(k_grid, -alpha)
	ik = np.trapz(integrand, k_grid)
	return float(1.0 / ik)


def generate_chunk(n, channels, image_size, alpha, amplitude, rng):
	ky = np.fft.fftfreq(image_size, d=1.0)[:, None]
	kx = np.fft.rfftfreq(image_size, d=1.0)[None, :]
	k = np.sqrt(kx**2 + ky**2)
	power = amplitude * np.where(k > 0, np.power(k, -alpha), 0.0)
	# Interior: scale = sqrt(power/2). Only x-edges (kx=0, kx=N/2) need double variance; y-edges do not (rfft stores all ky but only kx>=0).
	scale = np.sqrt(np.maximum(power, 0.0) / 2.0).astype(np.float32)
	iy = np.arange(image_size)
	ix = np.arange(image_size // 2 + 1)
	col_edge = (ix == 0) | (ix == image_size // 2)
	scale = np.where(col_edge[None, :], np.sqrt(np.maximum(power, 0.0)).astype(np.float32), scale)
	row_edge = (iy == 0) | (iy == image_size // 2)
	corner_mode = row_edge[:, None] & col_edge[None, :]

	coeff_real = rng.standard_normal(
		size=(n, channels, image_size, image_size // 2 + 1), dtype=np.float32
	)
	coeff_imag = rng.standard_normal(
		size=(n, channels, image_size, image_size // 2 + 1), dtype=np.float32
	)
	coeff_imag = np.where(corner_mode[None, None, :, :], 0.0, coeff_imag)
	coeff = (coeff_real + 1j * coeff_imag) * scale[None, None, :, :]
	coeff[:, :, 0, 0] = 0.0  # remove DC mode

	x = np.fft.irfft2(coeff, s=(image_size, image_size), axes=(-2, -1), norm='ortho')
	x = x.real.astype(np.float32)
	x = x - np.mean(x, axis=(-2, -1), keepdims=True)
	return x


def write_split_memmap(path, n_samples, channels, image_size, alpha, amplitude, seed, chunk_size):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	data = np.lib.format.open_memmap(
		path,
		mode='w+',
		dtype=np.float32,
		shape=(n_samples, channels, image_size, image_size),
	)
	rng = np.random.default_rng(seed)
	for start in range(0, n_samples, chunk_size):
		end = min(start + chunk_size, n_samples)
		chunk = generate_chunk(
			n=end - start,
			channels=channels,
			image_size=image_size,
			alpha=alpha,
			amplitude=amplitude,
			rng=rng,
		)
		data[start:end] = chunk
		if (start // chunk_size) % 10 == 0:
			print(f'  wrote {end}/{n_samples} to {path}')
	data.flush()


def main():
	parser = argparse.ArgumentParser(description='Generate toy translationally invariant Gaussian field dataset')
	parser.add_argument('--root', type=str, default='./data')
	parser.add_argument('--dirname', type=str, default='toy_gaussian_field')
	parser.add_argument('--train_samples', type=int, default=200000)
	parser.add_argument('--valid_samples', type=int, default=10000)
	parser.add_argument('--image_size', type=int, default=32)
	parser.add_argument('--channels', type=int, default=3)
	parser.add_argument('--alpha', type=float, default=3.0)
	parser.add_argument('--amplitude', type=float, default=None)
	parser.add_argument('--chunk_size', type=int, default=2048)
	parser.add_argument('--seed', type=int, default=1234)
	args = parser.parse_args()

	base_dir = os.path.join(args.root, args.dirname)
	os.makedirs(base_dir, exist_ok=True)

	amplitude = args.amplitude
	if amplitude is None:
		amplitude = default_amplitude_from_measure(args.image_size, args.alpha)
	print(f'Using amplitude A={amplitude:.8f}')

	train_path = os.path.join(base_dir, 'train_data.npy')
	valid_path = os.path.join(base_dir, 'valid_data.npy')

	print('Generating train split...')
	write_split_memmap(
		path=train_path,
		n_samples=args.train_samples,
		channels=args.channels,
		image_size=args.image_size,
		alpha=args.alpha,
		amplitude=amplitude,
		seed=args.seed,
		chunk_size=args.chunk_size,
	)

	print('Generating valid split...')
	write_split_memmap(
		path=valid_path,
		n_samples=args.valid_samples,
		channels=args.channels,
		image_size=args.image_size,
		alpha=args.alpha,
		amplitude=amplitude,
		seed=args.seed + 1,
		chunk_size=args.chunk_size,
	)

	metadata = {
		'name': 'toy_gaussian_field',
		'n_train': int(args.train_samples),
		'n_valid': int(args.valid_samples),
		'image_size': int(args.image_size),
		'num_channels': int(args.channels),
		'alpha': float(args.alpha),
		'amplitude': float(amplitude),
		'measure_normalization': 'integral dmu(k) = 1',
		'amplitude_constraint': 'integral dmu(k) * A * |k|^{-alpha} = 1',
		'train_file': 'train_data.npy',
		'valid_file': 'valid_data.npy',
		'mean': [0.0 for _ in range(args.channels)],
		'std': [1.0 for _ in range(args.channels)],
	}
	with open(os.path.join(base_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
		json.dump(metadata, f, indent=2)

	print(f'Dataset written to: {base_dir}')
	print('Done.')


if __name__ == '__main__':
	main()
