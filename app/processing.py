from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import welch


# State-of-the-art inspired canonical EEG bands (can be adapted per age/task)
BANDS: Dict[str, Tuple[float, float]] = {
	"delta": (1.0, 4.0),
	"theta": (4.0, 8.0),
	"alpha": (8.0, 13.0),
	"beta": (13.0, 30.0),
	"gamma": (30.0, 45.0),
}


@dataclass
class PsdResult:
	freqs: np.ndarray
	psd: np.ndarray  # shape: (channels, freqs)


def compute_welch_psd(data: np.ndarray, fs: float) -> PsdResult:
	"""Compute PSD via Welch's method for multi-channel data.

	data: shape (channels, samples)
	fs: sampling rate
	"""
	nperseg = min(1024, data.shape[1])
	overlap = nperseg // 2
	freqs, psd = welch(data, fs=fs, nperseg=nperseg, noverlap=overlap, axis=1)
	return PsdResult(freqs=freqs, psd=psd)


def band_power(psd: PsdResult, band: Tuple[float, float]) -> np.ndarray:
	fmin, fmax = band
	idx = np.logical_and(psd.freqs >= fmin, psd.freqs <= fmax)
	# Integrate power in band
	return np.trapz(psd.psd[:, idx], psd.freqs[idx], axis=1)


def compute_band_powers(psd: PsdResult) -> Dict[str, np.ndarray]:
	return {name: band_power(psd, rng) for name, rng in BANDS.items()}


def simple_contact_quality(eeg: np.ndarray) -> np.ndarray:
	"""Rudimentary contact quality proxy using high-frequency noise proportion.
	Lower values imply better contact. Not a substitute for impedance.
	Input shape: (channels, samples)
	"""
	if eeg.shape[1] < 256:
		return np.ones(eeg.shape[0])
	fs = 256.0
	psd = compute_welch_psd(eeg, fs)
	total = np.trapz(psd.psd, psd.freqs, axis=1)
	hf = band_power(psd, (40.0, 80.0))
	with np.errstate(divide='ignore', invalid='ignore'):
		ratio = np.where(total > 0, hf / total, 1.0)
	return ratio


def interpolate_topomap(channel_values: np.ndarray, layout: List[Tuple[float, float]], grid_size: int = 64) -> np.ndarray:
	"""Nearest-neighbor interpolation onto a 2D grid for simple topography.
	layout: list of (x,y) in head coords [-1,1]. For Muse: four channels typical.
	Returns image grid (grid_size x grid_size).
	"""
	x = np.linspace(-1, 1, grid_size)
	y = np.linspace(-1, 1, grid_size)
	gx, gy = np.meshgrid(x, y)

	# Compute distance to each electrode and take value of nearest
	img = np.zeros_like(gx)
	coords = np.array(layout)
	for i in range(grid_size):
		for j in range(grid_size):
			pt = np.array([gx[i, j], gy[i, j]])
			d = np.linalg.norm(coords - pt, axis=1)
			k = int(np.argmin(d))
			img[i, j] = channel_values[k]
	return img


def estimate_heart_rate_ppg(ppg: np.ndarray, fs: float) -> Tuple[float, float]:
	"""Estimate heart rate (bpm) and confidence from a 1-D PPG segment.
	Uses Welch PSD peak in 0.8–3.0 Hz (~48–180 bpm).
	Returns (bpm, confidence in [0,1]).
	"""
	if ppg.ndim != 1 or ppg.size < max(int(fs * 5), 256):
		return float('nan'), 0.0
	# Remove DC
	ppg = ppg - np.nanmean(ppg)
	# Welch PSD
	freqs, pxx = welch(ppg, fs=fs, nperseg=min(1024, ppg.size), noverlap=None)
	# Limit to physiologic band
	mask = (freqs >= 0.8) & (freqs <= 3.0)
	if not np.any(mask):
		return float('nan'), 0.0
	fband = freqs[mask]
	pband = pxx[mask]
	if pband.size == 0 or np.all(pband <= 0):
		return float('nan'), 0.0
	peak_idx = int(np.argmax(pband))
	peak_freq = float(fband[peak_idx])
	bpm = peak_freq * 60.0
	# Confidence as peak prominence ratio
	power_total = float(np.trapz(pband, fband))
	peak_power = float(pband[peak_idx])
	conf = max(0.0, min(1.0, peak_power / (power_total + 1e-12)))
	return bpm, conf