from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np

try:
	import mne
	from mne.filter import filter_data, notch_filter
	from mne.time_frequency import psd_array_welch
	_MNE_AVAILABLE = True
except Exception:
	_MNE_AVAILABLE = False


def _ensure_mne() -> None:
	if not _MNE_AVAILABLE:
		raise RuntimeError("MNE is required for preprocessing. Please install 'mne'.")


def filter_eeg_mne(eeg: np.ndarray, fs: float, line_freq: int = 50) -> np.ndarray:
	"""Band-pass EEG to 1–40 Hz and notch at the given line frequency (and harmonics).
	Input shape: (channels, samples). Returns same shape.
	"""
	_ensure_mne()
	if eeg.size == 0:
		return eeg
	data = eeg.copy()
	# Notch line noise and harmonics up to 240 Hz (overkill for Muse but OK)
	if line_freq in (50, 60):
		freqs = np.arange(line_freq, min(241, int(fs // 2) * 2), line_freq)
		if freqs.size > 0:
			data = notch_filter(data, sfreq=fs, freqs=freqs, method='iir')
	# Band-pass
	data = filter_data(data, sfreq=fs, l_freq=1.0, h_freq=40.0, method='iir', iir_params=dict(order=4, ftype='butter'))
	return data


def filter_ppg_mne(ppg: np.ndarray, fs: float) -> np.ndarray:
	"""Band-pass PPG to 0.5–5 Hz for HR estimation. Input shape: (samples,) or (1, samples)."""
	_ensure_mne()
	arr = ppg
	if ppg.ndim == 1:
		arr = ppg[np.newaxis, :]
	arr = filter_data(arr, sfreq=fs, l_freq=0.5, h_freq=5.0, method='iir', iir_params=dict(order=4, ftype='butter'))
	return arr[0] if ppg.ndim == 1 else arr


def psd_welch_mne(data: np.ndarray, fs: float, fmin: float = 1.0, fmax: float = 40.0) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute PSD via MNE Welch. Input shape: (channels, samples). Returns (psd, freqs)."""
	_ensure_mne()
	psd, freqs = psd_array_welch(data, sfreq=fs, fmin=fmin, fmax=fmax, n_fft=min(1024, data.shape[1]), n_overlap=None, average='mean')
	return psd, freqs