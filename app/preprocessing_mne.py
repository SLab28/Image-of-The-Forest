from __future__ import annotations

from typing import Tuple
import numpy as np

try:
	from mne.filter import filter_data, notch_filter  # type: ignore
	from mne.time_frequency import psd_array_welch  # type: ignore
except Exception:  # pragma: no cover
	filter_data = None
	notch_filter = None
	psd_array_welch = None


def _notch_then_bandpass(data: np.ndarray, fs: float, line_freq: int, l_freq: float, h_freq: float, axis: int = 1) -> np.ndarray:
	out = np.ascontiguousarray(data)
	try:
		if notch_filter is not None and line_freq in (50, 60):
			freqs = [line_freq, 2 * line_freq]
			out = notch_filter(out, Fs=fs, freqs=freqs, method='spectrum_fit', axis=axis, verbose='ERROR')
		if filter_data is not None:
			out = filter_data(out, sfreq=fs, l_freq=l_freq, h_freq=h_freq, method='fir', phase='zero-double', axis=axis, verbose='ERROR')
		return out
	except Exception:
		return data


def filter_eeg_mne(eeg: np.ndarray, fs: float, line_freq: int = 50, l_freq: float = 1.0, h_freq: float = 45.0, car: bool = True) -> np.ndarray:
	"""Filter EEG: notch at line + harmonic, band-pass 1–45 Hz, optional common-average reference.
	Input shape: (channels, samples).
	"""
	if eeg.size == 0:
		return eeg
	out = _notch_then_bandpass(eeg, fs=fs, line_freq=line_freq, l_freq=l_freq, h_freq=h_freq, axis=1)
	if car:
		try:
			out = out - np.mean(out, axis=0, keepdims=True)
		except Exception:
			pass
	return out


def filter_ppg_mne(ppg: np.ndarray, fs: float, line_freq: int = 50, l_freq: float = 0.5, h_freq: float = 5.0) -> np.ndarray:
	"""Filter PPG: notch at line + harmonic, band-pass 0.5–5 Hz. Accepts 1-D or (1, n)."""
	arr = np.asarray(ppg, dtype=float)
	if arr.ndim == 1:
		arr = arr[np.newaxis, :]
	out = _notch_then_bandpass(arr, fs=fs, line_freq=line_freq, l_freq=l_freq, h_freq=h_freq, axis=1)
	return out[0] if ppg.ndim == 1 else out


def psd_welch_mne(data: np.ndarray, fs: float, fmin: float = 1.0, fmax: float = 40.0) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute Welch PSD via MNE with sensible defaults; data shape (channels, samples)."""
	arr = np.ascontiguousarray(data)
	try:
		if psd_array_welch is None:
			raise RuntimeError("MNE not available")
		n_per_seg = min(1024, arr.shape[1])
		psd, freqs = psd_array_welch(arr, sfreq=fs, fmin=fmin, fmax=fmax, n_fft=n_per_seg, n_overlap=n_per_seg // 2, average='mean', verbose='ERROR')
		return psd, freqs
	except Exception:
		# Fallback to numpy/scipy if needed
		from scipy.signal import welch  # type: ignore
		nperseg = min(1024, arr.shape[1])
		overlap = nperseg // 2
		freqs, psd = welch(arr, fs=fs, nperseg=nperseg, noverlap=overlap, axis=1)
		# Limit to [fmin, fmax]
		mask = (freqs >= fmin) & (freqs <= fmax)
		return psd[:, mask], freqs[mask]