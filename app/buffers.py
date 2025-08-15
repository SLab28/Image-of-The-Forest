from __future__ import annotations

from typing import Tuple
import numpy as np


class RingBuffer:
	"""Fixed-size ring buffer for multi-channel data with timestamps."""

	def __init__(self, num_channels: int, capacity: int) -> None:
		self.num_channels = int(num_channels)
		self.capacity = int(capacity)
		self.data = np.zeros((self.num_channels, self.capacity), dtype=np.float64)
		self.times = np.zeros((self.capacity,), dtype=np.float64)
		self.write_idx = 0
		self.filled = False

	def append(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
		"""Append samples shape (num_channels, n) and timestamps shape (n,)."""
		if samples.size == 0:
			return
		assert samples.shape[0] == self.num_channels
		n = samples.shape[1]
		assert timestamps.shape[0] == n

		end = self.write_idx + n
		if end <= self.capacity:
			self.data[:, self.write_idx:end] = samples
			self.times[self.write_idx:end] = timestamps
			self.write_idx = end % self.capacity
			if end == self.capacity:
				self.filled = True
		else:
			first = self.capacity - self.write_idx
			self.data[:, self.write_idx:] = samples[:, :first]
			self.times[self.write_idx:] = timestamps[:first]
			remaining = n - first
			self.data[:, :remaining] = samples[:, first:]
			self.times[:remaining] = timestamps[first:]
			self.write_idx = remaining
			self.filled = True

	def get_last(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
		"""Return (times, data) for the last num_samples, clipped to available."""
		if not self.filled and self.write_idx == 0:
			return np.array([]), np.zeros((self.num_channels, 0))

		available = self.capacity if self.filled else self.write_idx
		n = max(0, min(num_samples, available))
		end = self.write_idx
		start = (end - n) % self.capacity
		if start < end:
			times = self.times[start:end]
			data = self.data[:, start:end]
		else:
			times = np.concatenate([self.times[start:], self.times[:end]])
			data = np.concatenate([self.data[:, start:], self.data[:, :end]], axis=1)
		return times, data

	def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
		if self.filled:
			times = np.concatenate([self.times[self.write_idx:], self.times[:self.write_idx]])
			data = np.concatenate([self.data[:, self.write_idx:], self.data[:, :self.write_idx]], axis=1)
			return times, data
		else:
			return self.times[: self.write_idx], self.data[:, : self.write_idx]