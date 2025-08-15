#!/usr/bin/env python3

import os
import sys
from typing import Optional, List, Dict, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtWidgets import (
	QApplication,
	QMainWindow,
	QWidget,
	QVBoxLayout,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QComboBox,
	QStatusBar,
)

import numpy as np
import pyqtgraph as pg
from pylsl import StreamInlet, resolve_byprop

# Support running as `python -m app.main` (preferred) or `python app/main.py`
try:
	from app.muse_control import list_devices as muse_list_devices, start_stream as muse_start_stream, stop_stream as muse_stop_stream, muselsl_available
except ModuleNotFoundError:
	# Add project root (parent of `app/`) to sys.path so `app` is importable
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
	from app.muse_control import list_devices as muse_list_devices, start_stream as muse_start_stream, stop_stream as muse_stop_stream, muselsl_available

APP_TITLE = "Muse Viewer (Minimal)"
FS_EEG_FALLBACK = 256.0


def load_inter_font() -> Optional[str]:
	assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
	for name in ["Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf"]:
		path = os.path.join(assets_dir, name)
		if os.path.exists(path):
			fid = QFontDatabase.addApplicationFont(path)
			if fid != -1:
				families = QFontDatabase.applicationFontFamilies(fid)
				if families:
					return families[0]
	return None


class MainWindow(QMainWindow):
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle(APP_TITLE)
		self.resize(1200, 700)
		self.setStyleSheet(
			"""
			QWidget { background-color: #0F0F10; color: #FFFFFF; }
			QLabel { color: #FFFFFF; }
			QPushButton { background-color: #1A1A1C; color: #FFFFFF; border: 1px solid #2A2A2E; padding:8px 12px; border-radius:6px; }
			QPushButton:hover { background-color: #242428; }
			QComboBox { background-color: #1A1A1C; color: #FFFFFF; border: 1px solid #2A2A2E; padding:6px 8px; border-radius:6px; }
			QStatusBar { background-color: #0F0F10; color: #A0A0A5; }
			"""
		)
		ff = load_inter_font()
		if ff:
			self.setFont(QFont(ff, 11))

		self.status = QStatusBar(); self.setStatusBar(self.status)

		# Controls
		self.device_combo = QComboBox()
		self.refresh_btn = QPushButton("Find Muse")
		self.start_btn = QPushButton("Start Stream")
		self.connect_btn = QPushButton("Connect LSL")
		self.disconnect_btn = QPushButton("Disconnect")
		self.stop_btn = QPushButton("Stop Stream")
		self.stop_btn.setEnabled(False)
		self.disconnect_btn.setEnabled(False)

		self.refresh_btn.clicked.connect(self.on_refresh)
		self.start_btn.clicked.connect(self.on_start_stream)
		self.connect_btn.clicked.connect(self.on_connect)
		self.disconnect_btn.clicked.connect(self.on_disconnect)
		self.stop_btn.clicked.connect(self.on_stop_stream)

		# Plot
		pg.setConfigOption("foreground", "w")
		pg.setConfigOption("background", (15, 15, 16))
		self.plot = pg.PlotWidget(title="EEG Time Series")
		self.plot.showGrid(x=True, y=True, alpha=0.3)
		self.curves: List[pg.PlotDataItem] = []
		self.info_label = QLabel("Idle. Click Find Muse.")
		self.info_label.setStyleSheet("QLabel { font-family: monospace; }")

		# Layout
		bar = QHBoxLayout()
		bar.addWidget(QLabel("Device/Stream:"))
		bar.addWidget(self.device_combo, 2)
		bar.addWidget(self.refresh_btn)
		bar.addWidget(self.start_btn)
		bar.addWidget(self.connect_btn)
		bar.addWidget(self.disconnect_btn)
		bar.addWidget(self.stop_btn)

		root = QVBoxLayout()
		root.addLayout(bar)
		root.addWidget(self.plot, 1)
		root.addWidget(self.info_label)

		w = QWidget(); w.setLayout(root)
		self.setCentralWidget(w)

		# State
		self.devices: List[Dict[str, str]] = []
		self.inlet: Optional[StreamInlet] = None
		self.fs = FS_EEG_FALLBACK
		self.timer = QTimer(self)
		self.timer.setInterval(50)
		self.timer.timeout.connect(self.on_timer)
		self.muse_proc = None
		self.timestamps: List[float] = []
		self.buffer: Optional[np.ndarray] = None  # shape (ch, n)

	def on_refresh(self) -> None:
		self.status.showMessage("Scanning for Muse via muselsl (fallback: LSL streams)...", 3000)
		self.device_combo.clear(); self.devices = []
		if muselsl_available():
			try:
				self.devices = muse_list_devices(timeout=8)
			except Exception:
				self.devices = []
		if self.devices:
			for d in self.devices:
				self.device_combo.addItem(f"{d.get('name','Muse')} ({d.get('address','?')})")
		else:
			try:
				streams = resolve_byprop("type", "EEG", timeout=3)
				for s in streams:
					self.device_combo.addItem(s.name())
				if not streams:
					self.device_combo.addItem("No EEG streams found")
			except Exception as exc:
				self.device_combo.addItem(f"Error: {exc}")

	def on_start_stream(self) -> None:
		if not self.devices:
			self.status.showMessage("No BLE devices listed; try Connect LSL if you already have a stream.", 4000)
			return
		idx = self.device_combo.currentIndex()
		if idx < 0 or idx >= len(self.devices):
			self.status.showMessage("Select a device first.", 3000)
			return
		addr = self.devices[idx].get("address")
		if not addr:
			self.status.showMessage("Selected device has no address.", 3000)
			return
		self.muse_proc = muse_start_stream(name_or_address=addr, acc=False, gyro=False, ppg=False)
		if self.muse_proc is None:
			self.status.showMessage("muselsl not available.", 4000)
			return
		self.start_btn.setEnabled(False)
		self.stop_btn.setEnabled(True)
		self.status.showMessage("Starting stream... connecting in 3s")
		QTimer.singleShot(3000, self.on_connect)

	def on_stop_stream(self) -> None:
		muse_stop_stream(self.muse_proc)
		self.muse_proc = None
		self.start_btn.setEnabled(True)
		self.stop_btn.setEnabled(False)
		self.status.showMessage("Stopped stream", 2000)

	def on_connect(self) -> None:
		try:
			arr = resolve_byprop("type", "EEG", timeout=3)
			if not arr:
				raise RuntimeError("No EEG LSL stream found")
			self.inlet = StreamInlet(arr[0])
			info = self.inlet.info()
			nch = max(1, info.channel_count())
			self.fs = info.nominal_srate() or FS_EEG_FALLBACK
			self.curves = [self.plot.plot(pen=pg.mkPen(c)) for c in [(255,255,255),(255,160,0),(0,200,255),(160,160,255),(0,255,160),(255,80,120)][:nch]]
			self.buffer = np.zeros((nch, int(self.fs * 5)), dtype=float)
			self.timestamps = []
			self.connect_btn.setEnabled(False)
			self.disconnect_btn.setEnabled(True)
			self.status.showMessage("Connected to EEG LSL")
			self.timer.start()
		except Exception as exc:
			self.status.showMessage(f"Connect failed: {exc}", 5000)

	def on_disconnect(self) -> None:
		self.timer.stop()
		self.inlet = None
		self.curves = []
		self.buffer = None
		self.timestamps = []
		self.connect_btn.setEnabled(True)
		self.disconnect_btn.setEnabled(False)
		self.status.showMessage("Disconnected", 2000)

	def on_timer(self) -> None:
		if self.inlet is None or self.buffer is None:
			return
		try:
			samples, ts = self.inlet.pull_chunk(max_samples=int(self.fs // 2), timeout=0.0)
			if not ts:
				return
			arr = np.asarray(samples, dtype=float).T  # (ch, n)
			nch, n = arr.shape
			# Shift buffer and append
			b = self.buffer
			if n >= b.shape[1]:
				self.buffer = arr[:, -b.shape[1]:]
			else:
				self.buffer[:, :-n] = b[:, n:]
				self.buffer[:, -n:] = arr
			# X axis
			t0 = ts[-1] - (self.buffer.shape[1] - 1) / self.fs
			x = t0 + np.arange(self.buffer.shape[1]) / self.fs
			for i, curve in enumerate(self.curves):
				if i < self.buffer.shape[0]:
					curve.setData(x, self.buffer[i])
			self.info_label.setText(f"t={ts[-1]:.3f} fs={self.fs:.1f} nch={self.buffer.shape[0]}")
		except Exception as exc:
			self.status.showMessage(f"Stream error: {exc}", 2000)


def main() -> int:
	app = QApplication(sys.argv)
	ff = load_inter_font()
	if ff:
		app.setFont(QFont(ff, 11))
	win = MainWindow(); win.show()
	return app.exec()


if __name__ == "__main__":
	sys.exit(main())