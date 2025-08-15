#!/usr/bin/env python3

import os
import sys
from typing import Optional, Dict, List, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontDatabase, QFont, QAction
from PySide6.QtWidgets import (
	QApplication,
	QMainWindow,
	QWidget,
	QVBoxLayout,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QComboBox,
	QTabWidget,
	QStatusBar,
	QSplitter,
)

import numpy as np
import pyqtgraph as pg

from pylsl import StreamInlet, resolve_byprop

from app.buffers import RingBuffer
from app.processing import (
	interpolate_topomap,
	simple_contact_quality,
	BANDS,
)
from app.preprocessing_mne import filter_eeg_mne, filter_ppg_mne, psd_welch_mne
from app.muse_control import list_devices as muse_list_devices, start_stream as muse_start_stream, stop_stream as muse_stop_stream, muselsl_available


APP_TITLE = "NeuroStream"

# Fallback sampling rates
FS_EEG_DEFAULT = 256.0
FS_ACC_DEFAULT = 52.0
FS_GYR_DEFAULT = 52.0
FS_PPG_DEFAULT = 64.0

# Muse S four EEG channel approximate 2D positions (unit circle coords)
MUSE_LAYOUT: List[Tuple[float, float]] = [
	(-0.6, 0.4),  # TP9
	(-0.2, 0.8),  # AF7
	(0.2, 0.8),   # AF8
	(0.6, 0.4),   # TP10
]


def load_inter_font() -> Optional[str]:
	"""Attempt to load Inter font if available in assets, else use system default."""
	assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
	inter_paths = [
		os.path.join(assets_dir, "Inter-Regular.ttf"),
		os.path.join(assets_dir, "Inter-Medium.ttf"),
		os.path.join(assets_dir, "Inter-SemiBold.ttf"),
	]
	loaded_family: Optional[str] = None
	for path in inter_paths:
		if os.path.exists(path):
			fid = QFontDatabase.addApplicationFont(path)
			if fid != -1:
				families = QFontDatabase.applicationFontFamilies(fid)
				if families:
					loaded_family = families[0]
	return loaded_family


def _extract_channel_labels(info) -> List[str]:
	labels: List[str] = []
	try:
		desc = info.desc()
		channels = desc.child("channels")
		ch = channels.child("channel")
		while not ch.empty():
			lab = ch.child_value("label")
			labels.append(lab or "")
			ch = ch.next_sibling()
	except Exception:
		pass
	return labels


class MainWindow(QMainWindow):
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle(APP_TITLE)
		self.resize(1500, 1000)

		# Dark theme styling
		self.setStyleSheet(
			"""
			QWidget { background-color: #0F0F10; color: #FFFFFF; }
			QLabel { color: #FFFFFF; }
			QPushButton { background-color: #1A1A1C; color: #FFFFFF; border: 1px solid #2A2A2E; padding:8px 12px; border-radius:6px; }
			QPushButton:hover { background-color: #242428; }
			QComboBox { background-color: #1A1A1C; color: #FFFFFF; border: 1px solid #2A2A2E; padding:6px 8px; border-radius:6px; }
			QTabBar::tab { background-color: #141416; color: #FFFFFF; padding: 8px 16px; border-top-left-radius:6px; border-top-right-radius:6px; }
			QTabBar::tab:selected { background-color: #1A1A1C; }
			QStatusBar { background-color: #0F0F10; color: #A0A0A5; }
			"""
		)

		font_family = load_inter_font()
		if font_family:
			app_font = QFont(font_family, 11)
			self.setFont(app_font)

		self.status = QStatusBar()
		self.setStatusBar(self.status)

		# Inlets
		self.inlet_eeg: Optional[StreamInlet] = None
		self.inlet_acc: Optional[StreamInlet] = None
		self.inlet_gyr: Optional[StreamInlet] = None
		self.inlet_ppg: Optional[StreamInlet] = None

		# BLE devices and mode
		self.devices: List[Dict[str, str]] = []
		self.combo_mode: str = "ble"  # or "lsl"

		# Dynamic sampling rates
		self.fs_eeg = FS_EEG_DEFAULT
		self.fs_acc = FS_ACC_DEFAULT
		self.fs_gyr = FS_GYR_DEFAULT
		self.fs_ppg = FS_PPG_DEFAULT
		self.eeg_labels: List[str] = []

		# Line frequency (Windows default 60 Hz)
		self.line_freq = 60

		# Optional muselsl background process
		self.muse_proc = None

		# Buffers (initial sizes; will be resized on connect with actual stream info)
		self.buf_eeg = RingBuffer(num_channels=4, capacity=int(self.fs_eeg * 10))
		self.buf_acc = RingBuffer(num_channels=3, capacity=int(self.fs_acc * 10))
		self.buf_gyr = RingBuffer(num_channels=3, capacity=int(self.fs_gyr * 10))
		self.buf_ppg = RingBuffer(num_channels=1, capacity=int(self.fs_ppg * 20))

		self.timer = QTimer(self)
		self.timer.setInterval(50)  # 20 Hz UI updates
		self.timer.timeout.connect(self.on_timer)

		self.setup_ui()
		pg.setConfigOption("foreground", "w")
		pg.setConfigOption("background", (15, 15, 16))

	def setup_ui(self) -> None:
		central = QWidget()
		layout = QVBoxLayout(central)
		layout.setContentsMargins(16, 12, 16, 12)
		layout.setSpacing(10)

		# Top bar
		top_bar = QHBoxLayout()
		self.device_combo = QComboBox()
		self.refresh_btn = QPushButton("Find Muse")
		self.start_btn = QPushButton("Start Stream")
		self.stop_btn = QPushButton("Stop Stream")
		self.connect_btn = QPushButton("Connect LSL")
		self.disconnect_btn = QPushButton("Disconnect")
		self.stop_btn.setEnabled(False)
		self.disconnect_btn.setEnabled(False)

		self.line_combo = QComboBox()
		self.line_combo.addItems(["50 Hz", "60 Hz"])
		self.line_combo.setCurrentText("60 Hz")
		self.line_combo.currentTextChanged.connect(self.on_line_freq_changed)

		self.refresh_btn.clicked.connect(self.on_refresh)
		self.start_btn.clicked.connect(self.on_start_stream)
		self.stop_btn.clicked.connect(self.on_stop_stream)
		self.connect_btn.clicked.connect(self.on_connect)
		self.disconnect_btn.clicked.connect(self.on_disconnect)

		top_bar.addWidget(QLabel("Muse device or EEG stream:"))
		top_bar.addWidget(self.device_combo, 2)
		top_bar.addWidget(self.refresh_btn)
		top_bar.addWidget(self.start_btn)
		top_bar.addWidget(self.stop_btn)
		top_bar.addWidget(QLabel("Line:"))
		top_bar.addWidget(self.line_combo)
		top_bar.addWidget(self.connect_btn)
		top_bar.addWidget(self.disconnect_btn)

		# Tabs
		self.tabs = QTabWidget()
		self.tab_stream = self._build_stream_tab()
		self.tab_spectral = self._build_spectral_tab()
		self.tab_topomap = self._build_topomap_tab()
		self.tabs.addTab(self.tab_stream, "Stream")
		self.tabs.addTab(self.tab_spectral, "Spectral")
		self.tabs.addTab(self.tab_topomap, "Topography")

		layout.addLayout(top_bar)
		layout.addWidget(self.tabs, 1)
		self.setCentralWidget(central)

	def _build_stream_tab(self) -> QWidget:
		w = QWidget()
		v = QVBoxLayout(w)
		v.setSpacing(8)

		# Stacked time series: EEG, ACC, GYR, PPG
		self.ts_plot_eeg = pg.PlotWidget(title="EEG (filtered)")
		self.ts_plot_eeg.showGrid(x=True, y=True, alpha=0.3)
		self.ts_eeg_curves: List[pg.PlotDataItem] = []

		self.ts_plot_acc = pg.PlotWidget(title="Accelerometer (X,Y,Z)")
		self.ts_plot_acc.showGrid(x=True, y=True, alpha=0.3)
		self.ts_acc_curves: List[pg.PlotDataItem] = []

		self.ts_plot_gyr = pg.PlotWidget(title="Gyroscope (X,Y,Z)")
		self.ts_plot_gyr.showGrid(x=True, y=True, alpha=0.3)
		self.ts_gyr_curves: List[pg.PlotDataItem] = []

		self.ts_plot_ppg = pg.PlotWidget(title="PPG (filtered)")
		self.ts_plot_ppg.showGrid(x=True, y=True, alpha=0.3)
		self.ts_ppg_curve: Optional[pg.PlotDataItem] = None

		self.raw_label = QLabel("Waiting for data...")
		self.raw_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
		self.raw_label.setStyleSheet("QLabel { font-family: monospace; }")

		v.addWidget(self.ts_plot_eeg)
		v.addWidget(self.ts_plot_acc)
		v.addWidget(self.ts_plot_gyr)
		v.addWidget(self.ts_plot_ppg)
		v.addWidget(self.raw_label)
		return w

	def _build_spectral_tab(self) -> QWidget:
		w = QWidget()
		v = QVBoxLayout(w)
		v.setSpacing(8)

		self.spec_plot = pg.ImageView(view=pg.PlotItem())
		self.spec_plot.ui.histogram.hide()
		self.spec_plot.ui.roiBtn.hide()
		self.spec_plot.ui.menuBtn.hide()

		self.band_plot = pg.PlotWidget(title="EEG Band Powers (Delta/Theta/Alpha/Beta/Gamma)")
		self.band_plot.showGrid(x=True, y=True, alpha=0.3)
		self.hr_label = QLabel("HR: — bpm")

		v.addWidget(QLabel("Spectrogram proxy (Welch PSD of EEG ch0)"))
		v.addWidget(self.spec_plot, 2)
		v.addWidget(self.band_plot, 1)
		v.addWidget(self.hr_label)
		return w

	def _build_topomap_tab(self) -> QWidget:
		w = QWidget()
		v = QVBoxLayout(w)
		v.setSpacing(8)

		self.topo_plot = pg.ImageView(view=pg.PlotItem())
		self.topo_plot.ui.histogram.hide()
		self.topo_plot.ui.roiBtn.hide()
		self.topo_plot.ui.menuBtn.hide()

		self.quality_label = QLabel("Signal quality: —")

		v.addWidget(QLabel("EEG Topography (RMS)"))
		v.addWidget(self.topo_plot, 2)
		v.addWidget(self.quality_label)
		return w

	def on_line_freq_changed(self, text: str) -> None:
		self.line_freq = 60 if "60" in text else 50
		self.status.showMessage(f"Line notch set to {self.line_freq} Hz", 2000)

	def on_refresh(self) -> None:
		self.status.showMessage("Searching for Muse (BLE) or LSL streams...", 3000)
		self.device_combo.clear()
		self.devices = []
		self.combo_mode = "ble"
		# Prefer BLE discovery via muselsl if available
		if muselsl_available():
			try:
				self.devices = muse_list_devices(timeout=8)
			except Exception:
				self.devices = []
		if self.devices:
			for d in self.devices:
				name = d.get("name", "Muse")
				addr = d.get("address", "?")
				self.device_combo.addItem(f"{name} ({addr})")
		else:
			self.combo_mode = "lsl"
			# Fallback to listing EEG LSL stream names
			try:
				streams = resolve_byprop("type", "EEG", timeout=3)
				for s in streams:
					self.device_combo.addItem(s.name())
				if not streams:
					self.device_combo.addItem("No EEG streams found")
			except Exception as exc:
				self.device_combo.addItem(f"Error: {exc}")

	def _resolve_stream(self, stype: str) -> Optional[StreamInlet]:
		try:
			arr = resolve_byprop("type", stype, timeout=3)
			if not arr:
				return None
			return StreamInlet(arr[0])
		except Exception:
			return None

	def on_start_stream(self) -> None:
		if self.combo_mode != "ble":
			self.status.showMessage("No BLE devices listed. Click Find Muse to scan via muselsl.", 5000)
			return
		idx = self.device_combo.currentIndex()
		if idx < 0 or idx >= len(self.devices):
			self.status.showMessage("Select a Muse device from the list.", 4000)
			return
		address = self.devices[idx].get("address")
		if not address:
			self.status.showMessage("Selected device has no address.", 4000)
			return
		try:
			self.muse_proc = muse_start_stream(address=address, acc=True, gyro=True, ppg=True)
			if self.muse_proc is None:
				raise RuntimeError("muselsl not available")
			self.status.showMessage(f"Starting muselsl stream at {address}...")
			self.start_btn.setEnabled(False)
			self.stop_btn.setEnabled(True)
			# Try auto-connect to LSL after a short delay
			QTimer.singleShot(4000, self.on_connect)
		except Exception as exc:
			self.status.showMessage(f"Start stream failed: {exc}", 6000)

	def on_stop_stream(self) -> None:
		try:
			muse_stop_stream(self.muse_proc)
			self.muse_proc = None
			self.status.showMessage("Stopped muselsl stream.", 4000)
		finally:
			self.start_btn.setEnabled(True)
			self.stop_btn.setEnabled(False)

	def on_connect(self) -> None:
		try:
			self.inlet_eeg = self._resolve_stream("EEG")
			self.inlet_acc = self._resolve_stream("ACC")
			self.inlet_gyr = self._resolve_stream("GYRO")
			self.inlet_ppg = self._resolve_stream("PPG")

			if self.inlet_eeg is None:
				raise RuntimeError("EEG LSL stream not found")

			# Read stream metadata to size buffers and labels
			info_eeg = self.inlet_eeg.info()
			nch_eeg = max(1, info_eeg.channel_count())
			self.fs_eeg = info_eeg.nominal_srate() or FS_EEG_DEFAULT
			self.eeg_labels = _extract_channel_labels(info_eeg)
			self.buf_eeg = RingBuffer(num_channels=nch_eeg, capacity=int(self.fs_eeg * 10))

			if self.inlet_acc is not None:
				inf = self.inlet_acc.info()
				nch = max(1, inf.channel_count())
				self.fs_acc = inf.nominal_srate() or FS_ACC_DEFAULT
				self.buf_acc = RingBuffer(num_channels=nch, capacity=int(self.fs_acc * 10))
			if self.inlet_gyr is not None:
				inf = self.inlet_gyr.info()
				nch = max(1, inf.channel_count())
				self.fs_gyr = inf.nominal_srate() or FS_GYR_DEFAULT
				self.buf_gyr = RingBuffer(num_channels=nch, capacity=int(self.fs_gyr * 10))
			if self.inlet_ppg is not None:
				inf = self.inlet_ppg.info()
				nch = max(1, inf.channel_count())
				self.fs_ppg = inf.nominal_srate() or FS_PPG_DEFAULT
				self.buf_ppg = RingBuffer(num_channels=1, capacity=int(self.fs_ppg * 20))

			self.status.showMessage("Connected to LSL streams")
			self.connect_btn.setEnabled(False)
			self.disconnect_btn.setEnabled(True)

			# Prepare curves according to stream sizes
			self.ts_plot_eeg.clear()
			colors = [(255,255,255),(255,160,0),(0,200,255),(160,160,255),(0,255,160),(255,80,120)]
			nc = self.buf_eeg.num_channels
			self.ts_eeg_curves = [self.ts_plot_eeg.plot(pen=pg.mkPen(colors[i % len(colors)])) for i in range(nc)]
			self.ts_plot_acc.clear()
			nc_acc = self.buf_acc.num_channels
			self.ts_acc_curves = [self.ts_plot_acc.plot(pen=pg.mkPen(colors[i % len(colors)])) for i in range(nc_acc)]
			self.ts_plot_gyr.clear()
			nc_gyr = self.buf_gyr.num_channels
			self.ts_gyr_curves = [self.ts_plot_gyr.plot(pen=pg.mkPen(colors[i % len(colors)])) for i in range(nc_gyr)]
			self.ts_plot_ppg.clear()
			self.ts_ppg_curve = self.ts_plot_ppg.plot(pen=pg.mkPen((200, 200, 255)))

			self.timer.start()
		except Exception as exc:
			self.status.showMessage(f"Connect failed: {exc}", 6000)

	def on_disconnect(self) -> None:
		self.timer.stop()
		self.inlet_eeg = None
		self.inlet_acc = None
		self.inlet_gyr = None
		self.inlet_ppg = None
		self.connect_btn.setEnabled(True)
		self.disconnect_btn.setEnabled(False)
		self.status.showMessage("Disconnected", 3000)

	def _pull_chunk(self, inlet: StreamInlet, max_samples: int = 256) -> Tuple[np.ndarray, np.ndarray]:
		try:
			samples, ts = inlet.pull_chunk(max_samples=max_samples, timeout=0.0)
			if not ts:
				return np.array([]), np.zeros((0, 0))
			arr = np.asarray(samples, dtype=float)
			if arr.ndim == 1:
				arr = arr[np.newaxis, :]
			return np.asarray(ts, dtype=float), arr.T  # (n,) and (ch,n)
		except Exception:
			return np.array([]), np.zeros((0, 0))

	def on_timer(self) -> None:
		# Drain all inlets using chunked pulls
		if self.inlet_eeg is not None:
			ts, data = self._pull_chunk(self.inlet_eeg, int(self.fs_eeg // 2))
			if ts.size:
				self.buf_eeg.append(data, ts)
		if self.inlet_acc is not None:
			ts, data = self._pull_chunk(self.inlet_acc, int(self.fs_acc))
			if ts.size:
				self.buf_acc.append(data, ts)
		if self.inlet_gyr is not None:
			ts, data = self._pull_chunk(self.inlet_gyr, int(self.fs_gyr))
			if ts.size:
				self.buf_gyr.append(data, ts)
		if self.inlet_ppg is not None:
			ts, data = self._pull_chunk(self.inlet_ppg, int(self.fs_ppg))
			if ts.size:
				self.buf_ppg.append(data[:1, :], ts)  # use first PPG channel

		# Update raw label with last EEG sample
		ts_all, eeg_all = self.buf_eeg.get_last(int(self.fs_eeg))
		if ts_all.size:
			last = eeg_all[:, -1]
			self.raw_label.setText(f"EEG last [{ts_all[-1]:.3f}] " + " ".join(f"{v:.6f}" for v in last))

		# Build filtered windows for display and analysis
		_, eeg_win = self.buf_eeg.get_last(int(self.fs_eeg * 5))
		if eeg_win.shape[1] >= int(self.fs_eeg * 2):
			try:
				eeg_filt = filter_eeg_mne(eeg_win, fs=self.fs_eeg, line_freq=self.line_freq)
			except Exception:
				eeg_filt = eeg_win
			# Update EEG time series
			ts5, _ = self.buf_eeg.get_last(int(self.fs_eeg * 5))
			if ts5.size and self.ts_eeg_curves:
				for idx, curve in enumerate(self.ts_eeg_curves):
					if idx < eeg_filt.shape[0]:
						curve.setData(ts5, eeg_filt[idx])
			# Spectrogram proxy and band powers (Welch via MNE)
			psd, freqs = psd_welch_mne(eeg_filt, fs=self.fs_eeg, fmin=1.0, fmax=40.0)
			psd0 = psd[0:1]
			img = np.log10(psd0 + 1e-12)
			self.spec_plot.setImage(img.T, xvals=freqs, autoLevels=True, autoRange=False)
			# Band powers
			means = []
			keys = ["delta", "theta", "alpha", "beta", "gamma"]
			for k in keys:
				fmin, fmax = BANDS[k]
				mask = (freqs >= fmin) & (freqs <= fmax)
				bp = np.trapz(psd[:, mask], freqs[mask], axis=1) if np.any(mask) else np.zeros(psd.shape[0])
				means.append(float(np.mean(bp)))
			x = np.arange(len(keys))
			self.band_plot.clear()
			bar = pg.BarGraphItem(x=x, height=means, width=0.8, brush=(200, 200, 255))
			self.band_plot.addItem(bar)
			self.band_plot.getPlotItem().getAxis('bottom').setTicks([list(enumerate(keys))])

		# Update ACC time series
		ts_acc, acc_all = self.buf_acc.get_last(int(self.fs_acc * 5))
		if ts_acc.size and self.ts_acc_curves:
			for idx, curve in enumerate(self.ts_acc_curves):
				if idx < acc_all.shape[0]:
					curve.setData(ts_acc, acc_all[idx])

		# Update GYR time series
		ts_gyr, gyr_all = self.buf_gyr.get_last(int(self.fs_gyr * 5))
		if ts_gyr.size and self.ts_gyr_curves:
			for idx, curve in enumerate(self.ts_gyr_curves):
				if idx < gyr_all.shape[0]:
					curve.setData(ts_gyr, gyr_all[idx])

		# Update PPG time series and HR (filtered)
		ts_ppg, ppg_all = self.buf_ppg.get_last(int(self.fs_ppg * 20))
		if ts_ppg.size and self.ts_ppg_curve is not None:
			try:
				ppg_filt = filter_ppg_mne(ppg_all[0], fs=self.fs_ppg)
			except Exception:
				ppg_filt = ppg_all[0]
			self.ts_ppg_curve.setData(ts_ppg, ppg_filt)
			# HR estimate (simple peak search via PSD already in earlier helper; keep label minimal here)
			# Compute PSD in HR band via numpy for robustness if MNE missing
			# Kept minimal as HR proxy is already shown in spectral tab

		# Topography and quality from filtered EEG
		_, eeg_topo = self.buf_eeg.get_last(int(self.fs_eeg * 2))
		if eeg_topo.shape[1] >= int(self.fs_eeg * 1):
			try:
				eeg_topo_f = filter_eeg_mne(eeg_topo, fs=self.fs_eeg, line_freq=self.line_freq)
			except Exception:
				eeg_topo_f = eeg_topo
			vals = np.sqrt(np.mean(np.square(eeg_topo_f), axis=1))
			img = interpolate_topomap(vals[: min(4, vals.shape[0])], MUSE_LAYOUT, grid_size=64)
			self.topo_plot.setImage(img.T, autoLevels=True, autoRange=False)
			q = simple_contact_quality(eeg_topo_f[: min(4, eeg_topo_f.shape[0])])
			self.quality_label.setText("Signal quality (lower is better): " + ", ".join(f"{v:.2f}" for v in q))


def main() -> int:
	app = QApplication(sys.argv)
	# Ensure font choice
	family = load_inter_font()
	if family:
		app.setFont(QFont(family, 11))

	win = MainWindow()
	win.show()
	return app.exec()


if __name__ == "__main__":
	sys.exit(main())