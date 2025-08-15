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
	compute_welch_psd,
	compute_band_powers,
	interpolate_topomap,
	simple_contact_quality,
	estimate_heart_rate_ppg,
)
from app.muse_control import list_devices as muse_list_devices, start_stream as muse_start_stream, stop_stream as muse_stop_stream, muselsl_available


APP_TITLE = "NeuroStream"

# Approximate sampling rates
FS_EEG = 256.0
FS_ACC = 52.0
FS_GYR = 52.0
FS_PPG = 64.0

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

		# Optional muselsl background process
		self.muse_proc = None

		# Buffers (10–20 seconds)
		self.buf_eeg = RingBuffer(num_channels=4, capacity=int(FS_EEG * 10))
		self.buf_acc = RingBuffer(num_channels=3, capacity=int(FS_ACC * 10))
		self.buf_gyr = RingBuffer(num_channels=3, capacity=int(FS_GYR * 10))
		self.buf_ppg = RingBuffer(num_channels=1, capacity=int(FS_PPG * 20))

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
		self.ts_plot_eeg = pg.PlotWidget(title="EEG (4 ch)")
		self.ts_plot_eeg.showGrid(x=True, y=True, alpha=0.3)
		self.ts_eeg_curves: List[pg.PlotDataItem] = []

		self.ts_plot_acc = pg.PlotWidget(title="Accelerometer (X,Y,Z)")
		self.ts_plot_acc.showGrid(x=True, y=True, alpha=0.3)
		self.ts_acc_curves: List[pg.PlotDataItem] = []

		self.ts_plot_gyr = pg.PlotWidget(title="Gyroscope (X,Y,Z)")
		self.ts_plot_gyr.showGrid(x=True, y=True, alpha=0.3)
		self.ts_gyr_curves: List[pg.PlotDataItem] = []

		self.ts_plot_ppg = pg.PlotWidget(title="PPG")
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
		self.band_bars: List[pg.BarGraphItem] = []
		self.hr_label = QLabel("HR: — bpm")

		v.addWidget(QLabel("Spectrogram (EEG representative)"))
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

		v.addWidget(QLabel("EEG Topography"))
		v.addWidget(self.topo_plot, 2)
		v.addWidget(self.quality_label)
		return w

	def on_refresh(self) -> None:
		self.status.showMessage("Searching for Muse (BLE) or LSL streams...", 3000)
		self.device_combo.clear()
		# Prefer BLE discovery via muselsl if available
		ble = []
		if muselsl_available():
			try:
				ble = muse_list_devices(timeout=8)
			except Exception:
				ble = []
		if ble:
			for name in ble:
				self.device_combo.addItem(name)
		else:
			# Fallback to listing EEG LSL stream names
			try:
				streams = resolve_byprop("type", "EEG", timeout=3)
				for s in streams:
					self.device_combo.addItem(s.name())
				if not streams:
					self.device_combo.addItem("No EEG streams found")
			except Exception as exc:
				self.device_combo.addItem(f"Error: {exc}")

	def _resolve_stream(self, stype: str, name: Optional[str] = None) -> Optional[StreamInlet]:
		try:
			if name:
				arr = resolve_byprop("name", name, timeout=2)
			else:
				arr = resolve_byprop("type", stype, timeout=2)
			if not arr:
				return None
			return StreamInlet(arr[0])
		except Exception:
			return None

	def on_start_stream(self) -> None:
		name = self.device_combo.currentText().strip()
		if not name or name.startswith("No EEG") or name.startswith("Error:"):
			self.status.showMessage("Select a Muse name first (or use Connect LSL if a stream exists).", 5000)
			return
		if not muselsl_available():
			self.status.showMessage("muselsl not found. Install it to start BLE streaming.", 5000)
			return
		try:
			self.muse_proc = muse_start_stream(name_or_mac=name, acc=True, gyro=True, ppg=True)
			if self.muse_proc is None:
				raise RuntimeError("Failed to start muselsl")
			self.status.showMessage(f"Starting muselsl stream for {name}...")
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
		name = self.device_combo.currentText().strip()
		try:
			# Connect to LSL streams; if name fails, connect by type
			self.inlet_eeg = self._resolve_stream("EEG", name if name and not name.startswith("Error:") else None)
			if self.inlet_eeg is None:
				self.inlet_eeg = self._resolve_stream("EEG")
			self.inlet_acc = self._resolve_stream("ACC")
			self.inlet_gyr = self._resolve_stream("GYRO")
			self.inlet_ppg = self._resolve_stream("PPG")

			if self.inlet_eeg is None:
				raise RuntimeError("EEG LSL stream not found")

			self.status.showMessage("Connected to LSL streams")
			self.connect_btn.setEnabled(False)
			self.disconnect_btn.setEnabled(True)

			# Prepare curves
			self.ts_plot_eeg.clear()
			self.ts_eeg_curves = [self.ts_plot_eeg.plot(pen=pg.mkPen(color)) for color in [
				(255, 255, 255), (255, 160, 0), (0, 200, 255), (160,160,255),
			]]
			self.ts_plot_acc.clear()
			self.ts_acc_curves = [self.ts_plot_acc.plot(pen=pg.mkPen(c)) for c in [(200,200,200),(200,120,0),(0,200,140)]]
			self.ts_plot_gyr.clear()
			self.ts_gyr_curves = [self.ts_plot_gyr.plot(pen=pg.mkPen(c)) for c in [(200,200,200),(200,120,0),(0,200,140)]]
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

	def _drain_inlet(self, inlet: StreamInlet, max_samples: int = 64) -> Tuple[np.ndarray, np.ndarray]:
		ts_list: List[float] = []
		vals: List[List[float]] = []
		for _ in range(max_samples):
			s, ts = inlet.pull_sample(timeout=0.0)
			if s is None:
				break
			ts_list.append(ts)
			vals.append(s)
		if not ts_list:
			return np.array([]), np.zeros((0, 0))
		arr = np.asarray(vals, dtype=float)
		return np.asarray(ts_list, dtype=float), arr.T  # (n,) and (ch,n)

	def on_timer(self) -> None:
		# Drain all inlets
		if self.inlet_eeg is not None:
			ts, data = self._drain_inlet(self.inlet_eeg, 128)
			if ts.size:
				self.buf_eeg.append(data, ts)
		if self.inlet_acc is not None:
			ts, data = self._drain_inlet(self.inlet_acc, 64)
			if ts.size:
				self.buf_acc.append(data, ts)
		if self.inlet_gyr is not None:
			ts, data = self._drain_inlet(self.inlet_gyr, 64)
			if ts.size:
				self.buf_gyr.append(data, ts)
		if self.inlet_ppg is not None:
			ts, data = self._drain_inlet(self.inlet_ppg, 64)
			if ts.size:
				self.buf_ppg.append(data[:1, :], ts)  # take first PPG channel

		# Update raw label with last EEG sample
		ts_all, eeg_all = self.buf_eeg.get_last(256)
		if ts_all.size:
			last = eeg_all[:, -1]
			self.raw_label.setText(f"EEG last [{ts_all[-1]:.3f}] " + " ".join(f"{v:.6f}" for v in last))

		# Update EEG time series
		if ts_all.size and self.ts_eeg_curves:
			for idx, curve in enumerate(self.ts_eeg_curves):
				if idx < eeg_all.shape[0]:
					curve.setData(ts_all, eeg_all[idx])

		# Update ACC time series
		ts_acc, acc_all = self.buf_acc.get_last(int(FS_ACC * 5))
		if ts_acc.size and self.ts_acc_curves:
			for idx, curve in enumerate(self.ts_acc_curves):
				if idx < acc_all.shape[0]:
					curve.setData(ts_acc, acc_all[idx])

		# Update GYR time series
		ts_gyr, gyr_all = self.buf_gyr.get_last(int(FS_GYR * 5))
		if ts_gyr.size and self.ts_gyr_curves:
			for idx, curve in enumerate(self.ts_gyr_curves):
				if idx < gyr_all.shape[0]:
					curve.setData(ts_gyr, gyr_all[idx])

		# Update PPG time series and HR
		ts_ppg, ppg_all = self.buf_ppg.get_last(int(FS_PPG * 20))
		if ts_ppg.size and self.ts_ppg_curve is not None:
			self.ts_ppg_curve.setData(ts_ppg, ppg_all[0])
			if ppg_all.shape[1] >= int(FS_PPG * 8):
				bpm, conf = estimate_heart_rate_ppg(ppg_all[0], FS_PPG)
				self.hr_label.setText(f"HR: {bpm:.0f} bpm (conf {conf:.2f})")

		# Spectrogram and bands using recent EEG window
		_, eeg_win = self.buf_eeg.get_last(int(FS_EEG * 5))
		if eeg_win.shape[1] >= int(FS_EEG * 2):
			psd = compute_welch_psd(eeg_win, FS_EEG)
			bands = compute_band_powers(psd)
			# Render log-PSD of channel 0 as a pseudo-spectrogram
			ch0 = eeg_win[0:1, :]
			psd0 = compute_welch_psd(ch0, FS_EEG)
			img = np.log10(psd0.psd + 1e-12)
			self.spec_plot.setImage(img.T, xvals=psd0.freqs, autoLevels=True, autoRange=False)
			# Band bar plot (mean across channels)
			self.band_plot.clear()
			keys = ["delta", "theta", "alpha", "beta", "gamma"]
			means = [float(np.mean(bands[k])) for k in keys]
			x = np.arange(len(keys))
			bar = pg.BarGraphItem(x=x, height=means, width=0.8, brush=(200, 200, 255))
			self.band_plot.addItem(bar)
			self.band_plot.getPlotItem().getAxis('bottom').setTicks([list(enumerate(keys))])

		# Topography and quality from EEG
		_, eeg_topo = self.buf_eeg.get_last(int(FS_EEG * 2))
		if eeg_topo.shape[1] >= int(FS_EEG * 1):
			vals = np.sqrt(np.mean(np.square(eeg_topo), axis=1))
			img = interpolate_topomap(vals[:4], MUSE_LAYOUT, grid_size=64)
			self.topo_plot.setImage(img.T, autoLevels=True, autoRange=False)
			q = simple_contact_quality(eeg_topo[:4])
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