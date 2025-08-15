from __future__ import annotations

import os
import sys
import shutil
import subprocess
from typing import List, Optional


def muselsl_available() -> bool:
	return shutil.which("muselsl") is not None


def _python_exe() -> str:
	return sys.executable or "python"


def list_devices(timeout: int = 8) -> List[str]:
	"""Return a list of Muse device names via muselsl list; empty if unavailable."""
	if not muselsl_available():
		return []
	try:
		proc = subprocess.run(
			["muselsl", "list"],
			capture_output=True,
			text=True,
			timeout=timeout,
		)
		out = (proc.stdout or "") + (proc.stderr or "")
		# Parse lines like: Found Muses: Muse-XXXX (or MAC addresses depending on platform)
		names: List[str] = []
		for line in out.splitlines():
			line = line.strip()
			if not line:
				continue
			# Heuristic: lines containing 'Muse' and a dash often indicate names
			if "Muse" in line:
				# Extract tokens that look like Muse-XXXX
				for token in line.replace(",", " ").split():
					if token.startswith("Muse"):
						names.append(token.strip())
		return sorted(set(names))
	except Exception:
		return []


def start_stream(name_or_mac: str, acc: bool = True, gyro: bool = True, ppg: bool = True) -> Optional[subprocess.Popen]:
	"""Launch muselsl stream for a device by name or MAC in background.
	Returns a Popen handle, or None if muselsl is unavailable.
	"""
	if not muselsl_available():
		return None
	cmd = [_python_exe(), "-m", "muselsl", "stream"]
	# Decide whether it is a MAC address or name
	if ":" in name_or_mac or name_or_mac.count("-") == 5:
		cmd += ["-a", name_or_mac]
	else:
		cmd += ["-n", name_or_mac]
	if acc:
		cmd.append("--acc")
	if gyro:
		cmd.append("--gyro")
	if ppg:
		cmd.append("--ppg")
	creationflags = 0
	if os.name == "nt":
		# Avoid opening a new console window on Windows
		creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
	return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)


def stop_stream(proc: Optional[subprocess.Popen]) -> None:
	if proc is None:
		return
	try:
		proc.terminate()
		proc.wait(timeout=5)
	except Exception:
		try:
			proc.kill()
		except Exception:
			pass