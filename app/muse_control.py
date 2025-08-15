from __future__ import annotations

import multiprocessing as mp
from typing import List, Dict, Optional


def _import_muselsl():
	try:
		import muselsl  # type: ignore
		return muselsl
	except Exception:
		return None


def muselsl_available() -> bool:
	return _import_muselsl() is not None


def list_devices(timeout: int = 8) -> List[Dict[str, str]]:
	"""Return a list of discovered Muse devices with name and address.
	If muselsl is unavailable or none found, returns an empty list.
	"""
	muselsl = _import_muselsl()
	if muselsl is None:
		return []
	try:
		muses = muselsl.list_muses(timeout=timeout)
		# muses is typically a list of dicts with keys 'name' and 'address'
		results: List[Dict[str, str]] = []
		for m in muses:
			name = str(m.get('name', 'Muse'))
			addr = str(m.get('address', ''))
			results.append({'name': name, 'address': addr})
		return results
	except Exception:
		return []


def _stream_target(name_or_address: str, acc: bool, gyro: bool, ppg: bool) -> None:
	muselsl = _import_muselsl()
	if muselsl is None:
		return
	kwargs = {
		'ppg': ppg,
		'acc': acc,
		'gyro': gyro,
		'disable_eeg': False,
	}
	# Heuristic to choose address vs name
	if ':' in name_or_address and len(name_or_address) >= 11:
		kwargs['address'] = name_or_address
	else:
		kwargs['name'] = name_or_address
	# This call blocks until interrupted. Running in a child process allows stopping via terminate().
	muselsl.stream(**kwargs)


def start_stream(name_or_address: str, acc: bool = True, gyro: bool = True, ppg: bool = True) -> Optional[mp.Process]:
	"""Start muselsl streaming in a background process. Returns the process handle or None."""
	if not muselsl_available():
		return None
	proc = mp.Process(target=_stream_target, args=(name_or_address, acc, gyro, ppg), daemon=True)
	proc.start()
	return proc


def stop_stream(proc: Optional[mp.Process]) -> None:
	if proc is None:
		return
	try:
		proc.terminate()
		proc.join(timeout=5)
	except Exception:
		pass