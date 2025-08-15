from __future__ import annotations

from typing import List, Dict, Optional
import multiprocessing as mp

try:
	from muselsl import list_muses, stream as muselsl_stream
	_MUSE_AVAILABLE = True
except Exception:
	_MUSE_AVAILABLE = False


def muselsl_available() -> bool:
	return _MUSE_AVAILABLE


def list_devices(timeout: int = 8) -> List[Dict[str, str]]:
	"""Return a list of Muse devices with 'name' and 'address' using muselsl API."""
	if not _MUSE_AVAILABLE:
		return []
	try:
		muses = list_muses(timeout=timeout)
		# muselsl returns list of dicts with keys like 'name' and 'address'
		return muses or []
	except Exception:
		return []


def _stream_target(address: str, acc: bool, gyro: bool, ppg: bool) -> None:
	# muselsl.stream is blocking; run inside a separate process
	muselsl_stream(address=address, ppg=ppg, acc=acc, gyro=gyro)


def start_stream(address: str, acc: bool = True, gyro: bool = True, ppg: bool = True) -> Optional[mp.Process]:
	"""Start muselsl streaming in a background process. Returns the Process or None."""
	if not _MUSE_AVAILABLE:
		return None
	ctx = mp.get_context('spawn')
	proc = ctx.Process(target=_stream_target, args=(address, acc, gyro, ppg), daemon=True)
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