#!/usr/bin/env python3

import argparse
import sys
from typing import Optional, List

from pylsl import StreamInlet, StreamInfo, resolve_byprop


def fetch_stream_info(stream_type: str, name: Optional[str], timeout: float) -> StreamInfo:
    """Resolve an LSL stream by type or name and return its StreamInfo.

    Prefers resolving by name if provided, otherwise resolves by type.
    Raises RuntimeError if no matching stream is found within the timeout.
    """
    streams = []
    if name:
        streams = resolve_byprop("name", name, timeout=timeout)
        if not streams:
            raise RuntimeError(
                f"No LSL stream found with name '{name}' within {timeout:.1f}s"
            )
    else:
        streams = resolve_byprop("type", stream_type, timeout=timeout)
        if not streams:
            raise RuntimeError(
                f"No LSL stream found with type '{stream_type}' within {timeout:.1f}s"
            )

    return streams[0]


def try_get_channel_labels(info: StreamInfo) -> Optional[List[str]]:
    """Best-effort extraction of channel labels from the stream metadata."""
    try:
        desc = info.desc()
        channels = desc.child("channels")
        if channels.empty():
            return None
        labels: List[str] = []
        ch = channels.child("channel")
        while not ch.empty():
            label = ch.child_value("label")
            labels.append(label if label else "")
            ch = ch.next_sibling()
        return labels or None
    except Exception:
        return None


def print_stream_header(info: StreamInfo) -> None:
    name = info.name()
    stype = info.type()
    channel_count = info.channel_count()
    nominal_srate = info.nominal_srate()
    source_id = info.source_id()

    print(
        f"Connected to LSL stream: name='{name}', type='{stype}', "
        f"channels={channel_count}, srate={nominal_srate}, source_id='{source_id}'",
        flush=True,
    )

    labels = try_get_channel_labels(info)
    if labels:
        print(f"Channel labels: {', '.join(labels)}", flush=True)


def receive_and_print_samples(
    inlet: StreamInlet, sample_limit: int
) -> None:
    printed = 0
    while True:
        try:
            sample, timestamp = inlet.pull_sample(timeout=5.0)
            if sample is None:
                # Timeout without data; continue waiting.
                continue

            # Print timestamp with 3 decimals and all channel values.
            values = " ".join(f"{v:.6f}" for v in sample)
            print(f"[{timestamp:.3f}] {values}", flush=True)

            printed += 1
            if sample_limit > 0 and printed >= sample_limit:
                break
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.", flush=True)
            break


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Receive and print samples from an LSL EEG stream (e.g., Muse).\n"
            "By default, resolves the first stream with type 'EEG'."
        )
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Exact LSL stream name to connect to (e.g., 'Muse' or 'MuseS').",
    )
    parser.add_argument(
        "--type",
        dest="stream_type",
        type=str,
        default="EEG",
        help="LSL stream type to resolve if --name is not provided (default: EEG).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for stream resolution (default: 10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Number of samples to print before exiting (0 means unlimited; default: 0)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        info = fetch_stream_info(args.stream_type, args.name, args.timeout)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1

    inlet = StreamInlet(info)
    print_stream_header(info)

    receive_and_print_samples(inlet, sample_limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())