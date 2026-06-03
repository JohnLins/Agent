"""Post-action verification: did the screen change? Cheap perceptual-hash based diff."""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from . import capture


def _avg_hash(image: Image.Image, size: int = 16) -> int:
    img = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p > avg:
            bits |= 1 << i
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def hash_path(path: Path) -> int:
    with Image.open(path) as im:
        return _avg_hash(im)


def wait_for_change(
    *,
    timeout_ms: int = 2000,
    interval_ms: int = 100,
    threshold: int = 8,
) -> dict:
    """Snapshot now, then poll until the average-hash hamming distance exceeds `threshold`
    or `timeout_ms` elapses. `threshold` is bits out of 256 (16x16)."""
    start_path = capture.capture_screen()
    start = hash_path(start_path)
    t0 = time.monotonic()
    last_dist = 0
    while (time.monotonic() - t0) * 1000 < timeout_ms:
        time.sleep(interval_ms / 1000)
        now_path = capture.capture_screen()
        now = hash_path(now_path)
        last_dist = _hamming(start, now)
        if last_dist >= threshold:
            return {
                "changed": True,
                "distance": last_dist,
                "elapsed_ms": int((time.monotonic() - t0) * 1000),
                "before": str(start_path),
                "after": str(now_path),
            }
    return {
        "changed": False,
        "distance": last_dist,
        "elapsed_ms": int((time.monotonic() - t0) * 1000),
        "before": str(start_path),
    }
