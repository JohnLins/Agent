"""Screen capture via Spectacle (KDE's native, works on Wayland with no portal dance)."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path


CAPTURE_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "computrr" / "captures"


def _ensure_dir() -> None:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


def default_path() -> Path:
    _ensure_dir()
    return CAPTURE_DIR / f"capture-{int(time.time() * 1000)}.png"


def capture_screen(out: Path | None = None, *, include_pointer: bool = False) -> Path:
    """Capture the full virtual display to a PNG file.

    Returns the path written. Raises RuntimeError if spectacle fails.
    """
    out = out or default_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["spectacle", "-b", "-n", "-o", str(out)]
    if include_pointer:
        args.append("-p")
    proc = subprocess.run(args, capture_output=True, text=True, timeout=10)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"spectacle failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    # spectacle in background mode sometimes writes asynchronously; wait briefly.
    for _ in range(20):
        if out.exists() and out.stat().st_size > 0:
            break
        time.sleep(0.05)
    if out.stat().st_size == 0:
        raise RuntimeError(f"spectacle produced empty file at {out}")
    return out


def capture_region(x: int, y: int, w: int, h: int, out: Path | None = None) -> Path:
    """Capture a screen region by first grabbing the full screen, then cropping."""
    from PIL import Image

    full = capture_screen()
    img = Image.open(full)
    crop = img.crop((x, y, x + w, y + h))
    out = out or default_path()
    crop.save(out)
    return out
