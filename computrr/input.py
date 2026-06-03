"""Input injection.

Mouse: direct /dev/uinput virtual ABSOLUTE pointer via python-evdev. We use
EV_ABS instead of EV_REL so KWin/libinput doesn't apply pointer acceleration,
and absolute coordinates land exactly where commanded.

Keyboard: ydotool 0.1.x — its `type` and `key` commands work reliably here,
and reimplementing keymap/layout handling in pure Python would be a lot of
code for no real win.
"""

from __future__ import annotations

import atexit
import subprocess
import time
from typing import Iterable


# Linux input event codes (subset). Source: /usr/include/linux/input-event-codes.h
KEYCODES: dict[str, int] = {
    # numbers
    "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
    "6": 7, "7": 8, "8": 9, "9": 10, "0": 11,
    # modifiers
    "ctrl": 29, "control": 29, "leftctrl": 29,
    "rightctrl": 97,
    "shift": 42, "leftshift": 42,
    "rightshift": 54,
    "alt": 56, "leftalt": 56,
    "rightalt": 100, "altgr": 100,
    "super": 125, "meta": 125, "win": 125, "leftmeta": 125,
    # navigation / control
    "enter": 28, "return": 28,
    "escape": 1, "esc": 1,
    "tab": 15,
    "space": 57,
    "backspace": 14,
    "delete": 111, "del": 111,
    "home": 102, "end": 107,
    "pageup": 104, "pagedown": 109,
    "up": 103, "down": 108, "left": 105, "right": 106,
    "insert": 110,
    "capslock": 58,
    "minus": 12, "-": 12,
    "equal": 13, "=": 13,
    "leftbrace": 26, "[": 26,
    "rightbrace": 27, "]": 27,
    "semicolon": 39, ";": 39,
    "apostrophe": 40, "'": 40,
    "grave": 41, "`": 41,
    "backslash": 43, "\\": 43,
    "comma": 51, ",": 51,
    "dot": 52, ".": 52,
    "slash": 53, "/": 53,
    # function keys
    **{f"f{i}": 58 + i for i in range(1, 11)},  # F1=59..F10=68
    "f11": 87, "f12": 88,
}

# Letters (Linux input event codes — non-sequential, must be enumerated explicitly).
KEYCODES.update({
    "a": 30, "b": 48, "c": 46, "d": 32, "e": 18, "f": 33, "g": 34,
    "h": 35, "i": 23, "j": 36, "k": 37, "l": 38, "m": 50, "n": 49,
    "o": 24, "p": 25, "q": 16, "r": 19, "s": 31, "t": 20, "u": 22,
    "v": 47, "w": 17, "x": 45, "y": 21, "z": 44,
})


class InputError(RuntimeError):
    pass


def _run_ydotool(*args: str) -> None:
    proc = subprocess.run(["ydotool", *args], capture_output=True, text=True, timeout=10)
    if proc.returncode != 0:
        raise InputError(
            f"ydotool {' '.join(args)} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )


# --- Virtual absolute pointer (EV_ABS uinput device) -------------------------

_uinput = None  # type: ignore[var-annotated]
_screen_w: int = 0
_screen_h: int = 0
_BUTTON_CODES: dict[str, int] = {}  # filled lazily after evdev import


def _detect_screen_size() -> tuple[int, int]:
    """Return (width, height) in PHYSICAL pixels of the screenshot space.

    We take a one-shot Spectacle capture and read its dimensions, so coordinates
    the agent sees in screenshots match what we feed uinput.
    """
    from PIL import Image  # local import: avoid cycle on module load

    from . import capture

    path = capture.capture_screen()
    with Image.open(path) as img:
        return img.size  # (w, h)


def _ensure_pointer() -> None:
    global _uinput, _screen_w, _screen_h, _BUTTON_CODES
    if _uinput is not None:
        return
    try:
        from evdev import AbsInfo, UInput, ecodes as e
    except ImportError as exc:
        raise InputError(
            "python-evdev not installed (pip install evdev) — required for mouse control"
        ) from exc

    _screen_w, _screen_h = _detect_screen_size()
    caps = {
        e.EV_ABS: [
            (e.ABS_X, AbsInfo(value=0, min=0, max=_screen_w - 1, fuzz=0, flat=0, resolution=0)),
            (e.ABS_Y, AbsInfo(value=0, min=0, max=_screen_h - 1, fuzz=0, flat=0, resolution=0)),
        ],
        e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
        e.EV_REL: [e.REL_WHEEL, e.REL_HWHEEL],
    }
    try:
        _uinput = UInput(caps, name="computrr-virtual-pointer", version=1)
    except PermissionError as exc:
        raise InputError(
            "cannot open /dev/uinput — grant access with: "
            "sudo setfacl -m u:$USER:rw /dev/uinput"
        ) from exc
    _BUTTON_CODES = {"left": e.BTN_LEFT, "right": e.BTN_RIGHT, "middle": e.BTN_MIDDLE}
    # Give the compositor a moment to enumerate the new device before the first event.
    time.sleep(0.25)
    atexit.register(_close_pointer)


def _close_pointer() -> None:
    global _uinput
    if _uinput is not None:
        try:
            _uinput.close()
        except Exception:
            pass
        _uinput = None


def move(x: int, y: int) -> None:
    from evdev import ecodes as e
    _ensure_pointer()
    cx = max(0, min(_screen_w - 1, int(x)))
    cy = max(0, min(_screen_h - 1, int(y)))
    _uinput.write(e.EV_ABS, e.ABS_X, cx)
    _uinput.write(e.EV_ABS, e.ABS_Y, cy)
    _uinput.syn()


def click(button: str = "left") -> None:
    from evdev import ecodes as e
    _ensure_pointer()
    code = _BUTTON_CODES.get(button.lower())
    if code is None:
        raise InputError(f"unknown button: {button}")
    _uinput.write(e.EV_KEY, code, 1); _uinput.syn()
    time.sleep(0.02)
    _uinput.write(e.EV_KEY, code, 0); _uinput.syn()


def click_at(x: int, y: int, button: str = "left") -> None:
    move(x, y)
    # Brief settle so the compositor processes the move before the button event.
    time.sleep(0.03)
    click(button)


def type_text(text: str) -> None:
    if not text:
        return
    _run_ydotool("type", text)


# Aliases mapping our normalized lowercase names to ydotool 0.1.x's expected key names.
_KEY_ALIASES: dict[str, str] = {
    "return": "Enter", "enter": "Enter",
    "esc": "Escape", "escape": "Escape",
    "del": "Delete", "delete": "Delete",
    "ins": "Insert", "insert": "Insert",
    "pgup": "PageUp", "pageup": "PageUp",
    "pgdown": "PageDown", "pgdn": "PageDown", "pagedown": "PageDown",
    "space": "Space",
    "tab": "Tab", "backspace": "BackSpace",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "home": "Home", "end": "End",
    "super": "Super", "meta": "Super", "win": "Super",
    "ctrl": "ctrl", "control": "ctrl",
    "alt": "alt", "shift": "shift",
}


def _normalize_hotkey(hotkey: str) -> str:
    """Translate a hotkey string like 'ctrl+enter' into ydotool 0.1.x's syntax."""
    parts = [p.strip() for p in hotkey.replace(" ", "").split("+") if p.strip()]
    if not parts:
        raise InputError(f"empty hotkey: {hotkey!r}")
    out: list[str] = []
    for p in parts:
        low = p.lower()
        if low in _KEY_ALIASES:
            out.append(_KEY_ALIASES[low])
        elif len(p) == 1:
            # single character (letter, digit, punctuation) — ydotool handles it
            out.append(p)
        elif low.startswith("f") and low[1:].isdigit():
            out.append(low.upper())  # f1 -> F1
        else:
            # pass through capitalized — let ydotool decide
            out.append(p)
    return "+".join(out)


def press(hotkey: str) -> None:
    """Press a key combination like 'ctrl+f' or 'super+space' or just 'enter'."""
    _run_ydotool("key", _normalize_hotkey(hotkey))


def scroll(dy: int) -> None:
    """Vertical scroll in wheel notches. Positive dy = scroll down."""
    from evdev import ecodes as e
    _ensure_pointer()
    # REL_WHEEL: +1 = up, -1 = down. Invert so positive dy scrolls down.
    notches = -int(dy)
    if notches == 0:
        return
    _uinput.write(e.EV_REL, e.REL_WHEEL, notches)
    _uinput.syn()


def available() -> tuple[bool, str]:
    """Quick health check for input injection. Returns (ok, message)."""
    import os
    if not os.path.exists("/dev/uinput"):
        return False, "/dev/uinput missing — load the uinput kernel module"
    if not os.access("/dev/uinput", os.W_OK):
        return (
            False,
            "/dev/uinput not writable for current user — grant via ACL: "
            "sudo setfacl -m u:$USER:rw /dev/uinput",
        )
    try:
        import evdev  # noqa: F401
    except ImportError:
        return False, "python-evdev not installed (pip install evdev)"
    try:
        proc = subprocess.run(["ydotool", "--help"], capture_output=True, timeout=3)
        if b"Usage" not in proc.stdout and b"Usage" not in proc.stderr:
            return False, "ydotool not callable (needed for keyboard)"
    except FileNotFoundError:
        return False, "ydotool not installed (needed for keyboard)"
    return True, "ok"

