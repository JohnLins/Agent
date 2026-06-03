"""App launch / hide via XDG .desktop entries and KWin."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _find_desktop_file(name: str) -> Path | None:
    """Search XDG_DATA_DIRS for `name.desktop` (case-insensitive match on Name= or filename)."""
    candidates_dirs: list[Path] = []
    data_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
    home_data = Path.home() / ".local/share"
    for d in [str(home_data), *data_dirs]:
        p = Path(d) / "applications"
        if p.is_dir():
            candidates_dirs.append(p)

    name_l = name.lower()
    for d in candidates_dirs:
        for f in d.glob("*.desktop"):
            if f.stem.lower() == name_l:
                return f
    # Fall back: scan Name= field
    for d in candidates_dirs:
        for f in d.glob("*.desktop"):
            try:
                with open(f) as fh:
                    for line in fh:
                        if line.startswith("Name=") and line[5:].strip().lower() == name_l:
                            return f
            except OSError:
                continue
    return None


def open_app(name: str) -> dict:
    """Launch `name`. Try .desktop file first, then PATH binary. Returns info dict."""
    desktop = _find_desktop_file(name)
    if desktop is not None:
        launcher = shutil.which("gtk-launch") or shutil.which("gio")
        if launcher and launcher.endswith("gtk-launch"):
            subprocess.Popen([launcher, desktop.stem],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        elif launcher:
            subprocess.Popen([launcher, "launch", str(desktop)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        else:
            # Fallback: parse Exec= line
            exec_line = ""
            with open(desktop) as fh:
                for line in fh:
                    if line.startswith("Exec="):
                        exec_line = line[5:].strip()
                        break
            cmd = [tok for tok in exec_line.split() if not tok.startswith("%")]
            subprocess.Popen(cmd,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        return {"launched": str(desktop), "via": "desktop_entry"}

    # Try as a PATH binary
    binary = shutil.which(name)
    if binary:
        subprocess.Popen([binary],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        return {"launched": binary, "via": "binary"}

    raise FileNotFoundError(f"no .desktop entry or PATH binary named {name!r}")


def hide_app(app_class: str) -> int:
    """Minimize all windows whose resourceClass matches `app_class`. Returns count hidden."""
    from . import windows as win_mod
    script = (
        "var n = 0;"
        "var wins = (typeof workspace.windowList === 'function') "
        "? workspace.windowList() : workspace.clientList();"
        "for (var i=0; i<wins.length; i++) {"
        f" if (String(wins[i].resourceClass).toLowerCase() === '{app_class.lower()}') {{"
        "   wins[i].minimized = true; n++;"
        " }"
        "}"
        "print('COMPUTRR_HIDDEN=' + n);"
    )
    out = win_mod._run_kwin_script(script)
    for line in out.splitlines():
        if "COMPUTRR_HIDDEN=" in line:
            try:
                return int(line.split("=", 1)[1].strip())
            except (ValueError, IndexError):
                pass
    return 0
