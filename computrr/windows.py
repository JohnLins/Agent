"""Window enumeration and focus via KWin scripting over D-Bus (Plasma 6 / Wayland).

KWin's JS engine cannot do file I/O, so scripts communicate back via `print()` which lands
in the user journal as 'js: <text>'. We tail the journal after running the script and pluck
out lines tagged with our markers.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time


_KWIN_LIST_SCRIPT = r"""
var out = [];
var wins = (typeof workspace.windowList === "function")
    ? workspace.windowList()
    : workspace.clientList();
for (var i = 0; i < wins.length; i++) {
    var w = wins[i];
    try {
        var g = w.frameGeometry || w.geometry;
        out.push({
            id: String(w.internalId || w.windowId || w.resourceClass),
            title: String(w.caption || ""),
            app: String(w.resourceClass || w.resourceName || ""),
            pid: w.pid || 0,
            x: g.x, y: g.y, w: g.width, h: g.height,
            active: !!w.active,
            minimized: !!w.minimized,
            on_desktop: !!w.onCurrentDesktop,
        });
    } catch (e) {}
}
print("COMPUTRR_WINDOWS=" + JSON.stringify(out));
"""


def _gdbus_call(dest: str, path: str, method: str, *args: str) -> str:
    proc = subprocess.run(
        ["gdbus", "call", "--session", "--dest", dest, "--object-path", path, "--method", method, *args],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gdbus call {method} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _journal_since(timestamp: str) -> str:
    proc = subprocess.run(
        ["journalctl", "--user", "--since", timestamp, "-o", "cat", "--no-pager"],
        capture_output=True, text=True, timeout=5,
    )
    return proc.stdout


def _run_kwin_script(script_text: str, marker: str) -> str:
    """Load + run a KWin JS script; return the substring of the journal output starting
    at the first occurrence of `marker`, or empty string if not found."""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, dir="/tmp") as fh:
        fh.write(script_text)
        script_path = fh.name

    try:
        plugin_name = f"computrr_{int(time.time() * 1000)}_{os.getpid()}"
        since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        try:
            load_out = _gdbus_call(
                "org.kde.KWin", "/Scripting",
                "org.kde.kwin.Scripting.loadScript",
                script_path, plugin_name,
            )
        except RuntimeError:
            load_out = _gdbus_call(
                "org.kde.KWin", "/Scripting",
                "org.kde.kwin.Scripting.loadScript",
                script_path,
            )

        digits = "".join(ch for ch in load_out if ch.isdigit())
        if not digits:
            raise RuntimeError(f"could not parse script id from {load_out!r}")
        script_id = digits

        _gdbus_call(
            "org.kde.KWin", f"/Scripting/Script{script_id}",
            "org.kde.kwin.Script.run",
        )
        time.sleep(0.2)

        for unload_arg in (plugin_name, script_path):
            try:
                _gdbus_call("org.kde.KWin", "/Scripting",
                            "org.kde.kwin.Scripting.unloadScript", unload_arg)
                break
            except RuntimeError:
                continue

        output = _journal_since(since)
        for line in reversed(output.splitlines()):
            idx = line.find(marker)
            if idx >= 0:
                return line[idx:]
        return ""
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def list_windows() -> list[dict]:
    """Return all visible windows with id/title/app/pid/x/y/w/h/active/minimized."""
    line = _run_kwin_script(_KWIN_LIST_SCRIPT, "COMPUTRR_WINDOWS=")
    if not line:
        return []
    payload = line[len("COMPUTRR_WINDOWS="):]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return []


def focus_window(window_id: str) -> bool:
    """Activate the window with the given internalId."""
    safe = window_id.replace("'", "")
    script = (
        "var wins = (typeof workspace.windowList === 'function') "
        "? workspace.windowList() : workspace.clientList();"
        "var ok = false;"
        "for (var i = 0; i < wins.length; i++) {"
        f"  if (String(wins[i].internalId) === '{safe}') {{"
        "    workspace.activeWindow = wins[i]; ok = true; break;"
        "  }"
        "}"
        "print('COMPUTRR_FOCUS=' + (ok ? 'ok' : 'miss'));"
    )
    line = _run_kwin_script(script, "COMPUTRR_FOCUS=")
    return line == "COMPUTRR_FOCUS=ok"


def active_window() -> dict | None:
    """Return the currently active window's info (or None)."""
    for w in list_windows():
        if w.get("active"):
            return w
    return None


def available() -> tuple[bool, str]:
    """Health check: is KWin reachable on the session bus?"""
    try:
        _gdbus_call("org.kde.KWin", "/KWin", "org.kde.KWin.currentDesktop")
    except Exception as e:
        return False, f"KWin D-Bus unreachable: {e}"
    return True, "ok"
