"""Command-line interface. Mirrors desktopctl's subcommand shape so prompts written for
one tool can mostly drive the other.

All commands print JSON to stdout. Exit code 0 on success, 1 on error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import apps, capture, input as kbm, observe, protocol, tokenize, windows


def _print(payload: Any) -> int:
    print(json.dumps(payload, indent=2))
    return 0 if (isinstance(payload, dict) and payload.get("ok", True)) else 1


def _try(fn, *args, **kwargs) -> dict:
    try:
        result = fn(*args, **kwargs)
        return protocol.ok(result)
    except FileNotFoundError as e:
        return protocol.err("TARGET_NOT_FOUND", str(e))
    except PermissionError as e:
        return protocol.err("PERMISSION_DENIED", str(e))
    except Exception as e:
        return protocol.err("INTERNAL", f"{type(e).__name__}: {e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="computrr", description="Linux desktop control")
    sub = p.add_subparsers(dest="cmd", required=True)

    # screen
    s = sub.add_parser("screen", help="Screen capture and tokenization")
    ss = s.add_subparsers(dest="screen_cmd", required=True)
    ssp = ss.add_parser("screenshot", help="Capture a screenshot")
    ssp.add_argument("--out", type=Path, default=None)
    ssp.add_argument("--pointer", action="store_true", help="Include mouse pointer")

    sst = ss.add_parser("tokenize", help="Capture + OCR + CV → structured element list")
    sst.add_argument("--image", type=Path, default=None, help="Use existing PNG instead of capturing")
    sst.add_argument("--words", action="store_true", help="Include raw word elements (verbose)")
    sst.add_argument("--no-boxes", action="store_true", help="Skip CV box detection")

    ssf = ss.add_parser("find", help="Find text on screen")
    ssf.add_argument("--text", required=True)
    ssf.add_argument("--case-sensitive", action="store_true")

    # pointer
    p_ = sub.add_parser("pointer", help="Mouse control")
    pp = p_.add_subparsers(dest="pointer_cmd", required=True)
    pm = pp.add_parser("move"); pm.add_argument("x", type=int); pm.add_argument("y", type=int)
    pc = pp.add_parser("click")
    pc.add_argument("x", type=int, nargs="?"); pc.add_argument("y", type=int, nargs="?")
    pc.add_argument("--text", help="Click center of element matching this text")
    pc.add_argument("--id", help="Click center of element with this id")
    pc.add_argument("--button", default="left", choices=["left", "right", "middle"])
    ps = pp.add_parser("scroll"); ps.add_argument("dy", type=int)

    # keyboard
    k = sub.add_parser("keyboard", help="Keyboard input")
    kk = k.add_subparsers(dest="keyboard_cmd", required=True)
    kt = kk.add_parser("type"); kt.add_argument("text")
    kp = kk.add_parser("press"); kp.add_argument("hotkey")

    # window
    w = sub.add_parser("window", help="Window enumeration / focus")
    ww = w.add_subparsers(dest="window_cmd", required=True)
    ww.add_parser("list")
    ww.add_parser("active")
    wf = ww.add_parser("focus"); wf.add_argument("id")

    # app
    a = sub.add_parser("app", help="App lifecycle")
    aa = a.add_subparsers(dest="app_cmd", required=True)
    ao = aa.add_parser("open"); ao.add_argument("name")
    ah = aa.add_parser("hide"); ah.add_argument("app_class")

    # observe
    o = sub.add_parser("observe", help="Wait for screen change after an action")
    o.add_argument("--timeout", type=int, default=2000)
    o.add_argument("--interval", type=int, default=100)
    o.add_argument("--threshold", type=int, default=8)

    # debug
    d = sub.add_parser("debug", help="Health checks")
    dd = d.add_subparsers(dest="debug_cmd", required=True)
    dd.add_parser("doctor")
    dd.add_parser("ping")

    # agent
    ag = sub.add_parser("agent", help="Run the Bedrock agent loop")
    ag.add_argument("task", help="Natural-language task description")
    ag.add_argument("--max-steps", type=int, default=30)

    return p


def _resolve_click_target(args) -> tuple[int, int]:
    if args.x is not None and args.y is not None:
        return args.x, args.y
    if args.text or args.id:
        payload = tokenize.tokenize_screen()
        if args.id:
            matches = [e for e in payload.elements if e.id == args.id]
        else:
            matches = tokenize.find_text(args.text, payload=payload)
        if not matches:
            raise ValueError(
                f"no element matched {'id=' + args.id if args.id else 'text=' + args.text!r}"
            )
        # Prefer the smallest match (likely most specific)
        m = min(matches, key=lambda e: e.bounds.w * e.bounds.h)
        return m.bounds.cx, m.bounds.cy
    raise ValueError("click requires either x y, or --text, or --id")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "screen":
        if args.screen_cmd == "screenshot":
            return _print(_try(lambda: {"path": str(capture.capture_screen(args.out, include_pointer=args.pointer))}))
        if args.screen_cmd == "tokenize":
            def go():
                payload = tokenize.tokenize_screen(
                    args.image,
                    include_words=args.words,
                    detect_boxes=not args.no_boxes,
                )
                return payload.as_dict()
            return _print(_try(go))
        if args.screen_cmd == "find":
            def go():
                matches = tokenize.find_text(args.text, case_sensitive=args.case_sensitive)
                return {"count": len(matches), "matches": [e.as_dict() for e in matches]}
            return _print(_try(go))

    if args.cmd == "pointer":
        if args.pointer_cmd == "move":
            return _print(_try(lambda: (kbm.move(args.x, args.y), {"x": args.x, "y": args.y})[1]))
        if args.pointer_cmd == "click":
            def go():
                x, y = _resolve_click_target(args)
                kbm.click_at(x, y, args.button)
                return {"x": x, "y": y, "button": args.button}
            return _print(_try(go))
        if args.pointer_cmd == "scroll":
            return _print(_try(lambda: (kbm.scroll(args.dy), {"dy": args.dy})[1]))

    if args.cmd == "keyboard":
        if args.keyboard_cmd == "type":
            return _print(_try(lambda: (kbm.type_text(args.text), {"typed": args.text})[1]))
        if args.keyboard_cmd == "press":
            return _print(_try(lambda: (kbm.press(args.hotkey), {"pressed": args.hotkey})[1]))

    if args.cmd == "window":
        if args.window_cmd == "list":
            return _print(_try(lambda: {"windows": windows.list_windows()}))
        if args.window_cmd == "active":
            return _print(_try(lambda: windows.active_window() or {}))
        if args.window_cmd == "focus":
            return _print(_try(lambda: {"focused": windows.focus_window(args.id)}))

    if args.cmd == "app":
        if args.app_cmd == "open":
            return _print(_try(lambda: apps.open_app(args.name)))
        if args.app_cmd == "hide":
            return _print(_try(lambda: {"hidden_count": apps.hide_app(args.app_class)}))

    if args.cmd == "observe":
        return _print(_try(lambda: observe.wait_for_change(
            timeout_ms=args.timeout, interval_ms=args.interval, threshold=args.threshold,
        )))

    if args.cmd == "debug":
        if args.debug_cmd == "ping":
            return _print(protocol.ok({"pong": True}))
        if args.debug_cmd == "doctor":
            checks = {
                "input_ydotool": kbm.available(),
                "windows_kwin_dbus": windows.available(),
            }
            # Tesseract check
            try:
                import pytesseract
                tess_version = str(pytesseract.get_tesseract_version())
                checks["ocr_tesseract"] = (True, tess_version)
            except Exception as e:
                checks["ocr_tesseract"] = (False, str(e))
            # Spectacle check
            import shutil as _sh
            checks["capture_spectacle"] = (
                bool(_sh.which("spectacle")),
                _sh.which("spectacle") or "not found in PATH",
            )
            ok_all = all(v[0] for v in checks.values())
            return _print({
                "ok": ok_all,
                "result": {k: {"ok": v[0], "info": v[1]} for k, v in checks.items()},
            })

    if args.cmd == "agent":
        from . import agent as agent_mod
        return _print(_try(lambda: agent_mod.run(args.task, max_steps=args.max_steps)))

    return _print(protocol.err("INVALID_ARGUMENT", f"unhandled command: {args.cmd}"))


if __name__ == "__main__":
    sys.exit(main())
