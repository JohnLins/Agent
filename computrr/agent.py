"""Bedrock agent loop. Gives Claude a tool set built from Computrr's primitives, then
lets it drive the desktop to complete a task. Uses the Converse API with tool_use."""

from __future__ import annotations

import os
from typing import Any

import boto3
from dotenv import load_dotenv

from . import apps, capture, input as kbm, observe, tokenize, windows

load_dotenv()

_DEFAULT_MODEL = os.getenv("MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
_DEFAULT_REGION = os.getenv("AWS_REGION", "us-east-1")


# Tool definitions for the Bedrock Converse API. Names + descriptions are what Claude sees.
TOOLS = [
    {
        "toolSpec": {
            "name": "tokenize_screen",
            "description": (
                "Take a screenshot and return a structured list of UI elements on screen "
                "(text lines, detected boxes/buttons/fields). Each element has an id, role, "
                "text, and bounds {x,y,w,h}. Use this to perceive what's on screen before acting."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}},
        }
    },
    {
        "toolSpec": {
            "name": "click_text",
            "description": "Click the on-screen element whose text matches the given string (substring match).",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to find and click."},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
                },
                "required": ["text"],
            }},
        }
    },
    {
        "toolSpec": {
            "name": "click_xy",
            "description": "Click at exact screen coordinates (use only when you have pixel coords).",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"}, "y": {"type": "integer"},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
                },
                "required": ["x", "y"],
            }},
        }
    },
    {
        "toolSpec": {
            "name": "type_text",
            "description": "Type the given text via the keyboard into whatever has focus.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }},
        }
    },
    {
        "toolSpec": {
            "name": "press_hotkey",
            "description": "Press a hotkey combo like 'ctrl+f', 'super+space', 'enter', 'escape'.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"hotkey": {"type": "string"}},
                "required": ["hotkey"],
            }},
        }
    },
    {
        "toolSpec": {
            "name": "open_app",
            "description": "Launch a desktop application by name (e.g. 'konsole', 'firefox').",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }},
        }
    },
    {
        "toolSpec": {
            "name": "list_windows",
            "description": "List all visible windows with their titles, app class, and geometry.",
            "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}},
        }
    },
    {
        "toolSpec": {
            "name": "wait_for_change",
            "description": "After an action, wait until the screen visibly changes (or timeout). Use this to confirm an action took effect before perceiving again.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {
                    "timeout_ms": {"type": "integer", "default": 2000},
                },
                "required": [],
            }},
        }
    },
    {
        "toolSpec": {
            "name": "done",
            "description": "Signal that the task is complete. Provide a short summary of what was accomplished.",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            }},
        }
    },
]


def _tokenize_summary() -> dict:
    payload = tokenize.tokenize_screen()
    # Return a slimmer view to keep the model's context manageable.
    return {
        "width": payload.width,
        "height": payload.height,
        "elements": [
            {"id": e.id, "role": e.role, "text": e.text, "bounds": e.bounds.as_dict()}
            for e in payload.elements
            if e.text or e.role in {"button", "field", "box"}
        ][:200],
    }


def _dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        if name == "tokenize_screen":
            return _tokenize_summary()
        if name == "click_text":
            matches = tokenize.find_text(args["text"])
            if not matches:
                return {"error": f"no element matched text {args['text']!r}"}
            m = min(matches, key=lambda e: e.bounds.w * e.bounds.h)
            kbm.click_at(m.bounds.cx, m.bounds.cy, args.get("button", "left"))
            return {"clicked": m.as_dict()}
        if name == "click_xy":
            kbm.click_at(int(args["x"]), int(args["y"]), args.get("button", "left"))
            return {"clicked": {"x": int(args["x"]), "y": int(args["y"])}}
        if name == "type_text":
            kbm.type_text(args["text"])
            return {"typed": args["text"]}
        if name == "press_hotkey":
            kbm.press(args["hotkey"])
            return {"pressed": args["hotkey"]}
        if name == "open_app":
            return apps.open_app(args["name"])
        if name == "list_windows":
            return {"windows": windows.list_windows()}
        if name == "wait_for_change":
            return observe.wait_for_change(timeout_ms=int(args.get("timeout_ms", 2000)))
        return {"error": f"unknown tool: {name}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


SYSTEM_PROMPT = (
    "You are an agent driving a Linux desktop (KDE Plasma 6, Wayland). "
    "You have tools to perceive the screen (tokenize_screen, list_windows), "
    "act on it (click_text, click_xy, type_text, press_hotkey, open_app), "
    "and verify (wait_for_change). "
    "Workflow: perceive → decide one small action → act → wait_for_change → perceive again. "
    "Prefer click_text over click_xy when possible. "
    "After typing into a text field, ALWAYS tokenize_screen to confirm the text actually landed "
    "in the right place before submitting — focus is easy to lose. "
    "Web app gotchas: in most chat / compose / post boxes (X/Twitter, Discord, Slack, "
    "ChatGPT, Gmail compose), plain Enter inserts a newline. To submit, use "
    "press_hotkey('ctrl+enter'). Plain Enter only submits single-line inputs like "
    "the browser URL bar or simple search fields. "
    "Call `done` when finished."
)


def run(task: str, *, max_steps: int = 10, model_id: str | None = None,
        region: str | None = None, verbose: bool = True) -> dict:
    """Run the Bedrock agent loop until `done` or `max_steps`."""
    client = boto3.client("bedrock-runtime", region_name=region or _DEFAULT_REGION)
    model = model_id or _DEFAULT_MODEL

    messages: list[dict] = [{"role": "user", "content": [{"text": task}]}]
    trace: list[dict] = []

    for step in range(max_steps):
        resp = client.converse(
            modelId=model,
            messages=messages,
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.0},
            toolConfig={"tools": TOOLS},
        )
        msg = resp["output"]["message"]
        messages.append(msg)
        stop = resp.get("stopReason")

        tool_uses = [c["toolUse"] for c in msg["content"] if "toolUse" in c]
        if not tool_uses:
            trace.append({"step": step, "text": "".join(c.get("text", "") for c in msg["content"])})
            return {"done": True, "reason": "no_tool_use", "steps": step + 1, "trace": trace, "final": msg}

        tool_results = []
        for tu in tool_uses:
            name = tu["name"]
            args = tu.get("input", {})
            if verbose:
                print(f"[step {step}] tool={name} args={args}")
            if name == "done":
                trace.append({"step": step, "tool": name, "args": args})
                return {"done": True, "reason": "agent_done", "summary": args.get("summary", ""),
                        "steps": step + 1, "trace": trace}
            result = _dispatch(name, args)
            trace.append({"step": step, "tool": name, "args": args, "result_keys": list(result.keys())})
            tool_results.append({
                "toolResult": {
                    "toolUseId": tu["toolUseId"],
                    "content": [{"json": result}],
                }
            })

        messages.append({"role": "user", "content": tool_results})

        if stop == "end_turn":
            return {"done": True, "reason": "end_turn", "steps": step + 1, "trace": trace}

    return {"done": False, "reason": "max_steps", "steps": max_steps, "trace": trace}
