"""Shared data shapes — kept close to desktopctl's JSON contract so an agent prompt
written for one can mostly drive the other."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class Bounds:
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


ElementSource = Literal["ocr", "cv", "ax"]


@dataclass
class Element:
    """One thing identified on screen — either OCR text, a CV-detected box, or both merged."""

    id: str
    role: str  # "text" | "line" | "box" | "button" | "field"
    bounds: Bounds
    text: str = ""
    confidence: float = 0.0
    source: ElementSource = "ocr"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "text": self.text,
            "bounds": self.bounds.as_dict(),
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


@dataclass
class TokenizePayload:
    width: int
    height: int
    image_path: str
    elements: list[Element] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "image_path": self.image_path,
            "elements": [e.as_dict() for e in self.elements],
            "element_count": len(self.elements),
        }


def ok(result: Any) -> dict[str, Any]:
    return {"ok": True, "result": result}


def err(code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"ok": False, "error": {"code": code, "message": message}}
    if extra:
        payload["error"].update(extra)
    return payload
