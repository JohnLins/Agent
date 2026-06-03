"""OCR via Tesseract (pytesseract). Returns word-level boxes with confidence."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytesseract

from .protocol import Bounds, Element


def recognize_words(image_path: Path, *, lang: str = "eng", min_conf: float = 30.0) -> list[Element]:
    """Run Tesseract on the image and return one Element per word.

    confidence is 0..1 (Tesseract reports 0..100, normalized here).
    """
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

    out: list[Element] = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < min_conf:
            continue
        out.append(
            Element(
                id=f"word_{i}",
                role="text",
                bounds=Bounds(
                    int(data["left"][i]),
                    int(data["top"][i]),
                    int(data["width"][i]),
                    int(data["height"][i]),
                ),
                text=text,
                confidence=conf / 100.0,
                source="ocr",
            )
        )
    return out


def group_into_lines(words: list[Element], y_tol: int = 6) -> list[Element]:
    """Cluster word-elements that share roughly the same baseline into line-elements."""
    if not words:
        return []

    # sort by y, then x
    sorted_words = sorted(words, key=lambda e: (e.bounds.y, e.bounds.x))
    lines: list[list[Element]] = []
    current: list[Element] = [sorted_words[0]]
    current_y = sorted_words[0].bounds.y

    for w in sorted_words[1:]:
        if abs(w.bounds.y - current_y) <= y_tol:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
            current_y = w.bounds.y
    lines.append(current)

    out: list[Element] = []
    for idx, line in enumerate(lines):
        line_sorted = sorted(line, key=lambda e: e.bounds.x)
        x0 = min(e.bounds.x for e in line_sorted)
        y0 = min(e.bounds.y for e in line_sorted)
        x1 = max(e.bounds.x + e.bounds.w for e in line_sorted)
        y1 = max(e.bounds.y + e.bounds.h for e in line_sorted)
        text = " ".join(e.text for e in line_sorted)
        conf = sum(e.confidence for e in line_sorted) / len(line_sorted)
        out.append(
            Element(
                id=f"line_{idx}",
                role="line",
                bounds=Bounds(x0, y0, x1 - x0, y1 - y0),
                text=text,
                confidence=conf,
                source="ocr",
            )
        )
    return out
