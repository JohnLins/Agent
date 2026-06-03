"""Tokenize a screenshot into a unified element list:
- OCR words and lines (text content)
- CV-detected boxes (buttons, fields, panels — anything with a clean rectangular boundary)

The goal is to give the agent a structured view of "every clickable / readable thing on screen"
without having to interpret raw pixels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import capture, ocr
from .protocol import Bounds, Element, TokenizePayload


def _detect_boxes(
    image_path: Path,
    *,
    min_area: int = 400,
    max_area_frac: float = 0.5,
) -> list[Element]:
    """Use Canny + contour detection to find rectangular UI boundaries."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    h_img, w_img = img.shape
    max_area = int(w_img * h_img * max_area_frac)

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(blurred, 60, 180)
    # Close small gaps so rectangles become contiguous contours.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[Element] = []
    seen: set[tuple[int, int, int, int]] = set()
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # Skip extremely thin slivers (likely noise / text underlines)
        if w < 12 or h < 12:
            continue
        # Skip near-duplicates within a few pixels
        key = (x // 4, y // 4, w // 4, h // 4)
        if key in seen:
            continue
        seen.add(key)
        # Rectangularity: contour area vs bbox area
        rect_ratio = area / float(w * h)
        if rect_ratio < 0.45:
            continue
        boxes.append(
            Element(
                id=f"box_{i}",
                role="box",
                bounds=Bounds(int(x), int(y), int(w), int(h)),
                confidence=float(rect_ratio),
                source="cv",
            )
        )
    return boxes


def _merge_text_into_boxes(boxes: list[Element], lines: list[Element]) -> None:
    """Attach OCR line text to any box that contains it (in-place)."""
    for box in boxes:
        contained = [
            ln.text
            for ln in lines
            if _contains(box.bounds, ln.bounds)
        ]
        if contained:
            box.text = " | ".join(contained)
            # Heuristic role refinement
            if box.bounds.h < 60 and len(contained) == 1 and len(contained[0]) < 40:
                box.role = "button"
            elif box.bounds.h < 50:
                box.role = "field"


def _contains(outer: Bounds, inner: Bounds, slack: int = 4) -> bool:
    return (
        inner.x >= outer.x - slack
        and inner.y >= outer.y - slack
        and inner.x + inner.w <= outer.x + outer.w + slack
        and inner.y + inner.h <= outer.y + outer.h + slack
    )


def tokenize_screen(
    image_path: Path | None = None,
    *,
    include_words: bool = False,
    detect_boxes: bool = True,
) -> TokenizePayload:
    """Capture (or reuse) a screenshot and return a structured element list.

    - lines: OCR text grouped into lines (always included)
    - words: raw word-level OCR (only if include_words=True; verbose)
    - boxes: CV-detected rectangular UI elements with any contained text merged in
    """
    if image_path is None:
        image_path = capture.capture_screen()

    from PIL import Image

    with Image.open(image_path) as im:
        w_img, h_img = im.size

    words = ocr.recognize_words(image_path)
    lines = ocr.group_into_lines(words)

    elements: list[Element] = list(lines)
    if include_words:
        elements.extend(words)
    if detect_boxes:
        boxes = _detect_boxes(image_path)
        _merge_text_into_boxes(boxes, lines)
        elements.extend(boxes)

    # Re-id sequentially for stable indexing across runs of the same screen
    for i, e in enumerate(elements):
        prefix = e.role
        e.id = f"{prefix}_{i}"

    return TokenizePayload(width=w_img, height=h_img, image_path=str(image_path), elements=elements)


def find_text(
    needle: str,
    payload: TokenizePayload | None = None,
    *,
    case_sensitive: bool = False,
) -> list[Element]:
    """Return all elements whose text matches `needle` (substring match)."""
    if payload is None:
        payload = tokenize_screen()
    if not case_sensitive:
        needle_l = needle.lower()
        return [e for e in payload.elements if needle_l in e.text.lower()]
    return [e for e in payload.elements if needle in e.text]
