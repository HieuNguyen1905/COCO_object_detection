"""Visualization utilities for object detection results."""

from __future__ import annotations

import random
from typing import Sequence

import cv2
import numpy as np


# A palette of distinguishable colours (BGR) — one per class index.
_PALETTE: list[tuple[int, int, int]] | None = None


def _get_color(class_id: int) -> tuple[int, int, int]:
    """Return a deterministic, visually distinct BGR colour for *class_id*."""
    global _PALETTE
    if _PALETTE is None:
        rng = random.Random(42)
        _PALETTE = [
            (rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255))
            for _ in range(200)
        ]
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray | Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Sequence[float] | None = None,
    class_names: list[str] | None = None,
    score_threshold: float = 0.0,
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes, class names and confidence scores on an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR or RGB format ``(H, W, 3)``.  A copy is made so the
        original is not modified.
    boxes : array-like, shape ``(N, 4)``
        Bounding boxes in **pascal_voc** format ``[xmin, ymin, xmax, ymax]``.
    labels : sequence of int, length N
        Class indices for each box.
    scores : sequence of float, length N, optional
        Confidence scores.  If provided, only detections with
        ``score >= score_threshold`` are drawn.
    class_names : list[str], optional
        Human-readable class names.  ``class_names[label]`` is used as the
        display text.  Falls back to the numeric label if not provided.
    score_threshold : float
        Minimum confidence to draw a detection (default 0.0 = draw all).
    line_thickness : int
        Box outline thickness in pixels.
    font_scale : float
        Font scale for the label text.

    Returns
    -------
    np.ndarray
        Annotated image (same dtype as input).
    """
    img = image.copy()
    boxes = np.asarray(boxes)

    if len(boxes) == 0:
        return img

    for i, (box, label) in enumerate(zip(boxes, labels)):
        score = scores[i] if scores is not None else None
        if score is not None and score < score_threshold:
            continue

        x1, y1, x2, y2 = map(int, box[:4])
        color = _get_color(int(label))

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        # Build label text
        name = class_names[int(label)] if class_names else str(int(label))
        if score is not None:
            text = f"{name} {score:.2f}"
        else:
            text = name

        # Draw label background + text
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        # Label background sits just above the box (or inside if at top edge)
        label_y1 = max(y1 - th - baseline - 4, 0)
        label_y2 = max(y1, th + baseline + 4)
        cv2.rectangle(img, (x1, label_y1), (x1 + tw + 4, label_y2), color, -1)
        cv2.putText(
            img,
            text,
            (x1 + 2, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return img


def save_result(
    image: np.ndarray,
    path: str,
    boxes: np.ndarray | Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Sequence[float] | None = None,
    class_names: list[str] | None = None,
    score_threshold: float = 0.3,
) -> None:
    """Draw detections and save the annotated image to *path*."""
    annotated = draw_detections(
        image, boxes, labels, scores, class_names, score_threshold
    )
    cv2.imwrite(path, annotated)
