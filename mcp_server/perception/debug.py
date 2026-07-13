"""Shared image annotation helpers for perception diagnostics."""

import cv2
import numpy as np


def draw_bboxes(
    img: np.ndarray,
    bboxes: list[list[int]],
    labels: list[str] | None = None,
) -> np.ndarray:
    out = img.copy()
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 128, 255),
    ]
    for index, bbox in enumerate(bboxes):
        color = colors[index % len(colors)]
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.circle(out, ((xmin + xmax) // 2, (ymin + ymax) // 2), 4, color, -1)
        if labels and index < len(labels):
            cv2.putText(out, labels[index], (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out
