from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrameExtractorResult:
    """Structured output produced by the original SeeDo FrameExtractor.

    Notes:
        keyframe_images are NumPy arrays in RGB channel order.
    """

    keyframes: tuple[int, ...]
    keyframe_images: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class VisualPromptingResult:
    """Structured output produced by the SeeDo visual-prompting stage."""

    annotated_video_path: Path
    track_id_map: dict[int, dict[str, object]]
    key_frame_coordinates: dict[str, list[str]]
    bounding_box_summary: str
    count_diagnostics: dict[str, object]