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
class KeyframeSelectionResult:
    """Structured output produced by the keyframe-selection module.

    Notes:
        keyframe_images are NumPy arrays in RGB channel order.
        artifacts_dir contains optional debug and inspection artifacts.
    """

    video_path: Path
    keyframes: tuple[int, ...]
    keyframe_images: tuple[np.ndarray, ...]
    artifacts_dir: Path