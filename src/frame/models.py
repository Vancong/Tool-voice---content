"""Domain models for frame extraction.

These lightweight containers hold metadata about the extracted frames and the
overall extraction result.  The actual image files are stored on disk – the
objects only contain the path and basic dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Frame:
    """Metadata for a single extracted frame.

    Attributes
    ----------
    scene_index: int
        Index of the source scene (0‑based).
    frame_index: int
        Index of the frame **within the scene** (0‑based).
    timestamp: float
        Timestamp in seconds from the start of the video.
    image_path: Path
        Path to the saved JPEG image.
    width: int
        Width of the stored image (after resizing).
    height: int
        Height of the stored image (after resizing).
    """

    scene_index: int
    frame_index: int
    timestamp: float
    image_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class FrameExtractionResult:
    """Result container returned by a frame‑extraction provider.

    Attributes
    ----------
    frames: List[Frame]
        All extracted frames across scenes.
    total_frames: int
        Length of ``frames`` – convenience field.
    processing_time: float
        Total time spent (seconds) for the extraction operation.
    """

    frames: List[Frame]
    total_frames: int
    processing_time: float

    @staticmethod
    def empty() -> "FrameExtractionResult":
        return FrameExtractionResult(frames=[], total_frames=0, processing_time=0.0)
