"""Domain models for scene detection.

The models are lightweight data containers used by the provider to return a
structured result while keeping the public API type‑safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Scene:
    """A single detected scene.

    Attributes
    ----------
    index: int
        Sequential index of the scene starting at ``0``.
    start_time: float
        Start timestamp in seconds.
    end_time: float
        End timestamp in seconds.
    duration: float
        ``end_time - start_time`` – cached for convenience.
    """

    index: int
    start_time: float
    end_time: float
    duration: float

    @staticmethod
    def from_bounds(index: int, start: float, end: float) -> "Scene":
        return Scene(index=index, start_time=start, end_time=end, duration=end - start)


@dataclass(frozen=True)
class SceneDetectionResult:
    """Result container returned by a scene‑detection provider.

    Attributes
    ----------
    scenes: List[Scene]
        Ordered list of detected scenes.
    total_duration: float
        Total duration of the video (sum of scene durations).
    """

    scenes: List[Scene]
    total_duration: float

    @staticmethod
    def empty() -> "SceneDetectionResult":
        return SceneDetectionResult(scenes=[], total_duration=0.0)
