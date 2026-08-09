"""Domain models for the timeline builder.

These lightweight data containers hold the final movie timeline constructed from
the vision analysis results.  No AI calls are made – the provider simply
re‑structures existing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TimelineEvent:
    """A single event in the movie timeline.

    Attributes
    ----------
    scene_index: int
        Index of the scene (0‑based).
    start_time: float
        Start timestamp of the scene (seconds).  The builder does not compute
        exact timings – they are set to ``0.0`` as a placeholder.
    end_time: float
        End timestamp of the scene (seconds).  Placeholder ``0.0``.
    summary: str
        Textual summary of the scene.
    characters: List[str]
        Key characters appearing in the scene.
    actions: List[str]
        Important actions detected.
    objects: List[str]
        Objects of interest.
    emotion: str
        Dominant emotion.
    """

    scene_index: int
    start_time: float
    end_time: float
    summary: str
    characters: List[str]
    actions: List[str]
    objects: List[str]
    emotion: str


@dataclass(frozen=True)
class MovieTimeline:
    """Container for a full movie timeline.

    Attributes
    ----------
    events: List[TimelineEvent]
        Ordered list of events (sorted by ``scene_index``).
    total_scenes: int
        Number of scenes.
    duration: float
        Approximate total duration – set to ``0.0`` because exact timing is not
        available without the original video timestamps.
    """

    events: List[TimelineEvent]
    total_scenes: int
    duration: float


@dataclass(frozen=True)
class TimelineResult:
    """Result wrapper returned by the timeline builder.

    Attributes
    ----------
    timeline: MovieTimeline
        The assembled timeline.
    processing_time: float
        Time taken (seconds) for the build operation.
    """

    timeline: MovieTimeline
    processing_time: float

    @staticmethod
    def empty() -> "TimelineResult":
        return TimelineResult(
            timeline=MovieTimeline(events=[], total_scenes=0, duration=0.0),
            processing_time=0.0,
        )
