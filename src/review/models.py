"""Domain models for review generation.

These containers represent the request to the LLM, its metadata, and the final
result.  All fields are type‑annotated and immutable (frozen dataclasses) to keep
the data flow clear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ReviewRequest:
    """Parameters sent to the LLM to generate a review.

    Attributes
    ----------
    timeline: "TimelineResult"
        The timeline produced by the previous stage.
    review_style: str
        Identifier for the style/template to use (e.g., ``youtube_long``).
    language: str
        Language code (ISO‑639) for the output.
    target_duration: int | None
        Desired script length in seconds – optional hint for the model.
    prompt_template: str
        Fully rendered prompt that will be sent to Gemini.
    """

    timeline: "TimelineResult"
    review_style: str
    language: str
    target_duration: Optional[int]
    prompt_template: str


@dataclass(frozen=True)
class ReviewMetadata:
    """Metadata about the generated review."""

    total_words: int = 0
    estimated_duration: float = 0.0
    model_name: str = "ai"
    processing_time: float = 0.0
    style: str = ""
    language: str = "vi"
    target_duration: Optional[int] = None
    word_count: int = 0
    generation_duration_ms: float = 0.0


@dataclass(frozen=True)
class ReviewResult:
    """Final output of the review generator.

    Attributes
    ----------
    title: str
        The review title (e.g., video title).
    hook: str
        Opening hook – a short intriguing sentence.
    script: str
        Full review script.
    metadata: ReviewMetadata
        Generation metadata.
    """

    title: str
    hook: str
    script: str
    metadata: ReviewMetadata
