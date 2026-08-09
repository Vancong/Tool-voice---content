"""Domain models for the video composer module.

These dataclasses capture the request parameters, output metadata and result of the
FFmpeg composition step.  All fields are typed and immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VideoComposeRequest:
    """Parameters required to compose the final review video.

    Attributes
    ----------
    original_video: Path
        Path to the source video file.
    frame_result: "src.frame.models.FrameExtractionResult"
        Result of the frame extraction step.
    review_result: "src.review.models.ReviewResult"
        Generated textual review.
    voice_result: "src.tts.models.VoiceResult"
        Synthesized voice audio.
    output_path: Path
        Desired location for the final composed video.
    resolution: str
        Target resolution (e.g., "1920x1080").
    fps: int
        Frames per second for the output video.
    bitrate: str
        Video bitrate (e.g., "4M").
    """

    original_video: Path
    frame_result: Optional[object] = None
    review_result: Optional[object] = None
    voice_result: Optional[object] = None
    output_path: Path = Path("data/output.mp4")
    resolution: str = "1280x720"
    fps: int = 30
    bitrate: str = "4M"


@dataclass(frozen=True)
class ComposeMetadata:
    """Metadata about the composed video.

    Attributes
    ----------
    duration: float
        Length of the output video in seconds.
    resolution: str
        Output resolution.
    fps: int
        Frames per second.
    codec: str
        Video codec used (e.g., "h264").
    processing_time: float
        Time spent (seconds) performing the composition.
    """

    duration: float
    resolution: str
    fps: int
    codec: str
    processing_time: float


@dataclass(frozen=True)
class VideoComposeResult:
    """Result of a successful video composition.

    Attributes
    ----------
    output_video: Path
        Path to the generated MP4 file.
    metadata: ComposeMetadata
        Information about the created video.
    """

    output_video: Path
    metadata: ComposeMetadata

    @staticmethod
    def empty() -> "VideoComposeResult":
        return VideoComposeResult(output_video=Path(""), metadata=ComposeMetadata(0.0, "", 0, "", 0.0))
