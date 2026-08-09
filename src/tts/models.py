"""Domain models for the Text‑to‑Speech (TTS) module.

These dataclasses are immutable and fully type‑annotated, matching the style of
the rest of the codebase.  They capture the request parameters, the generated
metadata and the final result (audio file path).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VoiceRequest:
    """Parameters required to synthesize speech.

    Attributes
    ----------
    script: str
        The full text to be spoken.
    language: str
        Language code (ISO‑639) for the TTS engine.
    voice_id: str
        Identifier of the voice model to use.
    speed: float
        Playback speed factor (1.0 = normal).
    pitch: float
        Pitch adjustment factor.
    volume: float
        Volume gain factor.
    """

    script: str
    language: str
    voice_id: str
    speed: float
    pitch: float
    volume: float


@dataclass(frozen=True)
class VoiceMetadata:
    """Metadata about the generated audio.

    Attributes
    ----------
    duration: float
        Length of the audio in seconds.
    sample_rate: int
        Sample rate in Hz.
    provider: str
        Name of the TTS provider (e.g., "CapCut").
    processing_time: float
        Time spent (seconds) generating the audio.
    """

    duration: float
    sample_rate: int
    provider: str
    processing_time: float


@dataclass(frozen=True)
class VoiceResult:
    """Result of a successful TTS synthesis.

    Attributes
    ----------
    audio_path: Path
        Absolute path to the generated audio file (e.g., a WAV or MP3).
    metadata: VoiceMetadata
        Generation metadata.
    """

    audio_path: Path
    metadata: VoiceMetadata
