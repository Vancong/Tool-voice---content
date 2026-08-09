# -*- coding: utf-8 -*-
"""src/stt/models.py — STT domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class WordTimestamp:
    """A single word with its start/end timestamps (in seconds)."""
    word: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    """One segment (sentence / phrase) from the transcript."""
    start: float
    end: float
    text: str
    words: Optional[List[WordTimestamp]] = field(default=None)


@dataclass
class Transcript:
    """Full transcript composed of ordered segments."""
    segments: List[TranscriptSegment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Return the concatenated transcript text."""
        return " ".join(s.text.strip() for s in self.segments)


@dataclass
class LanguageInfo:
    """Detected or requested language."""
    code: str          # ISO-639-1 code, e.g. "en", "vi"
    name: str          # Human-readable name, e.g. "English"


@dataclass
class STTResult:
    """Result returned by a successful STT transcription."""
    transcript: Transcript
    language: LanguageInfo
    model_name: str
    processing_time_secs: float
    audio_path: Path
