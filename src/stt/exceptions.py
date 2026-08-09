# -*- coding: utf-8 -*-
"""src/stt/exceptions.py — STT exception hierarchy."""

from __future__ import annotations


class STTError(Exception):
    """Base exception for all STT errors."""


class ModelLoadError(STTError):
    """Raised when the Whisper model cannot be loaded."""


class TranscriptionError(STTError):
    """Raised when transcription fails or times out."""


class UnsupportedFormatError(STTError):
    """Raised when the audio file format is not supported."""


class LanguageNotSupportedError(STTError):
    """Raised when the requested language is not recognised."""
