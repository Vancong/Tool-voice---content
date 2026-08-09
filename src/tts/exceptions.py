"""Exception hierarchy for the Text‑to‑Speech (TTS) module.

All custom exceptions inherit from :class:`TTSError` so the provider can wrap
them in the shared :class:`src.core.result.Result` type.
"""

from __future__ import annotations


class TTSError(Exception):
    """Base class for all TTS‑related errors."""


class VoiceGenerationError(TTSError):
    """Raised when the TTS service fails to generate audio or returns an error."""


class VoiceProviderError(TTSError):
    """Raised for problems communicating with the TTS provider (network, auth, etc.)."""


class TTSAPIError(TTSError):
    """Raised when TTS API returns an error response."""


class RateLimitError(TTSError):
    """Raised when TTS provider rate limit is exceeded."""


class TTSNetworkError(TTSError):
    """Raised when network connection to TTS provider fails."""


class AudioConcatenationError(TTSError):
    """Raised when stitching multi-chunk audio files fails."""

