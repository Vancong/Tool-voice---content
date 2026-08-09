# -*- coding: utf-8 -*-
"""
src/config/settings.py

Defines ``AppConfig`` – the project-wide configuration object – and the
singleton ``CONFIG`` instance that every provider imports as its default.

Configuration is loaded from two sources (in priority order):

1. ``config/config.json``  – static JSON file committed to the repository.
2. Environment variables (and/or a ``.env`` file in the project root).

``pydantic-settings`` merges both sources automatically.  Environment
variables override JSON values; they follow the naming convention
``SECTION__FIELD`` (double underscore as separator), e.g.::

    GEMINI_API_KEY=...       # top-level env var
    STT__MODEL_NAME=tiny     # overrides config.stt.model_name

The ``CONFIG`` singleton is created at module import time so that
``from src.config.settings import AppConfig, CONFIG`` always succeeds with a
fully populated object.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-section models (nested inside AppConfig)
# ---------------------------------------------------------------------------

class VideoConfig(BaseModel):
    supported_formats: List[str] = [".mp4", ".mkv", ".avi", ".mov"]
    default_fps: float = 30.0
    thumbnail_width: int = 320
    thumbnail_height: int = 180


class AudioConfig(BaseModel):
    sample_rate: int = 48000
    channels: int = 2
    output_filename: str = "audio.wav"


class STTConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    provider: str = "faster_whisper"
    model_name: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    best_of: int = 5
    vad_filter: bool = True
    vad_parameters: Optional[dict] = None
    word_timestamps: bool = True
    language: Optional[str] = None
    batch_size: int = 16
    timeout: float = 7200.0  # 2 hours for full movies (1h20-1h30+)
    retry: int = 2
    model_dir: str = "models"


class SceneConfig(BaseModel):
    threshold: float = 30.0
    min_scene_len: int = 15
    retry: int = 1
    timeout: float = 1800.0


class SceneFramesConfig(BaseModel):
    selection_strategy: str = "max"
    max_frames_per_scene: int = 3
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    output_dir: str = "data/frames"
    jpeg_quality: int = 90
    retry: int = 1
    timeout: float = 1800.0


class VisionConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    provider: str = "gemini"
    model_name: str = "gemini-2.0-flash"
    prompt: str = (
        "Analyse the following images and provide a concise summary, "
        "key objects, characters, actions and dominant emotion."
    )
    batch_size: int = 5
    max_batch_size: int = 10
    image_quality: int = 85
    temperature: float = 0.2
    retry: int = 1
    timeout: float = 3600.0


class TimelineConfig(BaseModel):
    max_events: int = 50
    retry: int = 1
    timeout: float = 600.0


class ReviewConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    provider: str = "gemini"
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_tokens: int = 4096
    style: str = "youtube"
    language: str = "vi"
    target_duration: int = 600
    retry: int = 2
    timeout: float = 1800.0


class TTSConfig(BaseModel):
    provider: str = "capcut"
    language: str = "vi"
    voice_id: str = "vi-VN-NamMinhNeural"
    speed: float = 1.0
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: float = 1.0
    output_dir: str = "data/audio"
    retry: int = 2
    timeout: float = 1800.0


class ComposerConfig(BaseModel):
    resolution: str = "1920x1080"
    fps: int = 30
    bitrate: str = "5000k"
    codec: str = "libx264"
    audio_codec: str = "aac"
    output_dir: str = "data/output"
    retry: int = 1
    timeout: float = 7200.0  # 2 hours for high-res video rendering


class LoggingConfig(BaseModel):
    log_file: str = "logs/movie_review.log"
    log_level: str = "INFO"


class ThreadPoolConfig(BaseModel):
    max_workers: int = 8


# ---------------------------------------------------------------------------
# Root config – merges JSON + environment
# ---------------------------------------------------------------------------

class AppConfig(BaseSettings):
    """Project-wide configuration.

    Values come from (in ascending priority order):
    1. Hard-coded defaults in the sub-models.
    2. ``config/config.json`` (loaded manually and injected via
       ``model_validate``).
    3. Environment variables / ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Top-level API keys read directly from environment (not in JSON).
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    capcut_api_key: str = Field(default="", alias="CAPCUT_API_KEY")
    capcut_session_id: str = Field(default="", alias="CAPCUT_SESSION_ID")
    google_sheet_webhook_url: str = Field(default="", alias="GOOGLE_SHEET_WEBHOOK_URL")
    google_sheet_id: str = Field(default="", alias="GOOGLE_SHEET_ID")

    # Nested sections.
    video: VideoConfig = VideoConfig()
    audio: AudioConfig = AudioConfig()
    stt: STTConfig = STTConfig()
    scene: SceneConfig = SceneConfig()
    scene_frames: SceneFramesConfig = SceneFramesConfig()
    vision: VisionConfig = VisionConfig()
    timeline: TimelineConfig = TimelineConfig()
    review: ReviewConfig = ReviewConfig()
    tts: TTSConfig = TTSConfig()
    composer: ComposerConfig = ComposerConfig()
    logging: LoggingConfig = LoggingConfig()
    thread_pool: ThreadPoolConfig = ThreadPoolConfig()

    @classmethod
    def load(cls) -> "AppConfig":
        """Load config from ``config/config.json`` merged with env vars.

        If the JSON file is missing, defaults are used and a warning is
        printed (not raised) so the app can still start.
        """
        json_data: dict = {}
        json_path = Path("config/config.json")
        if json_path.exists():
            try:
                with json_path.open(encoding="utf-8") as fh:
                    json_data = json.load(fh)
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Failed to load config/config.json: {exc}")
        # pydantic-settings will also merge env vars automatically.
        return cls.model_validate(json_data)


# ---------------------------------------------------------------------------
# Module-level singleton – providers import ``CONFIG`` as the default value.
# ---------------------------------------------------------------------------

CONFIG: AppConfig = AppConfig.load()


__all__ = ["AppConfig", "CONFIG"]
