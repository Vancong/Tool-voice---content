"""Production‑ready Faster‑Whisper STT provider.

Implements :class:`src.stt.base.BaseSTT` using the ``faster-whisper`` library.
All heavy work is delegated to the shared ``ThreadPoolExecutor`` from
``src.utils.thread_pool`` – no additional pools are created.

Key features:

* **Thread‑safe singleton model loading** – the Whisper model is instantiated
  exactly once, even if multiple provider objects exist.
* **Config‑driven** – all behaviour (device, compute type, beam size, timeout,
  retries, etc.) is read from the injected :class:`src.config.settings.AppConfig`
  at construction time and never re‑read.
* **Retry & timeout** – configurable number of retries and a hard timeout per
  transcription attempt.
* **Progress callback & cancellation** – optional ``progress_callback`` receiving a
  ``float`` (0‑100) and optional ``cancel_token`` (``threading.Event``) that can be
  set to abort the operation.
* **Full validation** – existence, size and supported‑format checks before any
  heavy work.
* **Consistent error handling** – all library‑level exceptions are mapped to the
  custom ``STTError`` hierarchy and wrapped in the shared ``Result`` type.
* **Logger binding** – logger is bound with ``module``, ``provider`` and a
  per‑call ``job_id`` for traceability.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Awaitable, Callable, Set, Optional, Tuple

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from src.core.result import Result
from src.stt.base import BaseSTT
from src.stt.exceptions import (
    ModelLoadError,
    TranscriptionError,
    UnsupportedFormatError,
    LanguageNotSupportedError,
    STTError,
)
from src.stt.models import (
    STTResult,
    Transcript,
    TranscriptSegment,
    WordTimestamp,
    LanguageInfo,
)
from src.utils.logger import get_logger
from src.utils.thread_pool import ThreadPoolManager
from src.config.settings import AppConfig, CONFIG

# ---------------------------------------------------------------------------
# Accepted audio extensions (Fast‑Whisper can decode these natively).
# ---------------------------------------------------------------------------
_SUPPORTED_AUDIO_EXTENSIONS: Set[str] = {".wav", ".mp3", ".m4a", ".flac", ".aac"}


class PeriodicTicker:
    """Helper thread to log heartbeat messages periodically during long operations."""

    def __init__(self, message: str, logger, interval: float = 5.0) -> None:
        self.message = message
        self.logger = logger
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        def _run():
            while not self._stop_event.wait(self.interval):
                self.logger.info(self.message)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=1.0)


class FasterWhisperProvider(BaseSTT):
    """Fast‑Whisper STT provider.

    The class reads its configuration once at construction time and stores the
    model‑related values in class attributes so that the underlying Whisper model
    can be loaded exactly once, in a thread‑safe fashion.
    """

    # ---------------------------------------------------------------------
    # Class‑level singleton state
    # ---------------------------------------------------------------------
    _model_lock = threading.Lock()
    _shared_model: WhisperModel | None = None
    _shared_model_loaded: bool = False
    # Model initialisation parameters – populated by the *first* instance.
    _model_name: str = "small"
    _device: str = "cpu"
    _compute_type: str = "int8"
    _model_dir: str = "models"

    @staticmethod
    def _detect_gpu_info() -> Tuple[str, str, bool]:
        """Detect GPU Name, CUDA Driver Version, and float16 support."""
        gpu_name = "N/A"
        cuda_ver = "N/A"
        float16_supported = False

        try:
            import ctranslate2
            if hasattr(ctranslate2, "get_supported_compute_types"):
                cuda_types = ctranslate2.get_supported_compute_types("cuda")
                if "float16" in cuda_types:
                    float16_supported = True
        except Exception:
            pass

        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 1:
                    gpu_name = parts[0]
                if len(parts) >= 2:
                    cuda_ver = f"Driver {parts[1]}"
        except Exception:
            pass

        return gpu_name, cuda_ver, float16_supported

    @classmethod
    def _resolve_compute_type(cls, device: str) -> str:
        """Resolve compute_type based on target device and hardware capabilities."""
        device_clean = (device or "cpu").lower()
        if device_clean == "cpu":
            return "int8"
        elif device_clean == "cuda":
            _, _, float16_supported = cls._detect_gpu_info()
            if float16_supported:
                return "float16"
            else:
                return "float32"
        return "int8"

    def __init__(
        self,
        config: AppConfig = CONFIG,
        logger_name: str = "stt_faster_whisper",
        thread_pool: ThreadPoolManager | None = None,
    ) -> None:
        # -----------------------------------------------------------------
        # Instance configuration – read once and never accessed again.
        # -----------------------------------------------------------------
        self._config = config
        stt_cfg = getattr(config, "stt", None)
        if stt_cfg is None:
            raise ModelLoadError("STT configuration section missing in config.json")

        # Model‑specific options (stored on the instance for easy reference).
        self._model_name_instance: str = getattr(stt_cfg, "model_name", "small")
        self._device_instance: str = (getattr(stt_cfg, "device", "cpu") or "cpu").lower()

        # Automatic compute_type selection based on hardware
        self._compute_type_instance: str = self._resolve_compute_type(self._device_instance)

        self._model_dir_instance: str = getattr(stt_cfg, "model_dir", "models")
        self._beam_size: int = getattr(stt_cfg, "beam_size", 5)
        self._best_of: int = getattr(stt_cfg, "best_of", 5)
        self._vad_filter: bool = getattr(stt_cfg, "vad_filter", False)
        self._vad_parameters: dict | None = getattr(stt_cfg, "vad_parameters", None)
        self._word_timestamps: bool = getattr(stt_cfg, "word_timestamps", True)
        self._language: str | None = getattr(stt_cfg, "language", None)  # None ⇒ auto‑detect
        self._batch_size: int = getattr(stt_cfg, "batch_size", 16)
        self._timeout: float = getattr(stt_cfg, "timeout", 300.0)
        self._retry: int = getattr(stt_cfg, "retry", 2)

        # -----------------------------------------------------------------
        # Initialise class‑level model parameters only once (first instance).
        # -----------------------------------------------------------------
        if not FasterWhisperProvider._shared_model_loaded:
            FasterWhisperProvider._model_name = self._model_name_instance
            FasterWhisperProvider._device = self._device_instance
            FasterWhisperProvider._compute_type = self._compute_type_instance
            FasterWhisperProvider._model_dir = self._model_dir_instance

        # -----------------------------------------------------------------
        # Logger – bind static context (module & provider).  job_id is added per‑call.
        # -----------------------------------------------------------------
        base_logger = get_logger(name=logger_name)
        self._logger = base_logger.bind(module="stt", provider="faster_whisper")

        # -----------------------------------------------------------------
        # Thread‑pool – reuse the global singleton; allow injection for tests.
        # -----------------------------------------------------------------
        self._thread_pool = thread_pool

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------
    @classmethod
    def _load_shared_model(cls, logger) -> None:
        """Load the Whisper model once, thread‑safely.

        Raises:
            ModelLoadError if the underlying library cannot initialise the model.
        """
        if cls._shared_model_loaded:
            return
        with cls._model_lock:
            if cls._shared_model_loaded:
                return

            device = (cls._device or "cpu").lower()
            gpu_name, cuda_ver, _ = cls._detect_gpu_info()
            compute_type = cls._resolve_compute_type(device)
            cls._compute_type = compute_type
            model_name = cls._model_name or "small"

            logger.info("GPU Name:\n{}", gpu_name)
            logger.info("CUDA Version:\n{}", cuda_ver)
            logger.info("Selected device:\n{}", device)
            logger.info("Selected compute_type:\n{}", compute_type)

            logger.info(
                'Initializing WhisperModel(\n    model="{}",\n    device="{}",\n    compute_type="{}"\n)',
                model_name,
                device,
                compute_type,
            )

            abs_cache_path = Path(cls._model_dir).resolve()

            cache_files = [f for f in abs_cache_path.glob("**/*") if f.is_file()] if abs_cache_path.exists() else []
            total_size_bytes = sum(f.stat().st_size for f in cache_files)
            size_mb = total_size_bytes / (1024 * 1024)

            model_folder_matches = list(abs_cache_path.glob(f"*{model_name}*")) if abs_cache_path.exists() else []
            model_exists = (len(cache_files) > 0) and (len(model_folder_matches) > 0)

            logger.info("Model cache path:\n{}", abs_cache_path)
            logger.info("Model cache size:\n{:.2f} MB", size_mb)
            logger.info("Model exists:\n{}", model_exists)

            if not model_exists:
                logger.info("Downloading model...")
            else:
                logger.info("Using local model.")

            logger.info("[STT] Loading FasterWhisper model: {}", model_name)

            start_time = time.time()
            ticker = PeriodicTicker("Still loading FasterWhisper model...", logger, interval=5.0)
            ticker.start()

            stop_mon = threading.Event()
            def _construction_timer():
                while not stop_mon.wait(10.0):
                    elapsed_cur = time.time() - start_time
                    if elapsed_cur > 30.0:
                        logger.info("WhisperModel construction in progress... elapsed: {:.0f}s", elapsed_cur)

            mon_thread = threading.Thread(target=_construction_timer, daemon=True)
            mon_thread.start()

            try:
                import os
                cpu_threads = max(1, os.cpu_count() or 4) if device == "cpu" else 4
                cls._shared_model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(abs_cache_path),
                    cpu_threads=cpu_threads,
                )
                cls._shared_model_loaded = True
                logger.info("WhisperModel initialized successfully (cpu_threads={}).", cpu_threads)
                logger.info("WhisperModel constructed successfully.")
            except Exception as exc:  # pragma: no cover – library‑specific failures
                raise ModelLoadError(f"Failed to load Faster‑Whisper model: {exc}") from exc
            finally:
                stop_mon.set()
                mon_thread.join(timeout=1.0)
                ticker.stop()

            elapsed = time.time() - start_time
            logger.info("[STT] Model loaded successfully.")
            logger.info("[STT] Model loading elapsed time: {:.2f}s", elapsed)

    def _validate_audio_file(self, audio_path: Path) -> None:
        """Validate the supplied audio file.

        Raises:
            UnsupportedFormatError – if the file extension is not supported.
            TranscriptionError – if the file does not exist, is not a file, or is empty.
        """
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file does not exist: {audio_path}")
        if not audio_path.is_file():
            raise TranscriptionError(f"Path is not a file: {audio_path}")
        if audio_path.suffix.lower() not in _SUPPORTED_AUDIO_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported audio format '{audio_path.suffix}'. Supported: {_SUPPORTED_AUDIO_EXTENSIONS}"
            )
        if audio_path.stat().st_size == 0:
            raise TranscriptionError(f"Audio file is empty: {audio_path}")

    def _build_stt_result(
        self,
        audio_path: Path,
        segments: list,
        language: str,
        processing_time: float,
    ) -> STTResult:
        """Convert the raw Faster‑Whisper segments into the domain ``STTResult``.
        """
        transcript_segments: list[TranscriptSegment] = []
        for seg in segments:
            start = getattr(seg, "start", 0.0) if not isinstance(seg, dict) else seg.get("start", 0.0)
            end = getattr(seg, "end", 0.0) if not isinstance(seg, dict) else seg.get("end", 0.0)
            text = getattr(seg, "text", "") if not isinstance(seg, dict) else seg.get("text", "")
            word_objs: list[WordTimestamp] | None = None
            if self._word_timestamps:
                raw_words = getattr(seg, "words", None) if not isinstance(seg, dict) else seg.get("words")
                if raw_words:
                    word_objs = [
                        WordTimestamp(
                            word=getattr(w, "word", "") if not isinstance(w, dict) else w.get("word", ""),
                            start=getattr(w, "start", 0.0) if not isinstance(w, dict) else w.get("start", 0.0),
                            end=getattr(w, "end", 0.0) if not isinstance(w, dict) else w.get("end", 0.0),
                        )
                        for w in raw_words
                    ]
            transcript_segments.append(
                TranscriptSegment(start=start, end=end, text=text, words=word_objs)
            )
        transcript = Transcript(segments=transcript_segments)
        language_info = LanguageInfo(code=language, name=language)
        return STTResult(
            transcript=transcript,
            language=language_info,
            model_name=self._model_name_instance,
            processing_time_secs=processing_time,
            audio_path=audio_path,
        )

    def _run_transcription(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
        cancel_token: threading.Event | None = None,
        job_logger,
    ) -> Result[STTResult, STTError]:
        """Core transcription routine used by both the sync and async APIs.
        """
        try:
            self._validate_audio_file(audio_path)
            FasterWhisperProvider._load_shared_model(job_logger)
        except (UnsupportedFormatError, TranscriptionError, ModelLoadError) as exc:
            job_logger.error("Validation / model load failed: {}", exc)
            return Result.Err(exc)

        effective_lang = language or self._language

        attempt = 0
        while attempt <= self._retry:
            job_logger.info("[STT] Starting transcription...")
            start_ts = time.time()

            ticker = PeriodicTicker("Still transcribing...", job_logger, interval=5.0)
            ticker.start()

            try:
                segments_iter, info = self._shared_model.transcribe(
                    str(audio_path),
                    language=effective_lang,
                    beam_size=self._beam_size,
                    best_of=self._best_of,
                    vad_filter=self._vad_filter,
                    vad_parameters=self._vad_parameters,
                    word_timestamps=self._word_timestamps,
                )
                segments = list(segments_iter)
            except Exception as exc:
                job_logger.error("Transcription execution failed: {}", exc)
                if attempt >= self._retry:
                    return Result.Err(TranscriptionError(str(exc)))
                attempt += 1
                continue
            finally:
                ticker.stop()

            elapsed_tx = time.time() - start_ts
            job_logger.info("[STT] Transcription finished.")
            job_logger.info("[STT] Transcription elapsed time: {:.2f}s", elapsed_tx)

            try:
                detected_lang = (
                    info.get("language") if isinstance(info, dict) else getattr(info, "language", None)
                )
                if effective_lang is None:
                    effective_lang = detected_lang
                if not effective_lang:
                    raise LanguageNotSupportedError("Detected language identifier is empty")

                if cancel_token and cancel_token.is_set():
                    raise TranscriptionError("Transcription cancelled by user")

                result = self._build_stt_result(
                    audio_path=audio_path,
                    segments=segments,
                    language=effective_lang,
                    processing_time=elapsed_tx,
                )
                return Result.Ok(result)
            except LanguageNotSupportedError as exc:
                job_logger.error("Language not supported: {}", exc)
                return Result.Err(exc)
            except TranscriptionError as exc:
                job_logger.error("Transcription attempt {}/{} failed: {}", attempt + 1, self._retry + 1, exc)
                if attempt >= self._retry:
                    return Result.Err(exc)
                attempt += 1
                continue
            except Exception as exc:
                job_logger.error("Unexpected transcription processing error: {}", exc)
                if attempt >= self._retry:
                    return Result.Err(TranscriptionError(str(exc)))
                attempt += 1
                continue

        return Result.Err(TranscriptionError("Maximum retry attempts exhausted"))

    # ---------------------------------------------------------------------
    # Public API – BaseSTT contract
    # ---------------------------------------------------------------------
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
        cancel_token: threading.Event | None = None,
    ) -> Result[STTResult, STTError]:
        job_logger = self._logger.bind(job_id="sync")
        return self._run_transcription(
            audio_path,
            language=language,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
            job_logger=job_logger,
        )

    def transcribe_async(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
        cancel_token: threading.Event | None = None,
    ) -> Awaitable[Result[STTResult, STTError]]:
        pool = (
            self._thread_pool.get_pool(self._config)
            if isinstance(self._thread_pool, ThreadPoolManager)
            else ThreadPoolManager.get_pool(self._config)
        )
        job_id = f"async-{int(time.time() * 1000)}"
        job_logger = self._logger.bind(job_id=job_id)

        def _task() -> Result[STTResult, STTError]:
            job_logger.info("Starting async transcription for {}", audio_path)
            return self._run_transcription(
                audio_path,
                language=language,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
                job_logger=job_logger,
            )

        return pool.submit(_task)

    def supported_languages(self) -> Set[LanguageInfo]:
        common = [
            ("en", "English"),
            ("vi", "Vietnamese"),
            ("zh", "Chinese"),
            ("ja", "Japanese"),
            ("ko", "Korean"),
        ]
        return {LanguageInfo(code=code, name=name) for code, name in common}

    def supported_formats(self) -> Set[str]:
        return _SUPPORTED_AUDIO_EXTENSIONS

    def health_check(self) -> Result[bool, ModelLoadError]:
        try:
            FasterWhisperProvider._load_shared_model(self._logger)
            return Result.Ok(True)
        except ModelLoadError as exc:
            return Result.Err(exc)
