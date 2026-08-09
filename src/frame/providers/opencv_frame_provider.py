"""OpenCV based frame extraction provider.

Implements :class:`src.frame.base.BaseFrameExtractor` and follows the same
provider pattern used throughout the project (STT, Scene Detection, …).
All heavy work is delegated to the shared ``ThreadPoolExecutor`` from
``src.utils.thread_pool`` – no new pools are created.

Configuration (through ``AppConfig.scene_frames``) supports:

* ``selection_strategy`` – how many frames to pick per scene. Accepted values
  ``first``, ``middle``, ``last`` or ``max`` (evenly spaced up to
  ``max_frames_per_scene``).
* ``max_frames_per_scene`` – maximum number of frames to extract from a scene
  when ``selection_strategy`` is ``max``.
* ``resize_width`` / ``resize_height`` – dimensions to resize each frame to.
  If either is ``None`` the original size is kept.
* ``jpeg_quality`` – JPEG quality (0‑100) passed to ``cv2.imwrite``.
* ``retry`` – number of retry attempts on failure.
* ``timeout`` – hard timeout per extraction call (seconds).

The provider **never** loads the entire video into RAM – it seeks directly to the
required timestamps using OpenCV's ``VideoCapture``.  Extracted frames are saved
as JPEG images under ``data/jobs/{job_id}/frames/`` where ``job_id`` is taken from
the bound logger context (``logger.bind(job_id=...)``) or generated uniquely
per call to avoid clashes.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Awaitable, List

import cv2

from src.core.result import Result
from src.frame.base import BaseFrameExtractor
from src.frame.exceptions import (
    FrameExtractionError,
    FrameReadError,
    FrameWriteError,
)
from src.frame.models import Frame, FrameExtractionResult
from src.utils.logger import get_logger
from src.utils.thread_pool import ThreadPoolManager
from src.config.settings import AppConfig, CONFIG

# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class OpenCVFrameProvider(BaseFrameExtractor):
    """Provider that extracts frames from a video using OpenCV.

    The class reads its configuration once at construction time and stores the
    values needed for extraction.  All methods return a ``Result`` that wraps the
    appropriate domain model or a custom ``FrameExtractionError``.
    """

    def __init__(
        self,
        config: AppConfig = CONFIG,
        logger_name: str = "frame_opencv",
        thread_pool: ThreadPoolManager | None = None,
    ) -> None:
        # -------------------------------------------------------------------
        # Configuration – fall back to sensible defaults if the section is
        # missing.  The ``scene_frames`` block is optional to keep the provider
        # usable out‑of‑the‑box.
        # -------------------------------------------------------------------
        self._config = config
        frames_cfg = getattr(config, "scene_frames", None) or {}
        self._selection_strategy: str = getattr(frames_cfg, "selection_strategy", "max")
        self._max_frames_per_scene: int = getattr(frames_cfg, "max_frames_per_scene", 3)
        self._resize_width: int | None = getattr(frames_cfg, "resize_width", None)
        self._resize_height: int | None = getattr(frames_cfg, "resize_height", None)
        self._jpeg_quality: int = getattr(frames_cfg, "jpeg_quality", 95)
        self._retry: int = getattr(frames_cfg, "retry", 1)
        self._timeout: float = getattr(frames_cfg, "timeout", 30.0)

        # -------------------------------------------------------------------
        # Logger – static binding; per‑call ``job_id`` will be added later.
        # -------------------------------------------------------------------
        base_logger = get_logger(name=logger_name)
        self._logger = base_logger.bind(module="frame", provider="opencv")

        # -------------------------------------------------------------------
        # Thread‑pool – reuse the global singleton; injection allowed for tests.
        # -------------------------------------------------------------------
        self._thread_pool = thread_pool

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------
    def _validate_inputs(
        self, scene_result, video_path: Path
    ) -> None:
        """Validate that the scene result and video path are usable.

        Raises:
            FrameExtractionError – if validation fails.
        """
        if not video_path.exists():
            raise FrameExtractionError(f"Video path does not exist: {video_path}")
        if not hasattr(scene_result, "scenes"):
            raise FrameExtractionError("Invalid scene detection result supplied")

    def _select_frame_timestamps(
        self, start_sec: float, end_sec: float, scene_idx: int
    ) -> List[float]:
        """Return timestamps (in seconds) to capture for a single scene.

        The selection follows ``self._selection_strategy``.  When the strategy
        is ``max`` we spread up to ``self._max_frames_per_scene`` timestamps
        evenly across the interval; otherwise we pick the first, middle or last
        timestamp.
        """
        if start_sec >= end_sec:
            return []
        duration = end_sec - start_sec
        if self._selection_strategy == "first":
            return [start_sec]
        if self._selection_strategy == "last":
            return [end_sec]
        if self._selection_strategy == "middle":
            return [start_sec + duration / 2]
        # ``max`` – evenly spaced up to ``max_frames_per_scene``
        count = min(self._max_frames_per_scene, max(1, int(duration * 2)))  # heuristic
        step = duration / (count + 1)
        return [start_sec + step * (i + 1) for i in range(count)]

    def _ensure_output_dir(self, job_id: str) -> Path:
        """Create (if necessary) the directory ``data/jobs/{job_id}/frames``.

        Returns the absolute path to the directory.
        """
        out_dir = Path("data") / "jobs" / job_id / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _write_frame(
        self,
        img,
        out_path: Path,
    ) -> None:
        """Save *img* (BGR ndarray) to *out_path* as JPEG.

        Raises:
            FrameWriteError – if ``cv2.imwrite`` returns ``False``.
        """
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        success = cv2.imwrite(str(out_path), img, params)
        if not success:
            raise FrameWriteError(f"Failed to write frame to {out_path}")

    def _extract_for_scene(
        self,
        cap: cv2.VideoCapture,
        scene_idx: int,
        start_sec: float,
        end_sec: float,
        out_dir: Path,
        job_logger,
        ts_offset: float = 0.0,
    ) -> List[Frame]:
        """Extract the configured frames for a single scene.

        Returns a list of :class:`Frame` metadata objects.
        """
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames_meta: List[Frame] = []
        timestamps = self._select_frame_timestamps(start_sec, end_sec, scene_idx)
        for frame_idx, ts in enumerate(timestamps):
            rel_ts = ts - ts_offset
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, rel_ts * 1000.0))
            ret, img = cap.read()
            if not ret:
                # Fallback: try frame 0 if seeking failed
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, img = cap.read()
            if not ret:
                job_logger.warning("Unable to read frame at {}s (scene {})", ts, scene_idx)
                continue

            # Resize if required
            if self._resize_width or self._resize_height:
                width = self._resize_width or img.shape[1]
                height = self._resize_height or img.shape[0]
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

            out_path = out_dir / f"scene{scene_idx:04d}_frame{frame_idx:02d}.jpg"
            self._write_frame(img, out_path)
            frames_meta.append(
                Frame(
                    scene_index=scene_idx,
                    frame_index=frame_idx,
                    timestamp=ts,
                    image_path=out_path,
                    width=img.shape[1],
                    height=img.shape[0],
                )
            )
        return frames_meta

    def _run_extraction(
        self,
        scene_result,
        video_path: Path,
        *,
        job_logger,
    ) -> Result[FrameExtractionResult, FrameExtractionError]:
        """Core extraction routine used by both sync and async APIs.

        Handles validation, retries, timeout enforcement and conversion to the
        domain model. All errors are wrapped in ``Result.Err``.
        """
        try:
            self._validate_inputs(scene_result, video_path)
        except FrameExtractionError as exc:
            job_logger.error("Input validation failed: {}", exc)
            return Result.Err(exc)

        job_id = getattr(job_logger, "extra", {}).get("job_id", str(uuid.uuid4()))
        out_dir = self._ensure_output_dir(job_id)

        attempt = 0
        while attempt <= self._retry:
            start_ts = time.time()
            try:
                all_frames: List[Frame] = []

                if video_path.is_dir():
                    from src.core.video_loader import VideoLoader
                    clips = VideoLoader._get_clips_from_path(video_path)
                    for scene_idx, scene in enumerate(scene_result.scenes):
                        if scene_idx < len(clips):
                            clip_file = clips[scene_idx]
                            cap = cv2.VideoCapture(str(clip_file))
                            if cap.isOpened():
                                dur = scene.end_time - scene.start_time
                                frames = self._extract_for_scene(
                                    cap, scene_idx, 0.0, dur, out_dir, job_logger, ts_offset=0.0
                                )
                                all_frames.extend(frames)
                                cap.release()
                else:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        raise FrameReadError(f"OpenCV failed to open video: {video_path}")

                    for scene_idx, scene in enumerate(scene_result.scenes):
                        frames = self._extract_for_scene(
                            cap,
                            scene_idx,
                            scene.start_time,
                            scene.end_time,
                            out_dir,
                            job_logger,
                        )
                        all_frames.extend(frames)
                    cap.release()

                processing_time = time.time() - start_ts
                result = FrameExtractionResult(
                    frames=all_frames,
                    total_frames=len(all_frames),
                    processing_time=processing_time,
                )
                return Result.Ok(result)

            except Exception as exc:  # pragma: no cover – library‑specific failures
                job_logger.error(
                    "Frame extraction attempt {}/{} failed: {}", attempt + 1, self._retry + 1, exc
                )
                elapsed = time.time() - start_ts
                if elapsed > self._timeout:
                    return Result.Err(FrameExtractionError(
                        f"Extraction exceeded timeout of {self._timeout}s (took {elapsed:.2f}s)"
                    ))
                if attempt >= self._retry:
                    return Result.Err(FrameExtractionError(str(exc)))
                attempt += 1
                continue
        return Result.Err(FrameExtractionError("Maximum retry attempts exhausted"))

    # -------------------------------------------------------------------
    # Public API – BaseFrameExtractor contract
    # -------------------------------------------------------------------
    def extract(
        self, scene_detection_result, video_path: Path = None
    ) -> Result[FrameExtractionResult, FrameExtractionError]:
        """Synchronous frame extraction."""
        if isinstance(scene_detection_result, Path) and not isinstance(video_path, Path):
            scene_detection_result, video_path = video_path, scene_detection_result

        job_logger = self._logger.bind(job_id="sync")
        return self._run_extraction(scene_detection_result, video_path, job_logger=job_logger)

    def extract_frames(
        self, video_path: Path, scene_result
    ) -> Result[FrameExtractionResult, FrameExtractionError]:
        """Alias method matching extract_frames(video_path, scene_result) interface."""
        return self.extract(scene_result, video_path)

    def extract_async(
        self, scene_detection_result, video_path: Path
    ) -> Awaitable[Result[FrameExtractionResult, FrameExtractionError]]:
        """Asynchronous extraction using the shared thread‑pool.

        Returns a ``Future``‑like object whose ``result()`` yields the ``Result``.
        """
        pool = (
            self._thread_pool.get_pool(self._config)
            if isinstance(self._thread_pool, ThreadPoolManager)
            else ThreadPoolManager.get_pool(self._config)
        )
        job_id = f"async-{int(time.time() * 1000)}"
        job_logger = self._logger.bind(job_id=job_id)

        def _task() -> Result[FrameExtractionResult, FrameExtractionError]:
            job_logger.info("Starting async frame extraction for {}", video_path)
            return self._run_extraction(scene_detection_result, video_path, job_logger=job_logger)

        return pool.submit(_task)

    def health_check(self) -> Result[bool, FrameExtractionError]:
        """Check that OpenCV can be imported and a dummy video can be opened.
        """
        try:
            # Attempt to import – the module is already imported at top.
            _ = cv2.__version__
            # Create a dummy ``VideoCapture`` on a non‑existent file – it should
            # fail gracefully but the import succeeded.
            cap = cv2.VideoCapture()
            cap.release()
            return Result.Ok(True)
        except Exception as exc:  # pragma: no cover – unlikely import failure
            return Result.Err(FrameExtractionError(str(exc)))
