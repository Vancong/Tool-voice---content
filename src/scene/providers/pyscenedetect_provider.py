"""PySceneDetect based scene‑detection provider.

Implements :class:`src.scene.base.BaseSceneDetector` using the `PySceneDetect`
library's :class:`scenedetect.detectors.ContentDetector`.  All heavy work is
delegated to the shared ``ThreadPoolExecutor`` from ``src.utils.thread_pool`` –
no new pool is instantiated.

The provider is fully configurable via ``AppConfig``:

* ``scene.threshold`` – sensitivity threshold for the content detector
* ``scene.min_scene_len`` – minimum scene length in frames (PySceneDetect unit)
* ``scene.retry`` – number of retries on failure
* ``scene.timeout`` – hard timeout per detection attempt (seconds)

Errors are wrapped in the project's ``Result`` type using the custom
exceptions defined in ``src.scene.exceptions``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Awaitable, List

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from src.core.result import Result
from src.scene.base import BaseSceneDetector
from src.scene.exceptions import (
    SceneDetectionError,
    VideoOpenError,
    SceneDetectError,
)
from src.scene.models import Scene, SceneDetectionResult
from src.utils.logger import get_logger
from src.utils.thread_pool import ThreadPoolManager
from src.config.settings import AppConfig, CONFIG

# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class PySceneDetectProvider(BaseSceneDetector):
    """Provider that wraps PySceneDetect's ``ContentDetector``.

    The class follows the same pattern as the STT providers: configuration is
    read once at construction time, a logger is bound with static context, and
    a shared thread‑pool is used for the asynchronous API.
    """

    def __init__(
        self,
        config: AppConfig = CONFIG,
        logger_name: str = "scene_pyscenedetect",
        thread_pool: ThreadPoolManager | None = None,
    ) -> None:
        # -------------------------------------------------------------------
        # Configuration – values are optional; sensible defaults are applied.
        # -------------------------------------------------------------------
        self._config = config
        scene_cfg = getattr(config, "scene", None)
        self._threshold: float = getattr(scene_cfg, "threshold", 30.0) if scene_cfg else 30.0
        self._min_scene_len: int = getattr(scene_cfg, "min_scene_len", 15) if scene_cfg else 15
        self._retry: int = getattr(scene_cfg, "retry", 1) if scene_cfg else 1
        self._timeout: float = getattr(scene_cfg, "timeout", 30.0) if scene_cfg else 30.0

        # -------------------------------------------------------------------
        # Logger – static binding; per‑call ``job_id`` will be added later.
        # -------------------------------------------------------------------
        base_logger = get_logger(name=logger_name)
        self._logger = base_logger.bind(module="scene", provider="pyscenedetect")

        # -------------------------------------------------------------------
        # Thread‑pool – reuse the global singleton; injection allowed for tests.
        # -------------------------------------------------------------------
        self._thread_pool = thread_pool

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------
    def _validate_video(self, video_path: Path) -> None:
        """Validate that *video_path* exists.

        Raises:
            VideoOpenError: If the path does not exist.
        """
        if not video_path.exists():
            raise VideoOpenError(f"Video path does not exist: {video_path}")

    def _build_result(self, scenes: List[tuple], fps: float) -> SceneDetectionResult:
        """Convert PySceneDetect scene tuples into :class:`SceneDetectionResult`.

        *scenes* is a list of ``(start_frame, end_frame)`` tuples.
        """
        scene_objs: List[Scene] = []
        total = 0.0
        for idx, (start, end) in enumerate(scenes):
            start_sec = start / fps
            end_sec = end / fps
            duration = end_sec - start_sec
            total += duration
            scene_objs.append(Scene.from_bounds(idx, start_sec, end_sec))
        return SceneDetectionResult(scenes=scene_objs, total_duration=total)

    def _run_detection(
        self,
        video_path: Path,
        *,
        job_logger,
    ) -> Result[SceneDetectionResult, SceneDetectionError]:
        """Core detection routine used by both sync and async APIs.

        Handles validation, pre-cut directory processing, retries, timeout, and conversion
        to the domain model.
        """
        try:
            self._validate_video(video_path)
        except VideoOpenError as exc:
            job_logger.error("Video validation failed: {}", exc)
            return Result.Err(exc)

        # Handle directory of pre-cut clips directly
        if video_path.is_dir():
            try:
                from src.core.video_loader import VideoLoader
                import cv2
                clips = VideoLoader._get_clips_from_path(video_path)
                job_logger.info("Processing {} pre-cut clips directly as scenes...", len(clips))
                scene_objs: List[Scene] = []
                cum_time = 0.0
                for idx, clip in enumerate(clips):
                    cap = cv2.VideoCapture(str(clip))
                    if not cap.isOpened():
                        dur = 10.0
                    else:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        dur = frames / fps if fps else 10.0
                        cap.release()
                    scene_objs.append(Scene.from_bounds(idx, cum_time, cum_time + dur))
                    cum_time += dur
                return Result.Ok(SceneDetectionResult(scenes=scene_objs, total_duration=cum_time))
            except Exception as exc:
                job_logger.error("Error creating scenes from pre-cut directory: {}", exc)
                return Result.Err(SceneDetectError(str(exc)))

        attempt = 0
        while attempt <= self._retry:
            start_ts = time.time()
            try:
                # PySceneDetect 0.6+ API: open_video + SceneManager
                video = open_video(str(video_path))
                scene_manager = SceneManager()
                detector = ContentDetector(
                    threshold=self._threshold,
                    min_scene_len=self._min_scene_len,
                )
                scene_manager.add_detector(detector)
                scene_manager.detect_scenes(video)

                fps = video.frame_rate
                raw_scenes = scene_manager.get_scene_list()
                # Each element is a (FrameTimecode, FrameTimecode) tuple.
                scene_tuples = [(s[0].get_frames(), s[1].get_frames()) for s in raw_scenes]

                result = self._build_result(scene_tuples, fps)
                return Result.Ok(result)

            except Exception as exc:  # pragma: no cover – library‑specific failures
                job_logger.error(
                    "Scene detection attempt {}/{} failed: {}", attempt + 1, self._retry + 1, exc
                )
                elapsed = time.time() - start_ts
                if elapsed > self._timeout:
                    return Result.Err(SceneDetectError(
                        f"Scene detection exceeded timeout of {self._timeout}s (took {elapsed:.2f}s)"
                    ))
                if attempt >= self._retry:
                    return Result.Err(SceneDetectError(str(exc)))
                attempt += 1
                continue
        return Result.Err(SceneDetectError("Maximum retry attempts exhausted"))

    # -------------------------------------------------------------------
    # Public API – BaseSceneDetector contract
    # -------------------------------------------------------------------
    def detect(self, video_path: Path) -> Result[SceneDetectionResult, SceneDetectionError]:
        """Synchronous scene detection.

        The method binds a deterministic ``job_id`` (``sync``) for logging.
        """
        job_logger = self._logger.bind(job_id="sync")
        return self._run_detection(video_path, job_logger=job_logger)

    def detect_async(
        self, video_path: Path
    ) -> Awaitable[Result[SceneDetectionResult, SceneDetectionError]]:
        """Asynchronous detection using the shared thread‑pool.

        Returns a ``Future``‑like object whose ``result()`` method yields the
        ``Result``.
        """
        # Resolve the shared pool (fallback to the singleton if none injected).
        pool = (
            self._thread_pool.get_pool(self._config)
            if isinstance(self._thread_pool, ThreadPoolManager)
            else ThreadPoolManager.get_pool(self._config)
        )
        job_id = f"async-{int(time.time() * 1000)}"
        job_logger = self._logger.bind(job_id=job_id)

        def _task() -> Result[SceneDetectionResult, SceneDetectionError]:
            job_logger.info("Starting async scene detection for {}", video_path)
            return self._run_detection(video_path, job_logger=job_logger)

        return pool.submit(_task)

    def health_check(self) -> Result[bool, SceneDetectionError]:
        """Simple health check – attempts to instantiate the ContentDetector.
        """
        try:
            ContentDetector(threshold=self._threshold, min_scene_len=self._min_scene_len)
            return Result.Ok(True)
        except Exception as exc:  # pragma: no cover – unlikely
            return Result.Err(SceneDetectError(str(exc)))
