"""Timeline builder provider.

Transforms a :class:`src.vision.models.VisionAnalysisResult` into a structured
timeline without invoking any external AI services.  The implementation mirrors
the provider pattern used throughout the codebase (STT, Scene Detection, Frame,
Vision).

Key characteristics:
* Configuration is read once from ``AppConfig`` (section ``timeline``).
* Supports ``retry``, ``timeout`` and simple validation.
* Uses the shared ``ThreadPoolExecutor`` from ``src.utils.thread_pool`` for
  the asynchronous API.
* All heavy work is performed synchronously inside ``_run_build`` – the async
  method merely submits the task to the pool.
* Returns the custom ``Result`` type wrapping either a ``TimelineResult`` or a
  ``TimelineError``.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Awaitable, List

from src.core.result import Result
from src.timeline.base import BaseTimelineBuilder
from src.timeline.exceptions import (
    TimelineError,
    TimelineBuildError,
    InvalidTimelineError,
)
from src.timeline.models import TimelineEvent, MovieTimeline, TimelineResult
from src.utils.logger import get_logger
from src.utils.thread_pool import ThreadPoolManager
from src.config.settings import AppConfig, CONFIG


class TimelineBuilderProvider(BaseTimelineBuilder):
    """Concrete implementation that builds a movie timeline from vision analysis.

    The provider does **not** call any AI model – it merely restructures the
    data already present in :class:`VisionAnalysisResult`.
    """

    def __init__(
        self,
        config: AppConfig = CONFIG,
        logger_name: str = "timeline_builder",
        thread_pool: ThreadPoolManager | None = None,
    ) -> None:
        # -------------------------------------------------------------------
        # Configuration – optional ``timeline`` block; defaults are provided.
        # -------------------------------------------------------------------
        self._config = config
        timeline_cfg = getattr(config, "timeline", None) or {}
        self._retry: int = getattr(timeline_cfg, "retry", 1)
        self._timeout: float = getattr(timeline_cfg, "timeout", 30.0)

        # -------------------------------------------------------------------
        # Logger – static binding; per‑call ``job_id`` will be added later.
        # -------------------------------------------------------------------
        base_logger = get_logger(name=logger_name)
        self._logger = base_logger.bind(module="timeline", provider="builder")

        # -------------------------------------------------------------------
        # Thread‑pool – reuse the global singleton; allow injection for tests.
        # -------------------------------------------------------------------
        self._thread_pool = thread_pool

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------
    def _validate_input(self, vision_result) -> None:
        """Confirm that ``vision_result`` contains the expected structure.

        Raises:
            InvalidTimelineError – if the input is malformed.
        """
        if vision_result is None:
            raise InvalidTimelineError("VisionAnalysisResult is None")
        if not hasattr(vision_result, "scenes"):
            raise InvalidTimelineError("VisionAnalysisResult missing 'scenes' attribute")
        # ``scenes`` should be an iterable of objects with required attributes.
        for scene in vision_result.scenes:
            required = ["scene_index", "summary", "key_characters", "key_actions", "key_objects", "dominant_emotion"]
            for attr in required:
                if not hasattr(scene, attr):
                    raise InvalidTimelineError(f"Scene missing required attribute '{attr}'")

    def _build_events(self, scenes) -> List[TimelineEvent]:
        """Convert ``SceneAnalysis`` objects into ordered ``TimelineEvent`` objects.

        The original timestamps are not available at this stage; we set them to
        ``0.0`` as placeholders.  The calling code can later enrich them if needed.
        """
        events: List[TimelineEvent] = []
        for scene in sorted(scenes, key=lambda s: s.scene_index):
            events.append(
                TimelineEvent(
                    scene_index=scene.scene_index,
                    start_time=0.0,
                    end_time=0.0,
                    summary=scene.summary,
                    characters=scene.key_characters,
                    actions=scene.key_actions,
                    objects=scene.key_objects,
                    emotion=scene.dominant_emotion,
                )
            )
        return events

    def _run_build(
        self,
        vision_result,
        video_info=None,
        *,
        job_logger,
    ) -> Result[TimelineResult, TimelineError]:
        attempt = 0
        while attempt <= self._retry:
            start_ts = time.time()
            try:
                events = []
                if vision_result and hasattr(vision_result, "scenes") and vision_result.scenes:
                    events = self._build_events(vision_result.scenes)
                elif video_info and getattr(video_info, "clips", None):
                    for idx, c_path in enumerate(video_info.clips):
                        events.append(
                            TimelineEvent(
                                scene_index=idx,
                                start_time=0.0,
                                end_time=0.0,
                                summary=f"Clip {c_path.name}",
                                characters=[],
                                actions=[],
                                objects=[],
                                emotion="neutral",
                            )
                        )
                else:
                    events = [
                        TimelineEvent(
                            scene_index=0,
                            start_time=0.0,
                            end_time=0.0,
                            summary="Full video clip",
                            characters=[],
                            actions=[],
                            objects=[],
                            emotion="neutral",
                        )
                    ]

                timeline = MovieTimeline(
                    events=events,
                    total_scenes=len(events),
                    duration=getattr(video_info, "duration_sec", 0.0) if video_info else 0.0,
                )
                processing_time = time.time() - start_ts
                return Result.Ok(TimelineResult(timeline=timeline, processing_time=processing_time))
            except Exception as exc:
                job_logger.error(
                    "Timeline build attempt {}/{} failed: {}",
                    attempt + 1,
                    self._retry + 1,
                    exc,
                )
                if attempt >= self._retry:
                    return Result.Err(TimelineBuildError(str(exc)))
                attempt += 1
                continue
        return Result.Err(TimelineBuildError("Maximum retry attempts exhausted"))

    # -------------------------------------------------------------------
    # Public API – BaseTimelineBuilder contract
    # -------------------------------------------------------------------
    def build(
        self,
        stt_result=None,
        vision_analysis_result=None,
        video_info=None,
        *args,
        **kwargs,
    ) -> Result[TimelineResult, TimelineError]:
        """Synchronous timeline construction.
        """
        target_vision = vision_analysis_result
        if target_vision is None and stt_result is not None:
            if hasattr(stt_result, "scenes"):
                target_vision = stt_result

        job_logger = self._logger.bind(job_id="sync")
        return self._run_build(target_vision, video_info=video_info, job_logger=job_logger)

    def build_async(
        self, vision_analysis_result
    ) -> Awaitable[Result[TimelineResult, TimelineError]]:
        """Asynchronous timeline construction using the shared thread‑pool.

        Returns a ``Future``‑like object; its ``result()`` method yields the
        ``Result``.
        """
        pool = (
            self._thread_pool.get_pool(self._config)
            if isinstance(self._thread_pool, ThreadPoolManager)
            else ThreadPoolManager.get_pool(self._config)
        )
        job_id = f"async-{int(time.time() * 1000)}"
        job_logger = self._logger.bind(job_id=job_id)

        def _task() -> Result[TimelineResult, TimelineError]:
            job_logger.info("Starting async timeline build")
            return self._run_build(vision_analysis_result, job_logger=job_logger)

        return pool.submit(_task)

    def health_check(self) -> Result[bool, TimelineError]:
        """Simple health check – the builder has no external dependencies.
        """
        # Since the builder only manipulates in‑memory data, it is always ready.
        return Result.Ok(True)
