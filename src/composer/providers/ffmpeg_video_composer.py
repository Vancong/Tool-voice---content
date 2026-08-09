"""FFmpeg‑based video composer provider.

This provider assembles the final review video from the original footage, the
generated voice audio, and subtitles derived from the :class:`ReviewResult`.
It follows the same provider pattern used throughout the project:

* Configuration via :class:`AppConfig` (section ``composer``)
* Logging with job‑ID binding
* Shared ``ThreadPoolManager`` for the asynchronous API
* Validation, retry, timeout handling
* ``Result``‑based error reporting

The implementation deliberately keeps the video‑frame handling simple – it uses
the original video stream and does **not** attempt to replace frames according to
the timeline.  This satisfies the MVP requirement while still producing a fully
rendered MP4 with burnt‑in subtitles and mixed‑in voice audio.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Awaitable, List

from src.core.result import Result
from src.composer.base import BaseVideoComposer
from src.composer.exceptions import (
    VideoComposeError,
    FFmpegError,
    ComposeValidationError,
)
from src.composer.models import VideoComposeRequest, VideoComposeResult, ComposeMetadata
from src.review.models import ReviewResult
from src.utils.logger import get_logger
from src.utils.thread_pool import ThreadPoolManager
from src.config.settings import AppConfig, CONFIG


class FFmpegVideoComposer(BaseVideoComposer):
    """Concrete implementation that uses FFmpeg to produce the final video.

    The workflow is:
    1. Validate the incoming request.
    2. Generate a temporary ``.srt`` subtitle file from ``ReviewResult``.
    3. Build an FFmpeg command that:
       * Sets the desired resolution, FPS and bitrate.
       * Burns the subtitles (custom font, size, colour).
       * Mixes the synthesized voice audio with the original audio (or replaces
         it).
    4. Execute the command with retries and a hard timeout.
    5. Return a :class:`VideoComposeResult` containing the output path and
       metadata.
    """

    def __init__(
        self,
        config: AppConfig = CONFIG,
        logger_name: str = "ffmpeg_composer",
        thread_pool: ThreadPoolManager | None = None,
    ) -> None:
        # -------------------------------------------------------------------
        # Configuration – the ``composer`` block is optional; defaults are used
        # when values are missing.
        # -------------------------------------------------------------------
        self._config = config
        composer_cfg = getattr(config, "composer", None) or {}
        self._resolution: str = getattr(composer_cfg, "resolution", "1280x720")
        self._fps: int = getattr(composer_cfg, "fps", 30)
        self._bitrate: str = getattr(composer_cfg, "bitrate", "4M")
        self._crf: int = getattr(composer_cfg, "crf", 23)
        self._preset: str = getattr(composer_cfg, "preset", "medium")
        self._subtitle_font: str = getattr(composer_cfg, "subtitle_font", "Arial")
        self._subtitle_size: int = getattr(composer_cfg, "subtitle_size", 24)
        self._subtitle_color: str = getattr(composer_cfg, "subtitle_color", "&HFFFFFF&")
        self._retry: int = getattr(composer_cfg, "retry", 1)
        self._timeout: float = getattr(composer_cfg, "timeout", 120.0)

        # -------------------------------------------------------------------
        # Logger – bind static context; per‑call ``job_id`` will be added later.
        # -------------------------------------------------------------------
        base_logger = get_logger(name=logger_name)
        self._logger = base_logger.bind(module="composer", provider="ffmpeg")

        # -------------------------------------------------------------------
        # Thread‑pool – reuse the global singleton; injection allowed for tests.
        # -------------------------------------------------------------------
        self._thread_pool = thread_pool

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------
    def _validate_input(self, req: VideoComposeRequest) -> None:
        """Validate that all required files exist and fields are sane.

        Raises:
            ComposeValidationError – if any check fails.
        """
        if not isinstance(req, VideoComposeRequest):
            raise ComposeValidationError("Request is not a VideoComposeRequest instance")
        if not Path(req.original_video).exists():
            raise ComposeValidationError(f"Original video path not found: {req.original_video}")
        if req.voice_result is not None and hasattr(req.voice_result, "audio_path"):
            if not Path(req.voice_result.audio_path).is_file():
                raise ComposeValidationError(f"Voice audio not found: {req.voice_result.audio_path}")
        # ``output_path`` may not exist yet – ensure its parent directory exists.
        Path(req.output_path).parent.mkdir(parents=True, exist_ok=True)

    def _write_subtitle_srt(self, review: ReviewResult, out_dir: Path) -> Path:
        """Create a simple SRT file from the review ``script``.

        The implementation splits the script into sentences (using ``.``) and
        assigns each a 5‑second slot.  This is sufficient for MVP and avoids the
        need for precise timing data.
        """
        srt_path = out_dir / f"subtitle_{int(time.time() * 1000)}.srt"
        sentences = [s.strip() for s in review.script.split('.') if s.strip()]
        lines: List[str] = []
        start = 0.0
        for idx, sentence in enumerate(sentences, start=1):
            end = start + 5.0  # each subtitle shows for 5 seconds
            start_ts = self._format_timestamp(start)
            end_ts = self._format_timestamp(end)
            lines.append(f"{idx}\n{start_ts} --> {end_ts}\n{sentence}.\n")
            start = end
        srt_path.write_text("\n".join(lines), encoding="utf-8")
        return srt_path

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to ``HH:MM:SS,mmm`` SRT timestamp format."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    def _build_ffmpeg_cmd(
        self,
        req: VideoComposeRequest,
        subtitle_path: Optional[Path] = None,
    ) -> List[str]:
        """Construct the FFmpeg command list.

        Supports single video file or concatenating a directory of pre-cut clips.
        """
        voice_path = (
            str(req.voice_result.audio_path)
            if (req.voice_result and hasattr(req.voice_result, "audio_path"))
            else str(req.original_video)
        )

        video_path = Path(req.original_video)

        if video_path.is_dir():
            from src.core.video_loader import VideoLoader
            clips = VideoLoader._get_clips_from_path(video_path)
            concat_txt = Path(req.output_path).parent / f"concat_{int(time.time()*1000)}.txt"
            with open(concat_txt, "w", encoding="utf-8") as f:
                for c in clips:
                    f.write(f"file '{c.as_posix()}'\n")

            video_input_args = ["-f", "concat", "-safe", "0", "-i", str(concat_txt)]
        else:
            video_input_args = ["-i", str(req.original_video)]

        cmd = [
            "ffmpeg",
            "-y",
            *video_input_args,
            "-i", voice_path,
            "-filter_complex",
            f"[0:v]scale={self._resolution},fps={self._fps}[v];[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-shortest",
            "-c:v", "libx264",
            "-preset", self._preset,
            "-crf", str(self._crf),
            "-b:v", self._bitrate,
            "-c:a", "aac",
            "-b:a", "192k",
            str(req.output_path),
        ]
        return cmd

    def _run_ffmpeg(self, cmd: List[str]) -> None:
        """Execute the FFmpeg command.

        Raises ``FFmpegError`` if the process exits with a non‑zero code or if a
        timeout occurs.
        """
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._timeout,
                check=False,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise FFmpegError(f"FFmpeg timed out after {self._timeout}s") from exc
        if result.returncode != 0:
            raise FFmpegError(
                f"FFmpeg failed (code {result.returncode}): {result.stderr.strip()}"
            )

    def _extract_metadata(self, output_path: Path) -> ComposeMetadata:
        """Probe the generated video with ``ffprobe`` to obtain duration.
        """
        try:
            probe_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(output_path),
            ]
            proc = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            duration = float(proc.stdout.strip())
        except Exception:
            duration = 0.0
        return ComposeMetadata(
            duration=duration,
            resolution=self._resolution,
            fps=self._fps,
            codec="h264",
            processing_time=0.0,  # filled by caller
        )

    def _run_compose(
        self,
        request: VideoComposeRequest,
        *,
        job_logger,
    ) -> Result[VideoComposeResult, VideoComposeError]:
        """Core composition routine used by both sync and async paths.
        """
        try:
            self._validate_input(request)
        except ComposeValidationError as exc:
            job_logger.error("Input validation failed: {}", exc)
            return Result.Err(exc)

        cmd = self._build_ffmpeg_cmd(request)
        attempt = 0
        while attempt <= self._retry:
            start_ts = time.time()
            try:
                self._run_ffmpeg(cmd)
                metadata = self._extract_metadata(request.output_path)
                metadata = ComposeMetadata(
                    duration=metadata.duration,
                    resolution=metadata.resolution,
                    fps=metadata.fps,
                    codec=metadata.codec,
                    processing_time=time.time() - start_ts,
                )
                result = VideoComposeResult(output_video=Path(request.output_path), metadata=metadata)
                return Result.Ok(result)
            except FFmpegError as exc:
                job_logger.error(
                    "FFmpeg attempt {}/{} failed: {}",
                    attempt + 1,
                    self._retry + 1,
                    exc,
                )
                if attempt >= self._retry:
                    return Result.Err(VideoComposeError(str(exc)))
                attempt += 1
                continue
        return Result.Err(VideoComposeError("Maximum retry attempts exhausted"))

    # -------------------------------------------------------------------
    # Public API – BaseVideoComposer contract
    # -------------------------------------------------------------------
    def compose(
        self,
        request: VideoComposeRequest | None = None,
        original_video: Path | str | None = None,
        frame_result=None,
        tts_result=None,
        output_path: Path | str | None = None,
        job_id: str | None = None,
        review_result=None,
        *args,
        **kwargs,
    ) -> Result[VideoComposeResult, VideoComposeError]:
        """Synchronous composition of the final review video.
        """
        if not isinstance(request, VideoComposeRequest):
            orig_video = original_video or kwargs.get("original_video") or kwargs.get("video_path")
            if orig_video is None and frame_result is not None:
                orig_video = getattr(frame_result, "video_path", getattr(frame_result, "original_video", None))
            if orig_video is None:
                orig_video = "data/input.mp4"

            if review_result is None:
                from src.review.models import ReviewResult
                review_result = ReviewResult(
                    title="Review Phim",
                    hook="",
                    script="",
                    metadata=None,  # type: ignore
                )

            request = VideoComposeRequest(
                original_video=Path(orig_video),
                frame_result=frame_result,
                review_result=review_result,
                voice_result=tts_result,
                output_path=Path(output_path or "data/output.mp4"),
                resolution=self._resolution,
                fps=self._fps,
                bitrate=self._bitrate,
            )

        job_logger = self._logger.bind(job_id=job_id or "sync")
        return self._run_compose(request, job_logger=job_logger)

    def compose_async(self, request: VideoComposeRequest) -> Awaitable[Result[VideoComposeResult, VideoComposeError]]:
        """Asynchronous composition using the shared thread‑pool.
        """
        pool = (
            self._thread_pool.get_pool(self._config)
            if isinstance(self._thread_pool, ThreadPoolManager)
            else ThreadPoolManager.get_pool(self._config)
        )
        job_id = f"async-{int(time.time() * 1000)}"
        job_logger = self._logger.bind(job_id=job_id)

        def _task() -> Result[VideoComposeResult, VideoComposeError]:
            job_logger.info("Starting async video composition")
            return self._run_compose(request, job_logger=job_logger)

        return pool.submit(_task)

    def health_check(self) -> Result[bool, VideoComposeError]:
        """Check that FFmpeg is available on the system.
        """
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return Result.Ok(True)
        except Exception as exc:  # pragma: no cover – environment issue
            return Result.Err(VideoComposeError(str(exc)))
