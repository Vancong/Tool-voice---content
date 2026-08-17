# -*- coding: utf-8 -*-
"""
src/core/workflow.py

WorkflowEngine – orchestration entry-point for the AI Movie Review pipeline.

Pipeline (simplified):
  1. VideoLoader   – load & split video into clips
  2. ReviewGenerator – per-clip AI analysis + content writing (MultiAgent)
  3. TTS           – synthesise narration audio
  4. VideoComposer – merge audio onto original video

Checkpoint/resume is supported after each stage.
"""

from __future__ import annotations

import json
import os
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Any, Dict

from src.core.result import Result
from src.core.video_loader import VideoLoader
from src.review.base import BaseReviewGenerator
from src.tts.base import BaseTTS
from src.composer.base import BaseVideoComposer
from src.utils.logger import get_logger, setup_job_file_logger

_logger = get_logger("workflow")


class WorkflowEngine:
    """Orchestrates the AI movie-review pipeline: load → review → tts → compose."""

    def __init__(
        self,
        video_loader: VideoLoader,
        review_generator: BaseReviewGenerator,
        tts: BaseTTS,
        video_composer: BaseVideoComposer,
    ) -> None:
        self._video_loader = video_loader
        self._review_generator = review_generator
        self._tts = tts
        self._video_composer = video_composer

    # ------------------------------------------------------------------
    # Checkpoint & Resume Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_checkpoint_dir(job_id: str) -> Path:
        p = Path("data") / "jobs" / job_id / "checkpoints"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def _save_checkpoint(cls, job_id: str, stage_name: str, data: Any) -> None:
        """Save stage result into data/jobs/<job_id>/checkpoint.json and data/jobs/<job_id>/checkpoints/<stage>.pkl."""
        try:
            ckpt_dir = cls._get_checkpoint_dir(job_id)
            pkl_file = ckpt_dir / f"{stage_name}.pkl"
            with open(pkl_file, "wb") as f:
                pickle.dump(data, f)

            json_file = Path("data") / "jobs" / job_id / "checkpoint.json"
            meta: Dict[str, Any] = {}
            if json_file.exists():
                try:
                    meta = json.loads(json_file.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}

            completed = meta.get("completed_stages", [])
            if stage_name not in completed:
                completed.append(stage_name)

            meta.update({
                "job_id": job_id,
                "last_completed_stage": stage_name,
                "completed_stages": completed,
                "updated_at": datetime.now().isoformat(),
            })
            json_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            _logger.info("[Checkpoint] Saved stage '{}' for job {}", stage_name, job_id)
        except Exception as exc:
            _logger.warning("[Checkpoint] Failed to save stage '{}': {}", stage_name, exc)

    @classmethod
    def _load_checkpoint(cls, job_id: str, stage_name: str) -> Optional[Any]:
        """Load cached stage result if checkpoint exists."""
        try:
            pkl_file = cls._get_checkpoint_dir(job_id) / f"{stage_name}.pkl"
            if pkl_file.exists():
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                _logger.info("[Checkpoint] Resumed stage '{}' from cache for job {}", stage_name, job_id)
                return data
        except Exception as exc:
            _logger.warning("[Checkpoint] Could not load cache for stage '{}': {}", stage_name, exc)
        return None

    # ------------------------------------------------------------------
    # Stage Execution Wrapper
    # ------------------------------------------------------------------

    @staticmethod
    def _stage(
        job_id: str,
        name: str,
        fn: Callable[[], Result],
        input_path: Optional[object] = None,
        output_path: Optional[object] = None,
        provider_obj: Optional[object] = None,
    ) -> Result:
        """Execute a single pipeline stage with rich instrumentation and timing."""
        log = _logger.bind(job_id=job_id, stage=name)
        log.info("START {}", name)

        if input_path is not None:
            log.info("Input path:\n{}", input_path)
        if output_path is not None:
            log.info("Output path:\n{}", output_path)
        if provider_obj is not None:
            prov_name = type(provider_obj).__name__
            cfg = getattr(provider_obj, "_config", None) or getattr(provider_obj, "__dict__", {})
            log.info("Configuration ({}):\n{}", prov_name, cfg)

        t0 = time.perf_counter()

        try:
            result: Result = fn()
        except Exception as exc:
            tb = traceback.format_exc()
            log.error("Unhandled exception in stage {}:\n{}", name, tb)
            result = Result.Err(exc)

        elapsed = time.perf_counter() - t0

        log.info("Provider returned:\n{}", type(result))
        log.info("Result.is_ok: {}", result.is_ok)

        if result.is_ok:
            val = result.unwrap() if hasattr(result, "unwrap") else getattr(result, "value", None)
            log.info("Result.value type: {}", type(val))
            log.info("Result.error: None")
        else:
            err = getattr(result, "error", None)
            log.info("Result.value type: None")
            log.info("Result.error: {}", err)
            log.error("[{}] FAILED {} — error: {}\nTraceback:\n{}", job_id, name, err, traceback.format_exc())

        log.info("END {}", name)
        log.info("Elapsed time: {:.2f}s", elapsed)

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: Path,
        output_path: Path,
        job_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        cancel_token: Optional[object] = None,
        debug_mode: bool = False,
        use_gemini_web: bool = False,
        custom_script: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        sample_style: Optional[str] = None,
        custom_sample_text: Optional[str] = None,
        google_sheet_webhook_url: Optional[str] = None,
        resume: bool = True,
        skip_media_generation: bool = False,
    ) -> Result:
        """Execute the full pipeline from *video_path* to *output_path* with checkpoint resume."""
        job_log_file = setup_job_file_logger(job_id, debug_mode=debug_mode)

        log = _logger.bind(job_id=job_id)
        log.info("START PIPELINE")
        log.info("Job ID: {}", job_id)
        log.info("Job log file: {}", job_log_file)
        log.info("Debug mode: {}", debug_mode)
        log.info("Use Gemini Web mode: {}", use_gemini_web)
        log.info("Input video path: {}", video_path)
        log.info("Output video path: {}", output_path)

        pipeline_start = time.perf_counter()

        def _check_cancel() -> Optional[Result]:
            if cancel_token is not None and getattr(cancel_token, "is_set", lambda: False)():
                err = RuntimeError(f"[{job_id}] Pipeline cancelled by user")
                log.warning("{}", err)
                return Result.Err(err)
            return None

        def _notify(stage_name: str, pct: float) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(stage_name, pct)
                except Exception:
                    log.warning("progress_callback raised an exception – ignoring")

        # ----------------------------------------------------------------
        # Stage 0 – Pre-flight Gemini Web Authentication & Health Check
        # ----------------------------------------------------------------
        if (cancelled := _check_cancel()) is not None:
            return cancelled

        if use_gemini_web:
            _notify("BƯỚC 1/2: Kiểm tra kết nối và đăng nhập Gemini Web...", 0.01)
            r_health = self._review_generator.health_check()
            if r_health.is_err:
                log.error("Pre-flight Gemini Web health check failed: {}", r_health.error)
                return r_health

        if (cancelled := _check_cancel()) is not None:
            return cancelled

        # Early validate ElevenLabs API Key only AFTER Gemini Web passes
        _notify("BƯỚC 2/2: Kiểm tra kết nối và API Key ElevenLabs (TTS)...", 0.02)
        r_tts_health = self._tts.health_check()
        if r_tts_health.is_err:
            log.error("Pre-flight TTS health check failed: {}", r_tts_health.error)
            return r_tts_health

        # ----------------------------------------------------------------
        # Stage 1 – Video Loader
        # ----------------------------------------------------------------
        if (cancelled := _check_cancel()) is not None:
            return cancelled

        video_info = self._load_checkpoint(job_id, "VideoLoader") if resume else None
        if video_info is None:
            r_video = self._stage(
                job_id, "VideoLoader",
                lambda: Result.Ok(self._video_loader.load(video_path, job_id).result()),
                input_path=video_path,
                provider_obj=self._video_loader,
            )
            if r_video.is_err:
                return r_video
            video_info = r_video.unwrap()
            self._save_checkpoint(job_id, "VideoLoader", video_info)
        else:
            if video_info and getattr(video_info, "clips", None):
                valid_clips = [
                    p for p in video_info.clips
                    if Path(p).exists() and not Path(p).name.endswith("_opt.mp4") and not Path(p).name.startswith(".")
                ]
                if len(valid_clips) != len(video_info.clips):
                    log.info("Sanitizing VideoLoader checkpoint clips: {} cached -> {} valid", len(video_info.clips), len(valid_clips))
                    video_info = video_info._replace(clips=valid_clips)
        _notify("VideoLoader", 0.15)

        # Dùng None làm placeholder cho timeline (MultiAgent không cần timeline thật)
        timeline_result = None

        # ----------------------------------------------------------------
        # Stage 2 – Review Generation & Google Sheet Export
        # ----------------------------------------------------------------
        if (cancelled := _check_cancel()) is not None:
            return cancelled

        review_result = self._load_checkpoint(job_id, "ReviewGenerator") if resume else None
        if review_result is None:
            if custom_script and custom_script.strip():
                from src.review.models import ReviewResult, ReviewMetadata
                log.info("Custom script provided manually. Bypassing ReviewGenerator.")
                final_script = custom_script.strip()
                review_result = ReviewResult(
                    title="Kịch bản Tùy chỉnh",
                    hook="",
                    script=final_script,
                    metadata=ReviewMetadata(
                        total_words=len(final_script.split()),
                        estimated_duration=0.0,
                        model_name="manual-script",
                        processing_time=0.0,
                    ),
                )
            else:
                target_dur = int(getattr(video_info, "duration_sec", getattr(video_info, "duration", 180))) if video_info else 180
                r_review = self._stage(
                    job_id, "ReviewGenerator",
                    lambda: self._review_generator.generate(
                        timeline_result,
                        job_id=job_id,
                        target_duration=target_dur,
                        use_gemini_web=use_gemini_web,
                        custom_instructions=custom_instructions,
                        sample_style=sample_style,
                        custom_sample_text=custom_sample_text,
                        video_info=video_info,
                        google_sheet_webhook_url=google_sheet_webhook_url,
                        progress_callback=lambda st, p: _notify(f"ReviewGenerator ({st})", 0.75 + p * 0.1),
                    ),
                    provider_obj=self._review_generator,
                )
                if r_review.is_err:
                    return r_review
                review_result = r_review.unwrap()
            self._save_checkpoint(job_id, "ReviewGenerator", review_result)

        # Automatically export local CSV (live per-clip sync already pushed to Google Sheet Webhook)
        try:
            from src.exporter.google_sheet_exporter import GoogleSheetExporter
            GoogleSheetExporter.sync_to_sheet(
                job_id=job_id,
                timeline=timeline_result,
                review_text=review_result.script,
                webhook_url=None,  # Live per-clip sync already handles Google Sheet webhook push
                video_info=video_info,
                push_webhook=False,
            )
        except Exception as exc:
            log.warning("Google Sheet local CSV export warning: {}", exc)

        _notify("ReviewGenerator", 0.85)

        if skip_media_generation:
            _notify("Hoàn tất (Đã đẩy Google Sheet & bỏ qua ghép audio/video)", 1.0)
            log.info("[Pipeline] Skip media generation requested. Returning early after successful Google Sheet sync.")
            return Result.Ok(output_path)

        # ----------------------------------------------------------------
        # Stage 3 – Text-to-Speech
        # ----------------------------------------------------------------
        if (cancelled := _check_cancel()) is not None:
            return cancelled

        tts_output_path = Path("data") / "jobs" / job_id / "tts" / "narration.mp3"
        tts_result = self._load_checkpoint(job_id, "TextToSpeech") if resume else None
        if tts_result is None or not tts_output_path.exists():
            r_tts = self._stage(
                job_id, "TextToSpeech",
                lambda: self._tts.synthesize(review_result.script, tts_output_path, job_id=job_id),
                input_path=review_result.script,
                output_path=tts_output_path,
                provider_obj=self._tts,
            )
            if r_tts.is_err:
                return r_tts
            tts_result = r_tts.unwrap()
            self._save_checkpoint(job_id, "TextToSpeech", tts_result)

        # Automatically copy standalone voice narration audio to output directory
        try:
            import shutil
            voice_out = output_path.parent / f"{output_path.stem}_voice.mp3"
            if tts_output_path.exists():
                shutil.copy2(tts_output_path, voice_out)
                log.info("Saved standalone voice narration file to: {}", voice_out)
        except Exception as exc_cp:
            log.warning("Could not copy voice file to output folder: {}", exc_cp)

        _notify("TextToSpeech", 0.92)

        # ----------------------------------------------------------------
        # Stage 4 – Video Composer
        # ----------------------------------------------------------------
        if (cancelled := _check_cancel()) is not None:
            return cancelled

        r_composer = self._stage(
            job_id, "VideoComposer",
            lambda: self._video_composer.compose(
                original_video=video_path,
                frame_result=frame_result,
                tts_result=tts_result,
                output_path=output_path,
                job_id=job_id,
                review_result=review_result,
            ),
            input_path=tts_output_path,
            output_path=output_path,
            provider_obj=self._video_composer,
        )
        if r_composer.is_err:
            return r_composer
        final_video_path = r_composer.unwrap()
        self._save_checkpoint(job_id, "VideoComposer", str(final_video_path))
        _notify("VideoComposer", 1.0)

        total_elapsed = time.perf_counter() - pipeline_start
        log.info("END PIPELINE")
        log.info("Total pipeline elapsed time: {:.2f}s", total_elapsed)

        return Result.Ok(final_video_path)
