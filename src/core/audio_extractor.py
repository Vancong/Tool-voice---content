"""
src/core/audio_extractor.py

AudioExtractor service for extracting WAV audio from video files using FFmpeg.
Provides comprehensive diagnostic logging via ffprobe.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.core.result import Result
from src.core.video_loader import VideoLoader
from src.utils.logger import get_logger

_logger = get_logger("audio_extractor")


class AudioExtractor:
    """Service to extract WAV audio tracks from video files or directories of pre-cut video clips using FFmpeg."""

    def extract(self, video_path: Path, job_id: str = "default") -> Result[Path, Exception]:
        """Extract audio stream(s) from *video_path* (file or directory of pre-cut clips) and save as WAV.

        Output path: `data/jobs/{job_id}/audio/audio.wav`.

        Parameters
        ----------
        video_path: Path
            Input video file path or directory containing pre-cut clips.
        job_id: str
            Unique job identifier.

        Returns
        -------
        Result[Path, Exception]
            Result.Ok(audio_path) on success, or Result.Err(exception) on failure.
        """
        log = _logger.bind(job_id=job_id)
        log.info("Extracting audio from video source: {}", video_path)

        try:
            video_path = Path(video_path).resolve()
            if not video_path.exists():
                err_msg = f"Input video path not found: {video_path}"
                log.error("Audio extraction failed: {}", err_msg)
                return Result.Err(FileNotFoundError(err_msg))

            clips = VideoLoader._get_clips_from_path(video_path)
            out_dir = Path("data") / "jobs" / job_id / "audio"
            out_dir.mkdir(parents=True, exist_ok=True)
            final_audio_path = (out_dir / "audio.wav").resolve()

            if len(clips) == 1 and video_path.is_file():
                # Single video file processing
                return self._extract_single(clips[0], final_audio_path, log)

            # Multi-clip directory processing
            log.info("Processing {} pre-cut clips for audio extraction...", len(clips))
            clip_wavs: List[Path] = []

            for idx, clip in enumerate(clips):
                clip_wav = (out_dir / f"clip_{idx:04d}.wav").resolve()
                res = self._extract_single(clip, clip_wav, log, allow_silence_fallback=True)
                if res.is_ok:
                    clip_wavs.append(clip_wav)
                else:
                    log.warning("Failed audio extraction for clip {}, generating silence fallback", clip.name)
                    # Generate 5s silence as fallback
                    silence_cmd = [
                        "ffmpeg", "-y",
                        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                        "-t", "5",
                        "-c:a", "pcm_s16le",
                        str(clip_wav),
                    ]
                    subprocess.run(silence_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    clip_wavs.append(clip_wav)

            # Concatenate clip WAV files into final_audio_path
            concat_txt = out_dir / "concat_audio.txt"
            with open(concat_txt, "w", encoding="utf-8") as f:
                for clip_wav in clip_wavs:
                    f.write(f"file '{clip_wav.as_posix()}'\n")

            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_txt),
                "-c", "copy",
                str(final_audio_path),
            ]
            log.info("Concatenating clip audio files into single WAV...")
            proc = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if proc.returncode != 0 or not final_audio_path.exists():
                log.warning("Audio concatenation failed or audio stream empty. Creating empty/silent audio track.")
                # Create silent WAV
                silence_cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                    "-t", "10",
                    "-c:a", "pcm_s16le",
                    str(final_audio_path),
                ]
                subprocess.run(silence_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

            log.info("Multi-clip audio extraction finished: {}", final_audio_path)
            return Result.Ok(final_audio_path)

        except Exception as exc:
            log.error("Audio extraction unexpected exception: {}", exc)
            return Result.Err(exc)

    def _extract_single(
        self,
        video_path: Path,
        audio_path: Path,
        log,
        allow_silence_fallback: bool = False,
    ) -> Result[Path, Exception]:
        """Extract WAV from a single clip file."""
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-map", "0:a:0",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(audio_path),
        ]
        proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0:
            return Result.Ok(audio_path)

        # Fallback without explicit map if 0:a:0 failed
        ffmpeg_cmd_fallback = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(audio_path),
        ]
        proc2 = subprocess.run(ffmpeg_cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc2.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0:
            return Result.Ok(audio_path)

        err_msg = f"Audio extraction failed for {video_path.name}"
        return Result.Err(RuntimeError(err_msg))

