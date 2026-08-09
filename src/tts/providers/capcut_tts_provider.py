"""Microsoft Edge TTS provider implementation.

Replaces the legacy REST provider with Microsoft Edge TTS via `edge-tts`.
Default voice: `vi-VN-HoaiMyNeural`.
No API Keys, Session IDs, or cookies required.
Outputs MP3 audio files.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from pathlib import Path
from typing import Awaitable, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

# Lazy import edge_tts – provide clear installation instructions if missing
_import_error: Optional[str] = None
try:
    import edge_tts
except Exception as exc:  # pragma: no cover
    edge_tts = None  # type: ignore
    _import_error = f"{type(exc).__name__}: {exc}"

from src.core.result import Result
from src.tts.base import BaseTTS
from src.tts.exceptions import (
    TTSError,
    VoiceGenerationError,
    VoiceProviderError,
)
from src.tts.models import VoiceRequest, VoiceResult, VoiceMetadata
from src.review.models import ReviewResult
from src.utils.logger import get_logger
from src.utils.thread_pool import ThreadPoolManager
from src.config.settings import AppConfig, CONFIG


class CapCutTTSProvider(BaseTTS):
    """Concrete TTS provider utilizing Microsoft Edge TTS (edge-tts).

    Keeps the CapCutTTSProvider class name for seamless drop-in compatibility with
    WorkflowEngine. Requires no API keys, cookies, or session authentication.
    """

    def __init__(
        self,
        config: AppConfig = CONFIG,
        logger_name: str = "tts_edge",
        thread_pool: ThreadPoolManager | None = None,
        http_client=None,
    ) -> None:
        base_logger = get_logger(name=logger_name)
        self._logger = base_logger.bind(module="tts", provider="edge_tts")

        if edge_tts is None:
            err_msg = (
                f"Không tìm thấy package edge-tts. Hãy chạy: pip install edge-tts\n"
                f"Chi tiết lỗi: {_import_error}"
            )
            self._logger.error(err_msg)
            raise VoiceProviderError(err_msg)

        self._config = config
        tts_cfg = getattr(config, "tts", None) or {}
        
        # Resolve voice id with alias support (CapCut-like voices)
        raw_voice = (
            getattr(tts_cfg, "voice_id", None)
            or getattr(tts_cfg, "default_voice_id", None)
            or os.environ.get("TTS__VOICE_ID", "vi-VN-NamMinhNeural")
        )
        
        # Voice alias mapping
        voice_aliases = {
            "thanh_nien": "vi-VN-NamMinhNeural",
            "nam_review": "vi-VN-NamMinhNeural",
            "chang_trai_tu_tin": "vi-VN-NamMinhNeural",
            "capcut nam review (thanh niên)": "vi-VN-NamMinhNeural",
            "capcut nam review": "vi-VN-NamMinhNeural",
            "capcut nam": "vi-VN-NamMinhNeural",
            "vi-vn-namminhneural (nam truyền cảm)": "vi-VN-NamMinhNeural",
            "nam_minh": "vi-VN-NamMinhNeural",
            "vi_male_1": "vi-VN-NamMinhNeural",
            "chi_ban_mai": "vi-VN-HoaiMyNeural",
            "ban_mai": "vi-VN-HoaiMyNeural",
            "hoai_my": "vi-VN-HoaiMyNeural",
            "co_gai_hoat_ngon": "vi-VN-HoaiMyNeural",
            "capcut nữ review (chị ban mai)": "vi-VN-HoaiMyNeural",
            "capcut nữ review": "vi-VN-HoaiMyNeural",
            "capcut nữ": "vi-VN-HoaiMyNeural",
            "vi-vn-hoaimyneural (nữ truyền cảm)": "vi-VN-HoaiMyNeural",
            "vi_female_1": "vi-VN-HoaiMyNeural",
            "en-us-guyneural (nam us)": "en-US-GuyNeural",
            "en-us-jennyneural (nữ us)": "en-US-JennyNeural",
        }
        
        self._voice: str = voice_aliases.get(str(raw_voice).lower(), str(raw_voice))
        if not self._voice or self._voice == "en_us_001":
            self._voice = "vi-VN-NamMinhNeural"

        self._rate: str = getattr(tts_cfg, "rate", "+0%") or "+0%"
        self._pitch: str = getattr(tts_cfg, "pitch", "+0Hz") or "+0Hz"
        self._retry: int = getattr(tts_cfg, "retry", 1)
        self._timeout: float = getattr(tts_cfg, "timeout", 60.0)
        self._thread_pool = thread_pool

    def _validate_input(self, review: ReviewResult) -> None:
        if not hasattr(review, "script") or not review.script:
            raise VoiceGenerationError("ReviewResult missing a non-empty script")

    def _ensure_output_dir(self, job_id: str) -> Path:
        out_dir = Path("data") / "jobs" / job_id / "voice"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _get_audio_duration(self, file_path: Path) -> float:
        try:
            import subprocess
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                return float(probe.stdout.strip())
        except Exception:
            pass
        return 0.0

    def _synthesize_bytedance_capcut(self, text: str, session_id: str, voice: str, out_file: Path) -> None:
        """Synthesize audio using official ByteDance/CapCut TTS API endpoint with chunking."""
        import base64
        import json
        import urllib.parse
        import urllib.request

        # Map to ByteDance speaker codes
        # In ByteDance TikTok/CapCut:
        # BV074_streaming is Female (Chị Ban Mai / Nữ review)
        # BV075_streaming is Male (Thanh niên / Nam review)
        speaker_map = {
            # Male speakers -> BV075_streaming
            "vi-vn-namminhneural": "BV075_streaming",
            "thanh_nien": "BV075_streaming",
            "nam_review": "BV075_streaming",
            "chang_trai_tu_tin": "BV075_streaming",
            "capcut nam review (thanh niên)": "BV075_streaming",
            "capcut nam review": "BV075_streaming",
            "capcut nam": "BV075_streaming",
            "vi-vn-namminhneural (nam truyền cảm)": "BV075_streaming",
            "bv075_streaming": "BV075_streaming",
            "vi_male_1": "BV075_streaming",

            # Female speakers -> BV074_streaming
            "vi-vn-hoaimyneural": "BV074_streaming",
            "chi_ban_mai": "BV074_streaming",
            "ban_mai": "BV074_streaming",
            "hoai_my": "BV074_streaming",
            "co_gai_hoat_ngon": "BV074_streaming",
            "capcut nữ review (chị ban mai)": "BV074_streaming",
            "capcut nữ review": "BV074_streaming",
            "capcut nữ": "BV074_streaming",
            "vi-vn-hoaimyneural (nữ truyền cảm)": "BV074_streaming",
            "bv074_streaming": "BV074_streaming",
            "vi_female_1": "BV074_streaming",
        }
        speaker = speaker_map.get(voice.lower(), "BV075_streaming")

        # Split text intelligently into small natural chunks (<= 110 chars / ~20 words)
        # to strictly satisfy ByteDance API single-request limits and prevent "Text too long" errors.
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        raw_sentences = re.split(r"([.!?\n]+)", cleaned_text)
        sentences = []
        for i in range(0, len(raw_sentences) - 1, 2):
            sentences.append((raw_sentences[i] + raw_sentences[i + 1]).strip())
        if len(raw_sentences) % 2 == 1 and raw_sentences[-1].strip():
            sentences.append(raw_sentences[-1].strip())

        raw_chunks = []
        max_chunk_chars = 110
        for s in sentences:
            if not s:
                continue
            if len(s) <= max_chunk_chars:
                raw_chunks.append(s)
            else:
                # Split further by clause marks
                sub_clauses = re.split(r"([,;:\-–—]+)", s)
                combined_clauses = []
                for j in range(0, len(sub_clauses) - 1, 2):
                    combined_clauses.append((sub_clauses[j] + sub_clauses[j + 1]).strip())
                if len(sub_clauses) % 2 == 1 and sub_clauses[-1].strip():
                    combined_clauses.append(sub_clauses[-1].strip())

                for c in combined_clauses:
                    if not c:
                        continue
                    if len(c) <= max_chunk_chars:
                        raw_chunks.append(c)
                    else:
                        # Split by words
                        words = c.split()
                        cur_words = []
                        cur_len = 0
                        for w in words:
                            if cur_len + len(w) + 1 <= max_chunk_chars:
                                cur_words.append(w)
                                cur_len += len(w) + 1
                            else:
                                if cur_words:
                                    raw_chunks.append(" ".join(cur_words))
                                cur_words = [w]
                                cur_len = len(w)
                        if cur_words:
                            raw_chunks.append(" ".join(cur_words))

        # Merge tiny adjacent chunks without exceeding max_chunk_chars
        chunks = []
        buf = ""
        for ch in raw_chunks:
            if not ch:
                continue
            if not buf:
                buf = ch
            elif len(buf) + len(ch) + 1 <= max_chunk_chars:
                buf = f"{buf} {ch}"
            else:
                chunks.append(buf)
                buf = ch
        if buf:
            chunks.append(buf)

        endpoints = [
            "https://api16-va.tiktokv.com/media/api/text/speech/invoke/",
            "https://api-va.tiktokv.com/media/api/text/speech/invoke/",
            "https://api.tiktokv.com/media/api/text/speech/invoke/",
            "https://api16-normal-c-useast1a.tiktokv.com/media/api/text/speech/invoke/",
        ]
        headers = {
            "User-Agent": "com.zhiliaoapp.musically/2022600030 (Linux; U; Android 7.1.2; es_ES; SM-G988N; Build/NRD90M;tt-ok/3.12.13.1)",
            "Cookie": f"sessionid={session_id}",
        }

        all_audio_data = bytearray()
        for idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            params = {
                "text_speaker": speaker,
                "req_text": chunk,
                "speaker_map_type": "0",
                "aid": "1233",
            }
            encoded_params = urllib.parse.urlencode(params)
            
            chunk_success = False
            last_err = ""
            for ep in endpoints:
                req_url = f"{ep}?{encoded_params}"
                req = urllib.request.Request(req_url, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        resp_json = json.loads(resp.read().decode("utf-8"))
                        if resp_json.get("status_code") == 0 and resp_json.get("data", {}).get("v_str"):
                            v_str = resp_json["data"]["v_str"]
                            chunk_bytes = base64.b64decode(v_str)
                            all_audio_data.extend(chunk_bytes)
                            chunk_success = True
                            break
                        else:
                            last_err = resp_json.get("message", "Non-zero status")
                except Exception as ep_err:
                    last_err = str(ep_err)
                    continue

            if not chunk_success:
                raise VoiceGenerationError(f"CapCut API chunk {idx+1} failed on all endpoints: {last_err}")

        if not all_audio_data:
            raise VoiceGenerationError("CapCut API returned empty audio data")

        out_file.write_bytes(all_audio_data)

    def _synthesize_edge_tts(self, text: str, voice: str, out_file: Path) -> None:
        rate = getattr(self, "_rate", None)
        pitch = getattr(self, "_pitch", None)
        kwargs = {}
        if rate and rate not in ("+0%", "0%"):
            kwargs["rate"] = rate
        if pitch and pitch not in ("+0Hz", "0Hz"):
            kwargs["pitch"] = pitch

        async def _run():
            communicate = edge_tts.Communicate(text, voice, **kwargs)
            await communicate.save(str(out_file))

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(lambda: asyncio.run(_run()))
                    future.result(timeout=180)
            else:
                asyncio.run(_run())
        except Exception as exc:
            raise VoiceGenerationError(f"Edge TTS synthesis failed: {exc}") from exc

    def _run_synthesize(
        self,
        review_result: ReviewResult,
        output_path: Path | None = None,
        *,
        job_logger,
    ) -> Result[VoiceResult, TTSError]:
        if edge_tts is None:
            err_msg = (
                f"Không tìm thấy package edge-tts. Hãy chạy: pip install edge-tts\n"
                f"Chi tiết lỗi: {_import_error}"
            )
            return Result.Err(TTSError(err_msg))

        try:
            self._validate_input(review_result)
        except VoiceGenerationError as exc:
            job_logger.error("Input validation failed: {}", exc)
            return Result.Err(exc)

        script_text = review_result.script.strip()
        job_id = getattr(job_logger, "extra", {}).get("job_id", str(uuid.uuid4())[:8])

        # ---------------------------------------------------------------
        # 1. Detailed Logging Before Calling TTS
        # ---------------------------------------------------------------
        char_count = len(script_text)
        word_count = len(script_text.split())
        first_200 = script_text[:200]
        last_200 = script_text[-200:] if len(script_text) > 200 else script_text

        job_logger.info("=== [TTS INPUT TEXT VERIFICATION] ===")
        job_logger.info("Text length (characters): {}", char_count)
        job_logger.info("Word count: {}", word_count)
        job_logger.info("First 200 chars: {}", first_200)
        job_logger.info("Last 200 chars: {}", last_200)
        job_logger.info("=====================================")

        # ---------------------------------------------------------------
        # 2. Save exact text sent to TTS into data/jobs/<job_id>/tts_input.txt
        # ---------------------------------------------------------------
        job_dir = Path("data") / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        tts_input_file = job_dir / "tts_input.txt"
        try:
            tts_input_file.write_text(script_text, encoding="utf-8")
            job_logger.info("Saved exact TTS input text ({} chars) to: {}", char_count, tts_input_file)
        except Exception as exc:
            job_logger.warning("Could not write tts_input.txt: {}", exc)

        if output_path is not None:
            audio_path = Path(output_path)
            audio_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = self._ensure_output_dir(job_id)
            audio_path = out_dir / f"voice_{int(time.time() * 1000)}.mp3"

        capcut_session_id = (
            getattr(self._config, "capcut_session_id", None)
            or os.environ.get("CAPCUT_SESSION_ID", "")
        ).strip()
        attempt = 0
        start_ts = time.time()
        provider_used = "EdgeTTS"

        while attempt <= self._retry:
            try:
                if capcut_session_id and len(capcut_session_id) >= 16:
                    try:
                        job_logger.info("Synthesizing speech via ByteDance CapCut API (session_id active, voice: {voice})", voice=self._voice)
                        self._synthesize_bytedance_capcut(script_text, capcut_session_id, self._voice, audio_path)
                        provider_used = "CapCutAPI"
                    except Exception as capcut_exc:
                        job_logger.warning("CapCut API attempt failed: {}. Falling back to Edge TTS.", capcut_exc)
                        self._synthesize_edge_tts(script_text, self._voice, audio_path)
                        provider_used = "EdgeTTS-Fallback"
                else:
                    job_logger.info("Synthesizing speech via Edge TTS (voice: {voice})", voice=self._voice)
                    self._synthesize_edge_tts(script_text, self._voice, audio_path)
                    provider_used = "EdgeTTS"

                if not audio_path.exists() or audio_path.stat().st_size == 0:
                    raise VoiceGenerationError("Generated MP3 audio file is missing or empty")

                proc_time = time.time() - start_ts
                out_size = audio_path.stat().st_size
                out_duration = self._get_audio_duration(audio_path)

                job_logger.info("=== [TTS OUTPUT VERIFICATION] ===")
                job_logger.info("Provider: {}", provider_used)
                job_logger.info("Output audio duration: {:.3f}s", out_duration)
                job_logger.info("Output file size: {} bytes", out_size)
                job_logger.info("Saved to: {}", audio_path)
                job_logger.info("=================================")

                metadata = VoiceMetadata(
                    duration=out_duration,
                    sample_rate=24000,
                    provider=provider_used,
                    processing_time=proc_time,
                )
                voice_result = VoiceResult(audio_path=audio_path, metadata=metadata)
                return Result.Ok(voice_result)
            except Exception as exc:
                attempt += 1
                job_logger.warning("TTS attempt {}/{} failed: {}", attempt, self._retry + 1, exc)
                if attempt > self._retry:
                    return Result.Err(TTSError(f"TTS synthesis failed after retries: {exc}"))
                time.sleep(1.0)

        return Result.Err(TTSError("Maximum retry attempts exhausted"))

    def synthesize(
        self,
        script_or_review: str | ReviewResult,
        output_path: Path | None = None,
        *args,
        **kwargs,
    ) -> Result[VoiceResult, TTSError]:
        if isinstance(script_or_review, str):
            review_obj = ReviewResult(
                title="",
                hook="",
                script=script_or_review,
                metadata=None,  # type: ignore
            )
        else:
            review_obj = script_or_review

        job_id = kwargs.get("job_id") or "sync"
        job_logger = self._logger.bind(job_id=job_id)
        return self._run_synthesize(review_obj, output_path=output_path, job_logger=job_logger)

    def synthesize_async(
        self, review_result: ReviewResult
    ) -> Awaitable[Result[VoiceResult, TTSError]]:
        pool = (
            self._thread_pool.get_pool(self._config)
            if isinstance(self._thread_pool, ThreadPoolManager)
            else ThreadPoolManager.get_pool(self._config)
        )
        job_id = f"async-{int(time.time() * 1000)}"
        job_logger = self._logger.bind(job_id=job_id)

        def _task() -> Result[VoiceResult, TTSError]:
            job_logger.info("Starting async Edge TTS synthesis")
            return self._run_synthesize(review_result, job_logger=job_logger)

        return pool.submit(_task)

    def health_check(self) -> Result[bool, TTSError]:
        if edge_tts is None:
            return Result.Err(TTSError(
                f"Không tìm thấy package edge-tts. Hãy chạy: pip install edge-tts\nChi tiết: {_import_error}"
            ))
        try:
            return Result.Ok(True)
        except Exception as exc:
            return Result.Err(TTSError(str(exc)))


# Alias EdgeTTSProvider for clean naming
EdgeTTSProvider = CapCutTTSProvider
