# -*- coding: utf-8 -*-
"""
src/tts/providers/elevenlabs_tts_provider.py

Concrete Text-to-Speech provider powered by ElevenLabs API or Playwright Chrome Browser Web Session.
Provides studio-grade, emotionally rich AI voices (including Voice Cloning support)
with automatic text chunking and audio concatenation.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import time
import uuid
import threading
from pathlib import Path
from typing import Awaitable, Dict, List, Optional, Any

import requests

from src.config.settings import AppConfig, CONFIG
from src.core.result import Result
from src.review.models import ReviewResult
from src.tts.base import BaseTTS
from src.tts.exceptions import (
    TTSError,
    VoiceGenerationError,
    VoiceProviderError,
    RateLimitError,
)
from src.tts.models import VoiceMetadata, VoiceRequest, VoiceResult
from src.utils.logger import get_logger
from src.utils.thread_pool import ThreadPoolManager

_logger = get_logger("elevenlabs_tts")

# Popular ElevenLabs Builtin Voice IDs
ELEVENLABS_VOICE_MAP = {
    "kat": "RiK8PTtVIeKKoFFTk9fg",     # Kat (Nữ Sharp Educator)
    "parker": "Dnd9VXpAjEGXiRGBf1O6",  # Parker (Nam Professional)
    "adam": "pNInz6obpgDQGcFmaJgB",    # Adam (Nam - Free API)
}


class ElevenLabsTTSProvider(BaseTTS):
    """Studio-grade Text-to-Speech provider using ElevenLabs (API Key or Playwright Chrome Web Session)."""

    _pw_lock = threading.RLock()
    _pw_playwright = None
    _pw_context = None
    _pw_page = None

    def __init__(
        self,
        config: AppConfig = CONFIG,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_multilingual_v2",
        thread_pool: Optional[ThreadPoolManager] = None,
    ) -> None:
        self._config = config
        self._api_key = api_key or getattr(config, "elevenlabs_api_key", "") or os.getenv("ELEVENLABS_API_KEY", "")
        self._model_id = model_id or "eleven_multilingual_v2"
        self._thread_pool = thread_pool or ThreadPoolManager()
        self._voice_id = self._resolve_voice_id(voice_id or getattr(config.tts, "voice_id", "RiK8PTtVIeKKoFFTk9fg"))

    def _resolve_voice_id(self, voice_name_or_id: str) -> str:
        """Resolve voice name/alias or formatted string to ElevenLabs Voice ID."""
        clean = (voice_name_or_id or "").strip()
        if not clean:
            return "RiK8PTtVIeKKoFFTk9fg"

        # 1. Extract Voice ID inside parentheses at the end if present, e.g. "Giọng Nam (pNInz6obpgDQGcFmaJgB)"
        match = re.search(r"\(([\w\-]+)\)\s*$", clean)
        if match:
            return match.group(1)

        # 2. Check built-in map
        lowered = clean.lower()
        if lowered in ELEVENLABS_VOICE_MAP:
            return ELEVENLABS_VOICE_MAP[lowered]
        for k, v in ELEVENLABS_VOICE_MAP.items():
            if k in lowered:
                return v

        # 3. Check saved custom voices configuration
        voice_file = Path("config/elevenlabs_voices.json")
        if voice_file.exists():
            try:
                custom_voices = json.loads(voice_file.read_text(encoding="utf-8"))
                for item in custom_voices:
                    if item.get("name") == clean or item.get("id") == clean:
                        return item.get("id", clean)
            except Exception:
                pass

        return clean

    @classmethod
    def save_cookie_state(cls, raw_input: str) -> bool:
        """Parse raw cookie string or JSON array and save to data/elevenlabs_session.json."""
        raw_input = (raw_input or "").strip()
        if not raw_input:
            return False

        session_file = Path("data/elevenlabs_session.json")
        session_file.parent.mkdir(parents=True, exist_ok=True)

        cookies_list = []
        origins_list = []

        if raw_input.startswith("[") or raw_input.startswith("{"):
            try:
                data = json.loads(raw_input)
                if isinstance(data, dict):
                    if "cookies" in data or "origins" in data:
                        cookies_list = data.get("cookies", [])
                        origins_list = data.get("origins", [])
                    else:
                        # Raw localStorage JSON dictionary exported from F12 console
                        ls_items = [{"name": str(k), "value": str(v)} for k, v in data.items()]
                        origins_list.append({
                            "origin": "https://elevenlabs.io",
                            "localStorage": ls_items
                        })
                elif isinstance(data, list):
                    raw_cookies = data
                    for c in raw_cookies:
                        if isinstance(c, dict) and "name" in c and "value" in c:
                            domain = c.get("domain") or ".elevenlabs.io"
                            if not domain.startswith("."):
                                domain = "." + domain.lstrip(".")
                            
                            cookie_obj = {
                                "name": str(c["name"]).strip(),
                                "value": str(c["value"]).strip(),
                                "domain": domain,
                                "path": c.get("path") or "/",
                                "secure": bool(c.get("secure", True)),
                                "httpOnly": bool(c.get("httpOnly", False)),
                            }
                            same_site = c.get("sameSite")
                            if same_site in ("Strict", "Lax", "None"):
                                cookie_obj["sameSite"] = same_site
                            elif same_site == "no_restriction":
                                cookie_obj["sameSite"] = "None"
                            
                            if "expirationDate" in c or "expires" in c:
                                exp = c.get("expirationDate") or c.get("expires")
                                if exp and float(exp) > 0:
                                    cookie_obj["expires"] = float(exp)
                            cookies_list.append(cookie_obj)
            except Exception as exc:
                _logger.warning("Error parsing ElevenLabs JSON: {}", exc)

        if not cookies_list and not origins_list and "=" in raw_input:
            for item in raw_input.split(";"):
                item = item.strip()
                if "=" in item:
                    k, v = item.split("=", 1)
                    cookies_list.append({
                        "name": k.strip(),
                        "value": v.strip(),
                        "domain": ".elevenlabs.io",
                        "path": "/",
                        "secure": True,
                        "sameSite": "None",
                    })

        state_data = {"cookies": cookies_list, "origins": origins_list}
        session_file.write_text(json.dumps(state_data, indent=2), encoding="utf-8")
        _logger.info("Saved {} cookies and {} origins to {}", len(cookies_list), len(origins_list), session_file)
        return (len(cookies_list) + len(origins_list)) > 0

    @classmethod
    def _ensure_browser(cls, api_key: str, navigate_url: str = "https://elevenlabs.io/app/speech-synthesis/text-to-speech") -> Any:
        """Launch persistent Playwright Chrome browser for ElevenLabs Web session."""
        with cls._pw_lock:
            if cls._pw_context is not None and cls._pw_page is not None:
                try:
                    if not cls._pw_page.is_closed():
                        if navigate_url and cls._pw_page.url != navigate_url:
                            cls._pw_page.goto(navigate_url, timeout=20000, wait_until="domcontentloaded")
                        return cls._pw_context
                except Exception:
                    pass

            profile_dir_path = Path("data/browser_profile/elevenlabs_session").resolve()
            profile_dir_path.mkdir(parents=True, exist_ok=True)

            # Clean up stale Chromium lock files
            for lock_name in ["SingletonLock", "SingletonCookie", "SingletonSocket"]:
                lock_file = profile_dir_path / lock_name
                if lock_file.exists():
                    try:
                        lock_file.unlink()
                    except Exception:
                        pass

            # Kill any orphan chrome processes holding lock on profile dir
            profile_path_str = str(profile_dir_path).lower()
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        name = (proc.info.get('name') or '').lower()
                        if 'chrome' in name or 'msedge' in name or 'chromium' in name:
                            cmdline = ' '.join(proc.info.get('cmdline') or []).lower()
                            if 'elevenlabs_session' in cmdline or profile_path_str in cmdline:
                                proc.kill()
                    except Exception:
                        pass
            except Exception:
                pass

            session_file = Path("data/elevenlabs_session.json")
            if not session_file.exists() or session_file.stat().st_size < 10:
                cls.save_cookie_state(api_key)

            from playwright.sync_api import sync_playwright
            if cls._pw_playwright is None:
                cls._pw_playwright = sync_playwright().start()

            _logger.info("Launching Playwright Chrome browser for ElevenLabs Web session...")
            
            args = [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-infobars",
                "--start-maximized",
            ]
            user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            )

            launch_kwargs = {
                "user_data_dir": str(profile_dir_path),
                "headless": False,
                "args": args,
                "viewport": {"width": 1280, "height": 800},
                "user_agent": user_agent,
                "timeout": 20000,
            }

            try:
                cls._pw_context = cls._pw_playwright.chromium.launch_persistent_context(
                    channel="chrome", **launch_kwargs
                )
            except Exception:
                cls._pw_context = cls._pw_playwright.chromium.launch_persistent_context(
                    **launch_kwargs
                )

            session_file = Path("data/elevenlabs_session.json")
            if session_file.exists() and session_file.stat().st_size > 10:
                try:
                    state_data = json.loads(session_file.read_text(encoding="utf-8"))
                    cookies = state_data.get("cookies", [])
                    if cookies:
                        cls._pw_context.add_cookies(cookies)
                        _logger.info("Injected {} cookies from data/elevenlabs_session.json into Playwright context.", len(cookies))
                except Exception as exc:
                    _logger.warning("Error loading cookies from data/elevenlabs_session.json: {}", exc)

            pages = cls._pw_context.pages
            cls._pw_page = pages[0] if len(pages) > 0 else cls._pw_context.new_page()
            try:
                cls._pw_page.goto(navigate_url, timeout=25000, wait_until="domcontentloaded")
                cls._pw_page.bring_to_front()
            except Exception:
                pass
            
            return cls._pw_context

    @classmethod
    def login_interactive(cls) -> bool:
        """Launch ElevenLabs Chrome browser outside review process so user can log in."""
        try:
            cls._ensure_browser("cookie_session", navigate_url="https://elevenlabs.io/app/sign-in")
            with cls._pw_lock:
                if cls._pw_page:
                    try:
                        cls._pw_page.bring_to_front()
                    except Exception:
                        pass
            return True
        except Exception as exc:
            _logger.error("Failed to launch interactive ElevenLabs browser: {}", exc)
            return False

    def health_check(self) -> Result[bool, TTSError]:
        """Check ElevenLabs connection via API Key (xi-api-key)."""
        api_key = (self._api_key or "").strip()
        if not api_key:
            return Result.Err(VoiceProviderError(
                "❌ [Lỗi ElevenLabs TTS] Chưa nhập API Key ElevenLabs!\n\n"
                "👉 Vui lòng bấm nút '🔑 Nhập API Key ElevenLabs' trên giao diện để bổ sung (Key bắt đầu bằng 'sk_...')."
            ))

        if not api_key.startswith("sk_"):
            return Result.Err(VoiceProviderError(
                "❌ [Lỗi ElevenLabs TTS] API Key ElevenLabs không hợp lệ!\n\n"
                "👉 API Key chuẩn phải bắt đầu bằng chữ 'sk_...'. Vui lòng bấm nút '🔑 Nhập API Key ElevenLabs' để nhập lại."
            ))

        url = "https://api.elevenlabs.io/v1/user"
        headers = {"xi-api-key": api_key}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                _logger.info("ElevenLabs API Key verified successfully ✓")
                return Result.Ok(True)
            elif resp.status_code == 401:
                return Result.Err(VoiceProviderError(
                    "❌ [Lỗi ElevenLabs TTS] API Key sai hoặc không có quyền truy cập (Lỗi 401)!\n\n"
                    "👉 Vui lòng kiểm tra lại Key trên trang elevenlabs.io và bấm '🔑 Nhập API Key ElevenLabs' để cập nhật."
                ))
            else:
                return Result.Err(VoiceProviderError(
                    f"❌ [Lỗi ElevenLabs API Key] Mã phản hồi: {resp.status_code} - {resp.text[:200]}"
                ))
        except Exception as exc:
            return Result.Err(VoiceProviderError(f"❌ Không thể kết nối tới ElevenLabs: {exc}"))

    @staticmethod
    def _split_text_into_chunks(text: str, max_chars: int = 2500) -> List[str]:
        """Split narration text into natural paragraph/sentence chunks."""
        text = text.strip()
        if len(text) <= max_chars:
            return [text]

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if current_len + len(p) + 2 > max_chars:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                if len(p) > max_chars:
                    sentences = re.split(r"(?<=[.!?。])\s+", p)
                    for s in sentences:
                        s = s.strip()
                        if not s:
                            continue
                        if current_len + len(s) + 1 > max_chars:
                            if current_chunk:
                                chunks.append(" ".join(current_chunk))
                                current_chunk = []
                                current_len = 0
                        current_chunk.append(s)
                        current_len += len(s) + 1
                else:
                    current_chunk.append(p)
                    current_len += len(p) + 2
            else:
                current_chunk.append(p)
                current_len += len(p) + 2

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _synthesize_chunk(self, chunk_text: str, voice_id: str) -> bytes:
        """Call ElevenLabs API or Playwright Chrome session for a single chunk."""
        is_cookie = "=" in self._api_key or self._api_key.strip().startswith("[")
        if is_cookie:
            ctx = self._ensure_browser(self._api_key)
            url = f"https://elevenlabs.io/app/api/speech-synthesis/v1/text-to-speech/{voice_id}/stream"
            payload = {
                "text": chunk_text,
                "model_id": self._model_id,
                "voice_settings": {
                    "stability": 0.50,
                    "similarity_boost": 0.80,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
            }
            with self._pw_lock:
                res = ctx.request.post(
                    url,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "audio/mpeg",
                    }
                )
            if res.status == 429:
                raise RateLimitError("ElevenLabs quota exceeded or rate limited.")
            if res.status != 200 or len(res.body()) < 100:
                raise VoiceGenerationError(f"ElevenLabs Web error ({res.status}): {res.text()[:200]}")
            return res.body()
        else:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": self._api_key.strip(),
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            }
            payload = {
                "text": chunk_text,
                "model_id": self._model_id,
                "voice_settings": {
                    "stability": 0.50,
                    "similarity_boost": 0.80,
                    "style": 0.0,
                    "use_speaker_boost": True,
                },
            }
            try:
                from curl_cffi import requests as c_requests
                resp = c_requests.post(url, headers=headers, json=payload, timeout=120, impersonate="chrome120")
            except ImportError:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)

            if resp.status_code == 429:
                raise RateLimitError("ElevenLabs quota exceeded or rate limited.")
            if resp.status_code == 402 or "payment_required" in resp.text.lower():
                raise VoiceGenerationError(
                    "❌ Tài khoản API ElevenLabs miễn phí không hỗ trợ giọng Thư viện trả phí (Kat/Parker)!\n\n"
                    "👉 Vui lòng đổi sang chọn giọng Miễn Phí như 'ElevenLabs: Rachel (Nữ - Free API)' hoặc 'ElevenLabs: Adam (Nam - Free API)' trên giao diện!"
                )
            if resp.status_code != 200:
                raise VoiceGenerationError(f"ElevenLabs API error {resp.status_code}: {resp.text}")

            return resp.content

    def synthesize(
        self,
        review_result: ReviewResult | str,
        output_path: Optional[Path] = None,
        *args,
        **kwargs,
    ) -> Result[VoiceResult, TTSError]:
        """Synthesize speech from ReviewResult or text."""
        try:
            if not self._api_key:
                return Result.Err(VoiceProviderError("ELEVENLABS_API_KEY is not set."))

            start_time = time.time()
            text = review_result.script if isinstance(review_result, ReviewResult) else str(review_result)
            text = text.strip()
            if not text:
                return Result.Err(VoiceGenerationError("Script text is empty."))

            if output_path is not None:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_dir = Path(self._config.tts.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                output_file = out_dir / f"elevenlabs_{uuid.uuid4().hex[:8]}.mp3"

            chunks = self._split_text_into_chunks(text, max_chars=2500)
            _logger.info("ElevenLabs synthesizing {} chars across {} chunk(s)...", len(text), len(chunks))

            audio_segments: List[bytes] = []
            for i, chunk in enumerate(chunks):
                _logger.info("Synthesizing chunk {}/{} ({} chars)...", i + 1, len(chunks), len(chunk))
                audio_bytes = self._synthesize_chunk(chunk, self._voice_id)
                audio_segments.append(audio_bytes)

            with output_file.open("wb") as fh:
                for seg in audio_segments:
                    fh.write(seg)

            duration = self._probe_audio_duration(output_file)
            proc_time = time.time() - start_time

            _logger.info("ElevenLabs synthesis complete: {}s audio saved to {}", round(duration, 2), output_file)

            meta = VoiceMetadata(
                duration=duration,
                sample_rate=44100,
                provider="ElevenLabs",
                processing_time=proc_time,
            )

            return Result.Ok(VoiceResult(
                audio_path=output_file,
                metadata=meta,
            ))

        except Exception as exc:
            _logger.exception("ElevenLabs TTS synthesis failed: {}", exc)
            return Result.Err(VoiceGenerationError(f"ElevenLabs TTS failed: {exc}"))

    def synthesize_async(
        self,
        review_result: ReviewResult | str,
        output_path: Optional[Path] = None,
        *args,
        **kwargs,
    ) -> Awaitable[Result[VoiceResult, TTSError]]:
        """Asynchronous synthesis."""
        return self._thread_pool.submit(self.synthesize, review_result, output_path, *args, **kwargs)

    @staticmethod
    def _probe_audio_duration(file_path: Path) -> float:
        """Probe audio duration in seconds using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(res.stdout.strip())
        except Exception:
            size = file_path.stat().st_size
            return size / 16000.0
