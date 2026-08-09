"""Gemini Web Provider integrating Playwright browser automation into the review generation pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Awaitable, Optional, Callable

from src.core.result import Result
from src.gemini_web.base import BaseGeminiWeb
from src.gemini_web.browser_manager import BrowserManager
from src.gemini_web.exceptions import GeminiWebError
from src.gemini_web.models import GeminiWebConfig, SessionStatus
from src.gemini_web.prompt_builder import PromptBuilder
from src.gemini_web.response_parser import ResponseParser
from src.gemini_web.session_manager import SessionManager
from src.review.base import BaseReviewGenerator
from src.review.exceptions import ReviewError
from src.review.models import ReviewResult
from src.timeline.models import TimelineResult
from src.utils.logger import get_logger
from src.utils.thread_pool import ThreadPoolManager


class GeminiWebProvider(BaseReviewGenerator, BaseGeminiWeb):
    """Primary review generator backend using automated browser session on Gemini Web."""

    def __init__(
        self,
        config: Optional[GeminiWebConfig] = None,
        thread_pool: Optional[ThreadPoolManager] = None,
    ) -> None:
        self._config = config or GeminiWebConfig()
        self._session_mgr = SessionManager(self._config)
        self._browser_mgr = BrowserManager(self._config, self._session_mgr)
        self._thread_pool = thread_pool
        self._logger = get_logger(name="gemini_web_provider").bind(module="gemini_web", provider="playwright")

    @property
    def browser_manager(self) -> BrowserManager:
        return self._browser_mgr

    @property
    def session_manager(self) -> SessionManager:
        return self._session_mgr

    def get_session_status(self) -> SessionStatus:
        """Check current session status."""
        return self._session_mgr.get_status()

    def import_cookies(self, raw_input: str) -> int:
        """Import cookies and hot-reload them into live browser context without app restart."""
        count = self._session_mgr.import_cookies_from_raw_string(raw_input)
        if count > 0:
            self._browser_mgr.reload_cookies()
        return count

    def login_interactive(self) -> Result[bool, GeminiWebError]:
        """Open interactive browser window for user to log in."""
        try:
            ok = self._browser_mgr.login_interactive()
            return Result.Ok(ok) if ok else Result.Err(GeminiWebError("Đăng nhập quá thời gian chờ."))
        except Exception as exc:
            return Result.Err(GeminiWebError(str(exc)))

    def clear_session(self) -> Result[bool, GeminiWebError]:
        """Clear session files and cookies."""
        try:
            self._browser_mgr.close()
            ok = self._session_mgr.clear_session()
            return Result.Ok(ok)
        except Exception as exc:
            return Result.Err(GeminiWebError(str(exc)))

    def generate_review(
        self,
        timeline: TimelineResult,
        review_style: str = "documentary",
        language: str = "vi",
        target_duration: Optional[int] = None,
        custom_instructions: Optional[str] = None,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Result[ReviewResult, GeminiWebError]:
        """Generate review script using Gemini Web chat automation."""
        start_ts = time.time()
        try:
            prompt = PromptBuilder.build_review_prompt(
                timeline=timeline,
                style=review_style,
                language=language,
                target_duration=target_duration,
                custom_instructions=custom_instructions,
            )

            self._logger.info("Generated prompt for Gemini Web ({} chars)", len(prompt))
            web_resp = self._browser_mgr.send_prompt(
                prompt=prompt,
                job_id=job_id,
                progress_callback=progress_callback,
            )

            elapsed = time.time() - start_ts
            review_res = ResponseParser.parse(
                raw_text=web_resp.text,
                processing_time=elapsed,
                model_name=web_resp.model_name,
            )
            self._logger.info(
                "Successfully parsed ReviewResult from Gemini Web ({} words)",
                review_res.metadata.total_words,
            )
            return Result.Ok(review_res)
        except GeminiWebError as gwe:
            self._logger.error("Gemini Web error during review generation: {}", gwe)
            return Result.Err(gwe)
        except Exception as exc:
            self._logger.error("Unexpected error during Gemini Web generation: {}", exc)
            return Result.Err(GeminiWebError(str(exc)))

    def generate(
        self,
        timeline_result: TimelineResult,
        review_style: str = "documentary",
        language: str = "vi",
        target_duration: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Result[ReviewResult, ReviewError]:
        """Synchronous generation conforming to BaseReviewGenerator."""
        return self.generate_review(
            timeline=timeline_result,
            review_style=kwargs.get("style", review_style),
            language=kwargs.get("lang", language),
            target_duration=kwargs.get("target_duration", target_duration),
            custom_instructions=kwargs.get("custom_instructions"),
            job_id=kwargs.get("job_id"),
            progress_callback=kwargs.get("progress_callback"),
        )

    def generate_async(
        self,
        timeline_result: TimelineResult,
        review_style: str = "documentary",
        language: str = "vi",
        target_duration: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Awaitable[Result[ReviewResult, ReviewError]]:
        """Asynchronous generation via thread pool."""
        pool = (
            self._thread_pool.get_pool()
            if isinstance(self._thread_pool, ThreadPoolManager)
            else ThreadPoolManager.get_pool()
        )

        def _task():
            return self.generate(
                timeline_result=timeline_result,
                review_style=review_style,
                language=language,
                target_duration=target_duration,
                *args,
                **kwargs,
            )

        return pool.submit(_task)

    def health_check(self) -> Result[bool, ReviewError]:
        """Check if session is active or browser is functional."""
        try:
            status = self._browser_mgr.check_is_logged_in()
            if status == SessionStatus.LOGGED_IN:
                return Result.Ok(True)
            return Result.Err(ReviewError("⚠️ Cookie Gemini Web chưa đăng nhập hoặc đã hết hạn. Vui lòng nhập lại cookie!"))
        except Exception as exc:
            return Result.Err(ReviewError("⚠️ Cookie Gemini Web chưa đăng nhập hoặc đã hết hạn. Vui lòng nhập lại cookie!"))

    def close(self) -> None:
        self._browser_mgr.close()
