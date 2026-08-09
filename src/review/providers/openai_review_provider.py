# -*- coding: utf-8 -*-
"""
src/review/providers/openai_review_provider.py

Concrete review generation provider utilizing OpenAI API (ChatGPT / GPT-4o).
Supports Few-Shot sample styles, custom writing instructions, and robust retry logic.
"""

from __future__ import annotations

import os
import time
from typing import Any, Awaitable, Dict, List, Optional

import requests

from src.config.settings import AppConfig, CONFIG
from src.core.result import Result
from src.review.base import BaseReviewGenerator
from src.review.exceptions import (
    ReviewError,
    ReviewAPIError,
    ReviewParseError,
    ReviewValidationError,
)
from src.review.models import ReviewRequest, ReviewResult, ReviewMetadata
from src.utils.logger import get_logger
from src.utils.thread_pool import ThreadPoolManager
from src.agents.sample_styles import get_sample_style_context

_logger = get_logger("openai_review")


class OpenAIReviewProvider(BaseReviewGenerator):
    """Review generator powered by OpenAI ChatGPT (GPT-4o / GPT-4o-mini)."""

    def __init__(
        self,
        config: AppConfig = CONFIG,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        sample_style: Optional[str] = None,
        custom_sample_text: Optional[str] = None,
        thread_pool: Optional[ThreadPoolManager] = None,
    ) -> None:
        self._config = config
        self._api_key = api_key or getattr(config, "openai_api_key", "") or os.getenv("OPENAI_API_KEY", "")
        self._model_name = model_name or "gpt-4o"
        self._sample_style = sample_style
        self._custom_sample_text = custom_sample_text
        self._thread_pool = thread_pool or ThreadPoolManager()
        self._timeout = 300.0

    def health_check(self) -> Result[bool, ReviewError]:
        """Check if OpenAI API key is configured."""
        if not self._api_key or not self._api_key.strip():
            return Result.Err(ReviewAPIError("OPENAI_API_KEY is not configured."))
        return Result.Ok(True)

    def generate_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Direct call to OpenAI Chat Completion API."""
        if not self._api_key:
            raise ReviewAPIError("OpenAI API Key is missing. Please set OPENAI_API_KEY.")

        headers = {
            "Authorization": f"Bearer {self._api_key.strip()}",
            "Content-Type": "application/json",
        }

        # Include sample style context in system prompt if specified
        style_ctx = get_sample_style_context(self._sample_style, self._custom_sample_text)
        full_system_prompt = system_prompt
        if style_ctx:
            full_system_prompt = f"{system_prompt}\n\n{style_ctx}"

        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = "https://api.openai.com/v1/chat/completions"
        _logger.info("Sending request to OpenAI API (model: {})...", self._model_name)

        resp = requests.post(url, headers=headers, json=payload, timeout=self._timeout)
        if resp.status_code != 200:
            err_body = resp.text
            _logger.error("OpenAI API error [{}]: {}", resp.status_code, err_body)
            raise ReviewAPIError(f"OpenAI API error {resp.status_code}: {err_body}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content

    def generate(self, timeline_result) -> Result[ReviewResult, ReviewError]:
        """Generate review script from TimelineResult synchronously."""
        try:
            start_time = time.time()
            events = getattr(timeline_result, "events", [])
            timeline_text = "\n".join(
                f"- Scene {ev.scene_index + 1}: {ev.summary} (Cảm xúc: {ev.emotion}, Nhân vật: {', '.join(ev.characters)})"
                for ev in events
            )

            system_prompt = (
                "Bạn là chuyên gia biên kịch video review phim chuyên nghiệp trên YouTube. "
                "Nhiệm vụ của bạn là viết một bài kịch bản review phim hoàn chỉnh, mạch lạc, hấp dẫn, "
                "khớp từng phân cảnh và giữ chân khán giả từ đầu đến cuối."
            )

            user_prompt = (
                f"Dưới đây là dòng thời gian và dữ liệu phân cảnh của bộ phim:\n\n"
                f"{timeline_text}\n\n"
                f"Hãy viết một kịch bản review phim hoàn chỉnh bằng tiếng Việt, chia thành các đoạn thuyết minh "
                f"hấp dẫn, không được ghi chú tiêu đề thừa, chỉ viết đúng lời đọc (narration text) tự nhiên."
            )

            script_text = self.generate_chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            proc_time = time.time() - start_time
            words = len(script_text.split())
            meta = ReviewMetadata(
                total_words=words,
                estimated_duration=words / 2.5,
                model_name=self._model_name,
                processing_time=proc_time,
            )

            return Result.Ok(ReviewResult(
                script=script_text,
                metadata=meta,
            ))

        except Exception as exc:
            _logger.exception("Failed to generate review with OpenAI: {}", exc)
            return Result.Err(ReviewError(f"OpenAI review generation failed: {exc}"))

    def generate_async(self, timeline_result) -> Awaitable[Result[ReviewResult, ReviewError]]:
        """Asynchronous generation via thread pool."""
        return self._thread_pool.submit(self.generate, timeline_result)
