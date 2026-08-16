"""Gemini Review Provider – generates textual review scripts from scene timelines.

The provider:
* expects a ``TimelineResult`` (produced by the timeline builder)
* generates prompt text based on configured prompt templates
* calls Gemini API (``google.generativeai``)
* parses textual output into title, hook and script
* returns a ``Result[ReviewResult, ReviewError]``
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Awaitable, Dict, List, Optional

# Lazy import google.generativeai – do not silently swallow ImportError
_import_error: Optional[str] = None
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse as GenerationResponse
except Exception as exc:  # pragma: no cover
    genai = None  # type: ignore
    GenerationResponse = None  # type: ignore
    _import_error = f"{type(exc).__name__}: {exc}"

from src.core.result import Result
from src.review.base import BaseReviewGenerator
from src.review.exceptions import (
    ReviewError,
    PromptError,
    ReviewGenerationError,
)
from src.review.models import ReviewRequest, ReviewResult, ReviewMetadata
from src.timeline.models import TimelineResult
from src.utils.logger import get_logger
from src.utils.thread_pool import ThreadPoolManager
from src.config.settings import AppConfig, CONFIG


class GeminiReviewProvider(BaseReviewGenerator):
    """Provider that generates a textual review using Gemini.

    The provider does **not** perform any media processing – it merely builds a
    prompt from the supplied timeline, calls the Gemini API and parses the result
    into a :class:`ReviewResult`.
    """

    def __init__(
        self,
        config: AppConfig = CONFIG,
        logger_name: str = "review_gemini",
        thread_pool: ThreadPoolManager | None = None,
        gemini_client=None,
    ) -> None:
        base_logger = get_logger(name=logger_name)
        self._logger = base_logger.bind(module="review", provider="gemini")

        if genai is None:
            self._logger.warning("Package google-generativeai không khả dụng ({}), Gemini API review sẽ bị vô hiệu hóa.", _import_error)
            self._client = None
        else:
            self._config = config
            review_cfg = getattr(config, "review", None) or {}
            self._retry: int = getattr(review_cfg, "retry", 1)
            self._timeout: float = getattr(review_cfg, "timeout", 60.0)
            self._model_name: str = getattr(review_cfg, "model_name", "gemini-1.5-flash-vision")
            self._temperature: float = getattr(review_cfg, "temperature", 0.2)
            self._prompt_templates: Dict[str, str] = getattr(review_cfg, "prompt_templates", {})

            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self._client = gemini_client or genai.GenerativeModel(self._model_name)
                except Exception as exc:
                    self._logger.warning("Lỗi cấu hình Gemini API key: {}", exc)
                    self._client = None
            else:
                self._client = gemini_client
                self._logger.warning("GEMINI_API_KEY chưa được thiết lập (Sẽ chỉ cần nếu dùng API trực tiếp).")

        self._thread_pool = thread_pool

    def _render_prompt(
        self,
        timeline: TimelineResult,
        style: str,
        language: str,
        target_duration: int | None,
    ) -> str:
        template = self._prompt_templates.get(style)
        if not template:
            template = (
                "You are a professional movie reviewer creating a {style} style review in {language}.\n"
                "Here is the timeline of events from the movie:\n{timeline}\n\n"
                "Write a compelling review script with Title, Hook, and Script."
            )
        timeline_str_lines: List[str] = []
        for event in timeline.timeline.events:
            line = (
                f"Scene {event.scene_index}: {event.summary} "
                f"(Objects: {', '.join(event.objects)}; "
                f"Characters: {', '.join(event.characters)}; "
                f"Actions: {', '.join(event.actions)}; "
                f"Emotion: {event.emotion})"
            )
            timeline_str_lines.append(line)
        timeline_str = "\n".join(timeline_str_lines)
        try:
            return template.format(
                timeline=timeline_str,
                language=language,
                target_duration=target_duration or "",
                style=style,
            )
        except Exception as exc:
            raise PromptError(f"Failed to render prompt template: {exc}") from exc

    def _call_shopaikey_text(self, prompt: str, api_key: str, model: str = "gemini-2.5-flash") -> str:
        import requests
        url = f"https://api.shopaikey.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self._temperature, "maxOutputTokens": 4096}
        }
        res = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=45)
        if res.status_code == 200:
            return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raise RuntimeError(f"ShopAIKey REST Error {res.status_code}: {res.text}")

    def _call_gemini(self, prompt: str) -> GenerationResponse:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if api_key.startswith("sk-"):
            models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            last_exc = None
            for m_name in models_to_try:
                try:
                    text_resp = self._call_shopaikey_text(prompt, api_key, model=m_name)
                    class SimpleResp:
                        def __init__(self, text: str):
                            self.text = text
                    return SimpleResp(text_resp)
                except Exception as exc:
                    last_exc = exc
                    self._logger.warning("ShopAIKey Review model '{}' error: {}", m_name, exc)
            raise ReviewGenerationError(str(last_exc))

        if genai is None:
            err_msg = (
                f"Không tìm thấy package google-generativeai. Hãy chạy: pip install google-generativeai\n"
                f"Chi tiết lỗi: {_import_error}"
            )
            raise ReviewGenerationError(err_msg)

        models_to_try = [
            self._model_name,
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
        ]
        unique_models = []
        for m in models_to_try:
            if m and m not in unique_models:
                unique_models.append(m)

        last_exc = None
        for m_name in unique_models:
            try:
                client = genai.GenerativeModel(m_name)
                response = client.generate_content(
                    prompt,
                    generation_config={"temperature": self._temperature},
                )
                return response
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                if "404" in err_str or "not found" in err_str or "not supported" in err_str:
                    self._logger.warning("Model '{}' không khả dụng, thử fallback model tiếp theo...", m_name)
                    continue
                if "rate limit" in err_str:
                    raise ReviewGenerationError(str(exc)) from exc
                raise ReviewGenerationError(str(exc)) from exc

        raise ReviewGenerationError(str(last_exc)) from last_exc

    def _parse_response(self, response: GenerationResponse) -> Dict[str, str]:
        text = getattr(response, "text", "").strip()
        result = {"title": "", "hook": "", "script": ""}
        current_section = None
        script_lines = []
        for line in text.splitlines():
            line_str = line.strip()
            lower_line = line_str.lower()
            if lower_line.startswith("title:") or lower_line.startswith("**title:**") or lower_line.startswith("tiêu đề:"):
                current_section = "title"
                val = line_str.split(":", 1)[1].strip().strip("*")
                result["title"] = val
            elif lower_line.startswith("hook:") or lower_line.startswith("**hook:**") or lower_line.startswith("mở đầu:"):
                current_section = "hook"
                val = line_str.split(":", 1)[1].strip().strip("*")
                result["hook"] = val
            elif lower_line.startswith("script:") or lower_line.startswith("**script:**") or lower_line.startswith("kịch bản:"):
                current_section = "script"
                val = line_str.split(":", 1)[1].strip().strip("*")
                if val:
                    script_lines.append(val)
            elif current_section == "script":
                script_lines.append(line)
            elif not current_section:
                script_lines.append(line)

        if script_lines:
            result["script"] = "\n".join(script_lines).strip()
        if not result["script"]:
            result["script"] = text
        return result

    def _build_review_metadata(
        self,
        style: str,
        language: str,
        target_duration: int | None,
        duration_ms: float,
    ) -> ReviewMetadata:
        word_count = 0
        return ReviewMetadata(
            style=style,
            language=language,
            target_duration=target_duration,
            word_count=word_count,
            generation_duration_ms=duration_ms,
        )

    def _run_generate(
        self,
        timeline: TimelineResult,
        job_logger,
        review_style: str = "documentary",
        language: str = "en",
        target_duration: int | None = None,
    ) -> Result[ReviewResult, ReviewError]:
        start_ts = time.time()
        job_logger.info("Model name:\n{}", self._model_name)
        job_logger.info("Device:\ncloud_api")
        job_logger.info("Compute type:\nfloat32_api")
        attempt = 0
        while attempt <= self._retry:
            try:
                prompt = self._render_prompt(
                    timeline,
                    style=review_style,
                    language=language,
                    target_duration=target_duration,
                )
                job_logger.info("Calling Gemini with style '{style}'", style=review_style)
                resp = self._call_gemini(prompt)
                parsed = self._parse_response(resp)

                duration_ms = (time.time() - start_ts) * 1000
                job_logger.info("Elapsed time: {:.2f}s", duration_ms / 1000.0)
                metadata = self._build_review_metadata(
                    style=review_style,
                    language=language,
                    target_duration=target_duration,
                    duration_ms=duration_ms,
                )
                metadata = ReviewMetadata(
                    style=metadata.style,
                    language=metadata.language,
                    target_duration=metadata.target_duration,
                    word_count=len(parsed["script"].split()),
                    generation_duration_ms=metadata.generation_duration_ms,
                )
                result = ReviewResult(
                    title=parsed["title"],
                    hook=parsed["hook"],
                    script=parsed["script"],
                    metadata=metadata,
                )
                job_logger.info("Successfully generated review script ({words} words)", words=metadata.word_count)
                return Result.Ok(result)
            except ReviewError as re:
                job_logger.error("Review generation failed: {err}", err=re)
                return Result.Err(re)
            except Exception as exc:
                attempt += 1
                if attempt > self._retry:
                    job_logger.exception("Unexpected error during review generation")
                    return Result.Err(ReviewError(str(exc)))
                time.sleep(2 ** attempt)

        return Result.Err(ReviewError("Exceeded max retries for review generation"))

    def generate(
        self,
        timeline_result: TimelineResult,
        review_style: str = "documentary",
        language: str = "en",
        target_duration: int | None = None,
    ) -> Result[ReviewResult, ReviewError]:
        """Generate a review synchronously (BaseReviewGenerator abstract method)."""
        job_logger = self._logger.bind(job_id="sync")
        return self._run_generate(
            timeline_result,
            job_logger=job_logger,
            review_style=review_style,
            language=language,
            target_duration=target_duration,
        )

    generate_sync = generate  # Alias for backward compatibility

    def generate_async(
        self,
        timeline_result: TimelineResult,
        review_style: str = "documentary",
        language: str = "en",
        target_duration: int | None = None,
    ) -> Awaitable[Result[ReviewResult, ReviewError]]:
        """Generate a review asynchronously (BaseReviewGenerator abstract method)."""
        pool = (
            self._thread_pool.get_pool(self._config)
            if isinstance(self._thread_pool, ThreadPoolManager)
            else ThreadPoolManager.get_pool(self._config)
        )
        job_id = f"async-{int(time.time() * 1000)}"
        job_logger = self._logger.bind(job_id=job_id)

        def _task() -> Result[ReviewResult, ReviewError]:
            job_logger.info("Starting async review generation")
            return self._run_generate(
                timeline_result,
                job_logger=job_logger,
                review_style=review_style,
                language=language,
                target_duration=target_duration,
            )

        return pool.submit(_task)

    def health_check(self) -> Result[bool, ReviewError]:
        """Simple health check (BaseReviewGenerator abstract method)."""
        if genai is None:
            return Result.Err(ReviewError(
                f"Không tìm thấy package google-generativeai. Hãy chạy: pip install google-generativeai\nChi tiết: {_import_error}"
            ))
        try:
            _ = self._client.generate_content(
                "Health check",
                generation_config={"temperature": 0.0},
            )
            return Result.Ok(True)
        except Exception as exc:
            return Result.Err(ReviewError(str(exc)))
