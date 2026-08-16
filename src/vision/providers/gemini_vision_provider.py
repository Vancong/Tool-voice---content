"""Gemini Vision based provider for scene‑level visual analysis.

The provider follows the same provider pattern used across the project:
* configuration is injected via ``AppConfig``
* a ``logger`` (Loguru) is bound with static context
* a shared ``ThreadPoolManager`` is used for the async API
* API key is read from the environment (``GEMINI_API_KEY``) – never hard‑coded
* retry, timeout and rate‑limit handling are implemented
* frames are batched per scene – a single Gemini request per scene
* prompt and generation parameters are configurable via ``config.vision``
"""

from __future__ import annotations

import os
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import Awaitable, List, Dict, Optional

import warnings

# Lazy import google.generativeai – do not silently swallow ImportError
_import_error: Optional[str] = None
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import google.generativeai as genai
        from google.generativeai.types import GenerateContentResponse as GenerationResponse
except Exception as exc:  # pragma: no cover
    genai = None  # type: ignore
    GenerationResponse = None  # type: ignore
    _import_error = f"{type(exc).__name__}: {exc}"

from src.core.result import Result
from src.vision.base import BaseVisionAnalyzer
from src.vision.exceptions import (
    VisionError,
    VisionAPIError,
    ImageReadError,
    RateLimitError,
)
from src.vision.models import FrameAnalysis, SceneAnalysis, VisionAnalysisResult
from src.utils.logger import get_logger
from src.utils.thread_pool import ThreadPoolManager
from src.config.settings import AppConfig, CONFIG

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _group_frames_by_scene(frames) -> Dict[int, List[Path]]:
    """Group ``Frame`` objects by their ``scene_index``.

    Returns a mapping ``scene_index -> list of image Path objects``.
    """
    grouped: Dict[int, List[Path]] = defaultdict(list)
    for f in frames:
        grouped[f.scene_index].append(f.image_path)
    return grouped

# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------

class GeminiVisionProvider(BaseVisionAnalyzer):
    """Vision analysis provider that uses Google Gemini's multimodal model.

    The provider expects a ``FrameExtractionResult`` (produced by the frame
    extraction step) and returns a ``VisionAnalysisResult`` containing one
    ``SceneAnalysis`` per scene.
    """

    def __init__(
        self,
        config: AppConfig = CONFIG,
        logger_name: str = "vision_gemini",
        thread_pool: ThreadPoolManager | None = None,
        gemini_client=None,
    ) -> None:
        base_logger = get_logger(name=logger_name)
        self._logger = base_logger.bind(module="vision", provider="gemini")

        if genai is None:
            self._logger.warning("Package google-generativeai không khả dụng ({}), Gemini Vision API sẽ bị vô hiệu hóa.", _import_error)
            self._client = None
        else:
            self._config = config
            vision_cfg = getattr(config, "vision", None) or {}
            self._prompt: str = getattr(vision_cfg, "prompt", "Analyse the following images and provide a concise summary, key objects, characters, actions and dominant emotion.")
            self._batch_size: int = getattr(vision_cfg, "batch_size", 5)
            self._retry: int = getattr(vision_cfg, "retry", 1)
            self._timeout: float = getattr(vision_cfg, "timeout", 60.0)
            self._model_name: str = getattr(vision_cfg, "model_name", "gemini-1.5-flash-vision")
            self._temperature: float = getattr(vision_cfg, "temperature", 0.2)

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

    def _read_image(self, path: Path) -> "genai.types.Image":
        if not path.exists() or not path.is_file():
            raise ImageReadError(f"Image file not found: {path}")
        try:
            return genai.upload_file(str(path))  # type: ignore[attr-defined]
        except Exception as exc:
            raise ImageReadError(str(exc)) from exc

    def _call_shopaikey_vision(self, images: List[Path], api_key: str, model: str = "gemini-2.5-flash") -> str:
        import base64
        import requests
        url = f"https://api.shopaikey.com/v1beta/models/{model}:generateContent?key={api_key}"
        parts = [{"text": self._prompt}]
        for img_path in images:
            if img_path and img_path.exists():
                with open(img_path, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64_data}})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": self._temperature, "maxOutputTokens": 2048}
        }
        res = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=45)
        if res.status_code == 200:
            return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        raise RuntimeError(f"ShopAIKey REST Error {res.status_code}: {res.text}")

    def _call_gemini(self, images: List[Path]) -> GenerationResponse:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if api_key.startswith("sk-"):
            models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            last_exc = None
            for m_name in models_to_try:
                try:
                    text_resp = self._call_shopaikey_vision(images, api_key, model=m_name)
                    class SimpleResp:
                        def __init__(self, text: str):
                            self.text = text
                    return SimpleResp(text_resp)
                except Exception as exc:
                    last_exc = exc
                    self._logger.warning("ShopAIKey Vision model '{}' error: {}", m_name, exc)
            raise VisionAPIError(str(last_exc))

        if genai is None:
            err_msg = (
                f"Không tìm thấy package google-generativeai. Hãy chạy: pip install google-generativeai\n"
                f"Chi tiết lỗi: {_import_error}"
            )
            raise VisionAPIError(err_msg)

        loaded_images = []
        for img_path in images:
            loaded_images.append(genai.upload_file(str(img_path)))  # type: ignore[attr-defined]

        content = [self._prompt] + loaded_images

        models_to_try = [
            self._model_name,
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest",
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
                    content,
                    generation_config={"temperature": self._temperature},
                )
                return response
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                if "404" in err_str or "not found" in err_str or "not supported" in err_str:
                    self._logger.warning("Model '{}' không khả dụng, thử fallback model tiếp theo...", m_name)
                    continue
                if "rate limit" in err_str or "429" in err_str or "quota" in err_str:
                    raise RateLimitError(str(exc)) from exc
                raise VisionAPIError(str(exc)) from exc

        raise VisionAPIError(str(last_exc)) from last_exc

    def _parse_response(self, response: GenerationResponse) -> Dict[str, any]:
        try:
            text = response.text.strip()
            sections = {k.lower(): v.strip() for k, v in (
                line.split(":", 1) for line in text.splitlines() if ":" in line
            )}
            def split_list(value: str) -> List[str]:
                return [item.strip() for item in value.split(",") if item.strip()]
            return {
                "summary": sections.get("summary", ""),
                "objects": split_list(sections.get("objects", "")),
                "characters": split_list(sections.get("characters", "")),
                "actions": split_list(sections.get("actions", "")),
                "emotion": sections.get("emotion", ""),
            }
        except Exception:
            return {"summary": "", "objects": [], "characters": [], "actions": [], "emotion": ""}

    def _analyze_scene(self, scene_index: int, image_paths: List[Path]) -> SceneAnalysis:
        attempt = 0
        while attempt <= self._retry:
            start_ts = time.time()
            try:
                batches = [image_paths[i:i + self._batch_size] for i in range(0, len(image_paths), self._batch_size)]
                aggregated = {
                    "summary": "",
                    "objects": [],
                    "characters": [],
                    "actions": [],
                    "emotion": "",
                }
                for batch in batches:
                    resp = self._call_gemini(batch)
                    parsed = self._parse_response(resp)
                    if parsed["summary"]:
                        aggregated["summary"] += " " + parsed["summary"]
                    aggregated["objects"].extend(parsed["objects"])
                    aggregated["characters"].extend(parsed["characters"])
                    aggregated["actions"].extend(parsed["actions"])
                    if parsed["emotion"]:
                        aggregated["emotion"] = parsed["emotion"]

                return SceneAnalysis(
                    scene_index=scene_index,
                    summary=aggregated["summary"].strip(),
                    key_objects=list(set(aggregated["objects"])),
                    key_characters=list(set(aggregated["characters"])),
                    key_actions=list(set(aggregated["actions"])),
                    dominant_emotion=aggregated["emotion"] or "neutral",
                )
            except RateLimitError as rle:
                attempt += 1
                if attempt > 5:
                    raise
                wait_secs = 15
                self._logger.warning("Vượt quá Rate Limit / Quota (429), tạm dừng {}s để làm mới Quota (lần {}/5)...", wait_secs, attempt)
                time.sleep(wait_secs)
            except Exception as exc:
                raise VisionAPIError(f"Failed to analyze scene {scene_index}: {exc}") from exc

        raise VisionAPIError(f"Exceeded max retries for scene {scene_index}")

    def _run_analysis(
        self, frame_extraction_result, job_logger
    ) -> Result[VisionAnalysisResult, VisionError]:
        try:
            frames = frame_extraction_result.frames
            if not frames:
                job_logger.warning("No frames provided for vision analysis")
                empty_res = VisionAnalysisResult(scenes=[], processing_time=0.0)
                return Result.Ok(empty_res)

            grouped = _group_frames_by_scene(frames)
            scene_analyses: List[SceneAnalysis] = []
            start_total = time.time()

            job_logger.info("Model name:\n{}", self._model_name)
            job_logger.info("Device:\ncloud_api")
            job_logger.info("Compute type:\nfloat32_api")

            for scene_idx, paths in sorted(grouped.items()):
                job_logger.info("Analyzing scene {idx} with {count} frames", idx=scene_idx, count=len(paths))
                analysis = self._analyze_scene(scene_idx, paths)
                scene_analyses.append(analysis)

            total_duration_ms = (time.time() - start_total) * 1000
            elapsed_sec = total_duration_ms / 1000.0
            job_logger.info("Elapsed time: {:.2f}s", elapsed_sec)
            result = VisionAnalysisResult(
                scenes=scene_analyses,
                processing_time=elapsed_sec,
            )
            job_logger.info("Completed vision analysis for {count} scenes in {dur:.2f}ms", count=len(scene_analyses), dur=total_duration_ms)
            return Result.Ok(result)
        except VisionError as ve:
            job_logger.error("Vision analysis failed: {err}", err=ve)
            return Result.Err(ve)
        except Exception as exc:
            job_logger.exception("Unexpected error during vision analysis")
            return Result.Err(VisionError(str(exc)))

    def analyze(
        self, frame_extraction_result
    ) -> Result[VisionAnalysisResult, VisionError]:
        """Synchronous analysis of extracted frames (BaseVisionAnalyzer abstract method)."""
        job_logger = self._logger.bind(job_id="sync")
        return self._run_analysis(frame_extraction_result, job_logger=job_logger)

    def analyze_frames(
        self, frame_extraction_result
    ) -> Result[VisionAnalysisResult, VisionError]:
        """Alias for analyze(frame_extraction_result)."""
        return self.analyze(frame_extraction_result)

    analyze_sync = analyze  # Alias for backward compatibility

    def analyze_async(
        self, frame_extraction_result
    ) -> Awaitable[Result[VisionAnalysisResult, VisionError]]:
        """Asynchronous analysis using the shared thread‑pool (BaseVisionAnalyzer abstract method)."""
        pool = (
            self._thread_pool.get_pool(self._config)
            if isinstance(self._thread_pool, ThreadPoolManager)
            else ThreadPoolManager.get_pool(self._config)
        )
        job_id = f"async-{int(time.time() * 1000)}"
        job_logger = self._logger.bind(job_id=job_id)

        def _task() -> Result[VisionAnalysisResult, VisionError]:
            job_logger.info("Starting async vision analysis")
            return self._run_analysis(frame_extraction_result, job_logger=job_logger)

        return pool.submit(_task)

    def health_check(self) -> Result[bool, VisionError]:
        """Verify that Gemini Vision API is reachable (BaseVisionAnalyzer abstract method)."""
        if genai is None:
            return Result.Err(VisionError(
                f"Không tìm thấy package google-generativeai. Hãy chạy: pip install google-generativeai\nChi tiết: {_import_error}"
            ))
        try:
            resp = self._client.generate_content("Health check", generation_config={"temperature": 0.0})
            _ = resp.text
            return Result.Ok(True)
        except Exception as exc:
            return Result.Err(VisionError(str(exc)))
