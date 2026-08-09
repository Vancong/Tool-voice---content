"""Prompt builder for Gemini Web chat review generation.

Includes token estimation, warning, and automatic intelligent prompt trimming.
"""

from __future__ import annotations

from typing import Optional, Any
from src.utils.logger import get_logger

_logger = get_logger("gemini_prompt_builder")


class PromptBuilder:
    """Builds optimized prompts from movie timeline for Gemini Web chat with token estimation."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for Vietnamese/English text (~3.5 characters per token)."""
        if not text:
            return 0
        return max(1, int(len(text) / 3.5))

    @classmethod
    def build_review_prompt(
        cls,
        timeline: Any,
        style: str = "documentary",
        language: str = "vi",
        target_duration: Optional[int] = None,
        custom_instructions: Optional[str] = None,
        max_prompt_tokens: int = 15000,
    ) -> str:
        """Construct a comprehensive prompt for movie review generation."""
        # Extract events / scenes from TimelineResult or MovieTimeline
        events = []
        total_duration = 0.0

        if hasattr(timeline, "timeline") and hasattr(timeline.timeline, "events"):
            events = timeline.timeline.events
            total_duration = getattr(timeline.timeline, "duration", 0.0)
        elif hasattr(timeline, "events"):
            events = timeline.events
            total_duration = getattr(timeline, "duration", 0.0)
        elif hasattr(timeline, "scenes"):
            events = timeline.scenes
            total_duration = getattr(timeline, "total_duration", 0.0)

        num_events = len(events)
        all_chars = set()

        scenes_summary_lines = []
        for s in events[:30]:  # include key events
            s_idx = getattr(s, "scene_index", 0) + 1
            s_start = getattr(s, "start_time", 0.0)
            s_end = getattr(s, "end_time", 0.0)
            s_desc = getattr(s, "description", "") or getattr(s, "summary", "") or ""
            chars = getattr(s, "characters", []) or []
            for c in chars:
                if c:
                    all_chars.add(c)
            s_char = ", ".join(chars)
            line = f"- Cảnh {s_idx} ({s_start:.1f}s - {s_end:.1f}s): {s_desc}"
            if s_char:
                line += f" [Nhân vật: {s_char}]"
            scenes_summary_lines.append(line)

        scenes_text = "\n".join(scenes_summary_lines) if scenes_summary_lines else "Diễn biến các phân cảnh trong video."

        # 2. Characters list
        chars_text = ", ".join(all_chars) if all_chars else "Các nhân vật trong video"

        # 3. Audio transcript / STT
        transcript = getattr(timeline, "transcript", "") or ""
        if not transcript:
            dialogues = [getattr(s, "dialogue", "") for s in events if getattr(s, "dialogue", "")]
            transcript = " ".join(dialogues)

        # Truncate transcript to reasonable size if too long
        transcript_preview = transcript[:3000] if transcript else "Không có lời thoại gốc trích xuất."

        # 4. Target length hint
        words_hint = "khoảng 300 - 600 từ"
        if target_duration:
            approx_words = int((target_duration / 60) * 150)
            words_hint = f"khoảng {approx_words} từ (tương ứng thời lượng {target_duration} giây)"

        lang_name = "tiếng Việt" if language.startswith("vi") else language

        prompt = f"""Bạn là một chuyên gia review phim hàng đầu trên YouTube và TikTok.
Hãy viết một kịch bản review phim hoàn chỉnh, cuốn hút, hấp dẫn người nghe từ đầu đến cuối dựa trên dữ liệu video dưới đây:

--- THÔNG TIN VIDEO & TIMELINE ---
- Tổng thời lượng: {total_duration:.1f} giây ({num_events} phân cảnh)
- Nhân vật xuất hiện: {chars_text}
- Diễn biến các cảnh quay chính:
{scenes_text}

- Lời thoại / Transcript trích xuất:
{transcript_preview}

--- YÊU CẦU KỊCH BẢN ---
- Phong cách viết: {style.upper()} (lôi cuốn, kịch tính, nhịp điệu mượt mà).
- Ngôn ngữ: {lang_name}.
- Độ dài kịch bản: {words_hint}.
- Lời văn tự nhiên để đọc lồng tiếng (TTS), không dùng các ký hiệu thừa, không chèn ghi chú thời gian như [00:01] vào phần lời đọc.
{f"- Yêu cầu bổ sung: {custom_instructions}" if custom_instructions else ""}

--- ĐỊNH DẠNG ĐẦU RA BẮT BUỘC ---
Hãy trả về kết quả theo đúng cấu trúc sau (giữ nguyên tiêu đề các mục để hệ thống phân tích tự động):

Title: [Tên tiêu đề review phim thật cuốn hút, giật gân]
Hook: [Câu mở đầu 1-2 câu cực kỳ tò mò giữ chân khán giả]
Script:
[Toàn bộ nội dung kịch bản review chi tiết từ mở đầu, diễn biến, cao trào đến kết luận]
"""
        prompt = prompt.strip()
        tokens = cls.estimate_tokens(prompt)
        _logger.info("Estimated prompt size: {} tokens ({} characters)", tokens, len(prompt))

        if tokens > max_prompt_tokens:
            _logger.warning("Prompt size ({} tokens) exceeds budget ({} tokens). Trimming transcript...", tokens, max_prompt_tokens)
            # Trim transcript preview
            trim_len = max(500, len(transcript_preview) - (tokens - max_prompt_tokens) * 4)
            transcript_preview = transcript_preview[:trim_len] + "... [Đã rút gọn transcript do vượt hạn mức token]"
            # Rebuild prompt with trimmed transcript
            prompt = f"""Bạn là một chuyên gia review phim hàng đầu trên YouTube và TikTok.
Hãy viết một kịch bản review phim hoàn chỉnh, cuốn hút, hấp dẫn người nghe từ đầu đến cuối dựa trên dữ liệu video dưới đây:

--- THÔNG TIN VIDEO & TIMELINE ---
- Tổng thời lượng: {total_duration:.1f} giây ({num_events} phân cảnh)
- Nhân vật xuất hiện: {chars_text}
- Diễn biến các cảnh quay chính:
{scenes_text}

- Lời thoại / Transcript trích xuất:
{transcript_preview}

--- YÊU CẦU KỊCH BẢN ---
- Phong cách viết: {style.upper()} (lôi cuốn, kịch tính, nhịp điệu mượt mà).
- Ngôn ngữ: {lang_name}.
- Độ dài kịch bản: {words_hint}.
- Lời văn tự nhiên để đọc lồng tiếng (TTS), không dùng các ký hiệu thừa, không chèn ghi chú thời gian như [00:01] vào phần lời đọc.
{f"- Yêu cầu bổ sung: {custom_instructions}" if custom_instructions else ""}

--- ĐỊNH DẠNG ĐẦU RA BẮT BUỘC ---
Hãy trả về kết quả theo đúng cấu trúc sau (giữ nguyên tiêu đề các mục để hệ thống phân tích tự động):

Title: [Tên tiêu đề review phim thật cuốn hút, giật gân]
Hook: [Câu mở đầu 1-2 câu cực kỳ tò mò giữ chân khán giả]
Script:
[Toàn bộ nội dung kịch bản review chi tiết từ mở đầu, diễn biến, cao trào đến kết luận]
""".strip()

        return prompt
