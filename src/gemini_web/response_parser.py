"""Response parser for Gemini Web chat outputs.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from src.review.models import ReviewMetadata, ReviewResult


class ResponseParser:
    """Parses raw text response from Gemini Web into ReviewResult."""

    @staticmethod
    def parse(
        raw_text: str,
        processing_time: float = 0.0,
        model_name: str = "gemini-web",
    ) -> ReviewResult:
        """Parse Gemini Web output into ReviewResult object."""
        text = raw_text.strip()
        title = ""
        hook = ""
        script_lines = []
        current_section = None

        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            lower_line = line_str.lower()

            # Filter out Gemini Web UI header artifacts
            if lower_line in ("gemini said", "gemini đã nói", "show thinking", "hide thinking", "thinking...", "gemini"):
                continue
            if lower_line.startswith("gemini said:") or lower_line.startswith("gemini đã nói:"):
                line_str = line_str.split(":", 1)[1].strip()
                lower_line = line_str.lower()

            # Clean markdown bold/italics
            cleaned_lower = re.sub(r"[\*\_#]", "", lower_line).strip()

            if cleaned_lower.startswith("title:") or cleaned_lower.startswith("tiêu đề:"):
                current_section = "title"
                val = line_str.split(":", 1)[1].strip().strip("*#_ ")
                title = val
            elif cleaned_lower.startswith("hook:") or cleaned_lower.startswith("mở đầu:") or cleaned_lower.startswith("mở bài:"):
                current_section = "hook"
                val = line_str.split(":", 1)[1].strip().strip("*#_ ")
                hook = val
            elif cleaned_lower.startswith("script:") or cleaned_lower.startswith("kịch bản:") or cleaned_lower.startswith("nội dung:"):
                current_section = "script"
                val = line_str.split(":", 1)[1].strip().strip("*#_ ")
                if val:
                    script_lines.append(val)
            elif current_section == "script":
                script_lines.append(line_str)
            elif current_section == "hook" and not hook:
                hook = line_str
            elif current_section == "title" and not title:
                title = line_str
            elif not current_section:
                script_lines.append(line_str)

        script = "\n\n".join([s for s in script_lines if s.strip()]).strip()
        if not script:
            script = text

        if not title:
            # Fallback title from first sentence
            first_line = script.split("\n")[0]
            title = first_line[:60] + "..." if len(first_line) > 60 else first_line

        if not hook:
            hook = script.split("\n")[0]

        word_count = len(script.split())
        estimated_duration = (word_count / 150.0) * 60.0

        metadata = ReviewMetadata(
            total_words=word_count,
            estimated_duration=estimated_duration,
            model_name=model_name,
            processing_time=processing_time,
        )

        return ReviewResult(
            title=title,
            hook=hook,
            script=script,
            metadata=metadata,
        )
