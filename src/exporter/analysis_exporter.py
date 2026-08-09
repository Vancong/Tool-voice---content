# -*- coding: utf-8 -*-
"""
src/exporter/analysis_exporter.py

Exports structured video analysis data (Timeline, Vision details, Whisper Dialogues,
Characters, Emotional Flow, and Story Beats) to JSON, CSV, and Markdown.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

_logger = get_logger("analysis_exporter")


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS format."""
    total_secs = int(seconds)
    hours, remainder = divmod(total_secs, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class AnalysisExporter:
    """Exports multi-modal video analysis data into structured formats."""

    @staticmethod
    def export(
        job_id: str,
        timeline: Any,
        movie_memory: Optional[Any] = None,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Export analysis data to JSON, CSV, and Markdown.

        Returns a dictionary of generated file paths.
        """
        if output_dir is None:
            out_path = Path("data") / "jobs" / job_id / "analysis"
        else:
            out_path = output_dir

        out_path.mkdir(parents=True, exist_ok=True)

        # Build structured scene records
        scenes_data: List[Dict[str, Any]] = []

        # Map scene details from MovieMemory if available
        memory_scene_map: Dict[int, Any] = {}
        if movie_memory and hasattr(movie_memory, "scene_details"):
            for s in movie_memory.scene_details:
                idx = getattr(s, "scene_index", None)
                if idx is not None:
                    memory_scene_map[idx] = s

        # Extract events from timeline
        events = getattr(timeline, "events", []) if timeline else []
        for i, ev in enumerate(events):
            s_idx = getattr(ev, "scene_index", i)
            s_start = getattr(ev, "start_time", 0.0)
            s_end = getattr(ev, "end_time", 0.0)
            s_summary = getattr(ev, "summary", "")
            s_chars = getattr(ev, "characters", [])
            s_actions = getattr(ev, "actions", [])
            s_objects = getattr(ev, "objects", [])
            s_emotion = getattr(ev, "emotion", "")

            # Merge with detailed reasoning from MovieMemory if present
            mem_detail = memory_scene_map.get(s_idx)
            why_matters = getattr(mem_detail, "why_this_scene_matters", getattr(mem_detail, "why_matters", "")) if mem_detail else ""
            scene_type = getattr(mem_detail, "scene_type", "Development") if mem_detail else "Development"
            importance = getattr(mem_detail, "importance_score", 5) if mem_detail else 5

            time_range = f"{format_seconds(s_start)} - {format_seconds(s_end)}" if s_end > 0 else f"Scene #{s_idx + 1}"

            scenes_data.append({
                "scene_index": s_idx + 1,
                "start_time_sec": round(s_start, 2),
                "end_time_sec": round(s_end, 2),
                "timestamp": time_range,
                "summary": s_summary,
                "characters": ", ".join(s_chars) if isinstance(s_chars, list) else str(s_chars),
                "actions": ", ".join(s_actions) if isinstance(s_actions, list) else str(s_actions),
                "objects": ", ".join(s_objects) if isinstance(s_objects, list) else str(s_objects),
                "dominant_emotion": s_emotion,
                "scene_type": scene_type,
                "importance_score": importance,
                "why_it_matters": why_matters,
            })

        # Top-level meta summary
        export_payload = {
            "job_id": job_id,
            "total_scenes": len(scenes_data),
            "characters_detected": [
                {
                    "name": getattr(c, "name", ""),
                    "role": getattr(c, "role", ""),
                    "personality": getattr(c, "personality", ""),
                    "motivation": getattr(c, "motivation", ""),
                }
                for c in getattr(movie_memory, "characters", [])
            ] if movie_memory and hasattr(movie_memory, "characters") else [],
            "scenes": scenes_data,
        }

        # 1. Write JSON file
        json_file = out_path / "analysis_data.json"
        json_file.write_text(json.dumps(export_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        _logger.info("Exported analysis JSON to: {}", json_file)

        # 2. Write CSV file (spreadsheet ready)
        csv_file = out_path / "analysis_data.csv"
        fieldnames = [
            "scene_index",
            "timestamp",
            "summary",
            "characters",
            "actions",
            "objects",
            "dominant_emotion",
            "scene_type",
            "importance_score",
            "why_it_matters",
        ]
        with csv_file.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in scenes_data:
                writer.writerow(row)
        _logger.info("Exported analysis CSV to: {}", csv_file)

        # 3. Write Markdown Summary
        md_file = out_path / "analysis_data.md"
        md_lines = [
            f"# 🎬 Bảng Phân Tích Nội Dung Video [Job: {job_id}]",
            f"\n- **Tổng số phân cảnh:** {len(scenes_data)}",
            "\n## 👥 Danh sách nhân vật chính:",
        ]
        if export_payload["characters_detected"]:
            for ch in export_payload["characters_detected"]:
                md_lines.append(f"- **{ch['name']}** ({ch['role']}): {ch['motivation']} (Tính cách: {ch['personality']})")
        else:
            md_lines.append("- *Chưa phân tích danh tính nhân vật cụ thể*")

        md_lines.append("\n## ⏱️ Chi tiết từng phân cảnh (Timeline Breakdown):")
        md_lines.append("\n| STT | Mốc Thời Gian | Tóm Tắt Diễn Biến Cảnh | Cảm Xúc | Nhân Vật | Tầm Quan Trọng |")
        md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for s in scenes_data:
            s_sum = s['summary'].replace("|", "/")
            md_lines.append(
                f"| {s['scene_index']} | {s['timestamp']} | {s_sum} | {s['dominant_emotion']} | {s['characters']} | {s['importance_score']}/10 |"
            )

        md_file.write_text("\n".join(md_lines), encoding="utf-8")
        _logger.info("Exported analysis Markdown to: {}", md_file)

        return {
            "json": json_file,
            "csv": csv_file,
            "md": md_file,
        }
