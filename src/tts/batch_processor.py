# -*- coding: utf-8 -*-
"""
src/tts/batch_processor.py

Batch Text-to-Speech processor that reads content from Google Sheets, CSV files, or text lists
and generates individual audio files matching exact video sequence numbers (stt video).
"""

from __future__ import annotations

import csv
import io
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.core.result import Result
from src.tts.providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from src.tts.providers.capcut_tts_provider import CapCutTTSProvider
from src.utils.logger import get_logger

_logger = get_logger("batch_tts_processor")


class BatchTTSProcessor:
    """Processes batch TTS tasks from Google Sheets or CSV files."""

    @staticmethod
    def parse_google_sheet_url(url: str) -> Optional[str]:
        """Convert a standard Google Sheet sharing/view URL to a direct CSV export URL."""
        url = (url or "").strip()
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
        if not sheet_id_match:
            return None
        sheet_id = sheet_id_match.group(1)

        gid_match = re.search(r"[#&?]gid=(\d+)", url)
        gid = gid_match.group(1) if gid_match else "0"

        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    @classmethod
    def fetch_rows_from_source(cls, source: str) -> Result[List[Dict[str, str]], str]:
        """Fetch and parse rows from a Google Sheet URL, CSV file path, or raw CSV string."""
        source = (source or "").strip()
        if not source:
            return Result.Err("Nguồn dữ liệu rỗng. Vui lòng dán URL Google Sheet hoặc đường dẫn file CSV.")

        csv_content = ""

        # Case 1: Google Sheet URL or HTTP Link
        if source.startswith("http://") or source.startswith("https://"):
            csv_url = cls.parse_google_sheet_url(source) or source
            try:
                _logger.info("Fetching CSV from Google Sheet URL: {}", csv_url)
                resp = requests.get(csv_url, timeout=15, allow_redirects=True)
                if resp.status_code == 200:
                    csv_content = resp.content.decode("utf-8-sig", errors="ignore")
                else:
                    return Result.Err(f"Không thể tải Google Sheet (Lỗi HTTP {resp.status_code}). Đảm bảo link ở chế độ 'Bất kỳ ai có liên kết đều có thể xem'.")
            except Exception as exc:
                return Result.Err(f"Lỗi khi tải kết nối Google Sheet: {exc}")

        # Case 2: Local File Path (.csv or .txt)
        elif Path(source).exists():
            try:
                csv_content = Path(source).read_text(encoding="utf-8-sig", errors="ignore")
            except Exception as exc:
                return Result.Err(f"Không thể đọc file {source}: {exc}")

        # Case 3: Raw CSV text string passed directly
        else:
            csv_content = source

        # Parse CSV rows
        try:
            reader = csv.reader(io.StringIO(csv_content))
            header_raw = next(reader, None)
            if not header_raw:
                return Result.Err("File CSV hoặc Google Sheet không có dữ liệu.")

            # Normalize column names
            headers = [h.strip().lower() for h in header_raw]
            
            # Find column indices
            stt_idx = -1
            text_idx = -1
            voice_idx = -1

            for idx, h in enumerate(headers):
                if any(k in h for k in ["stt", "stt video", "video_stt", "index", "id"]):
                    stt_idx = idx
                elif any(k in h for k in ["nội dung", "noi dung", "content", "script", "text"]):
                    text_idx = idx
                elif any(k in h for k in ["voice", "audio", "file", "filename"]):
                    voice_idx = idx

            # Fallbacks if headers differ
            if text_idx == -1 and len(headers) >= 2:
                text_idx = 1
            if stt_idx == -1 and len(headers) >= 1:
                stt_idx = 0

            parsed_rows: List[Dict[str, str]] = []
            for row_num, row in enumerate(reader, start=2):
                if not row or not any(c.strip() for c in row):
                    continue

                stt_val = row[stt_idx].strip() if stt_idx < len(row) and stt_idx >= 0 else str(len(parsed_rows) + 1)
                text_val = row[text_idx].strip() if text_idx < len(row) and text_idx >= 0 else ""
                voice_val = row[voice_idx].strip() if voice_idx < len(row) and voice_idx >= 0 else ""

                if not text_val:
                    continue

                # Auto determine audio filename with AT- prefix (e.g. AT-08.mp3)
                if not voice_val:
                    if stt_val.isdigit():
                        voice_val = f"AT-{int(stt_val):02d}.mp3"
                    else:
                        voice_val = f"AT-{stt_val}.mp3"
                else:
                    if not voice_val.endswith(".mp3") and not voice_val.endswith(".wav"):
                        voice_val = f"{voice_val}.mp3"
                    if not voice_val.lower().startswith("at-"):
                        clean_stem = Path(voice_val).stem
                        if clean_stem.isdigit():
                            voice_val = f"AT-{int(clean_stem):02d}.mp3"
                        else:
                            voice_val = f"AT-{voice_val}"

                parsed_rows.append({
                    "stt": stt_val,
                    "text": text_val,
                    "filename": voice_val,
                    "line_number": str(row_num),
                })

            if not parsed_rows:
                return Result.Err("Không tìm thấy dòng kịch bản hợp lệ nào trong file.")

            return Result.Ok(parsed_rows)

        except Exception as exc:
            return Result.Err(f"Lỗi phân tích cú pháp CSV: {exc}")

    @classmethod
    def process_batch(
        cls,
        source: str,
        output_dir: Path | str = "data/output_voices",
        tts_provider: Any = None,
        progress_callback: Optional[Any] = None,
    ) -> Result[Dict[str, Any], str]:
        """Process batch rows into audio files saved cleanly in output_dir."""
        r_rows = cls.fetch_rows_from_source(source)
        if r_rows.is_err:
            return Result.Err(r_rows.error)

        rows = r_rows.value
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if tts_provider is None:
            tts_provider = ElevenLabsTTSProvider()

        _logger.info("Starting Batch TTS for {} items into directory: {}", len(rows), out_path)
        
        generated_files: List[Dict[str, Any]] = []
        errors: List[str] = []

        for idx, row in enumerate(rows, start=1):
            stt = row["stt"]
            text = row["text"]
            filename = row["filename"]
            target_file = out_path / filename

            pct = idx / len(rows)
            if progress_callback:
                progress_callback(f"[{idx}/{len(rows)}] Đang tạo voice STT {stt} ({filename})...", pct)

            _logger.info("Generating audio [{}/{}] STT {} -> {}", idx, len(rows), stt, target_file)

            res = tts_provider.synthesize(text, output_path=target_file)
            if res.is_ok:
                duration_val = getattr(res.value.metadata, "duration", 0.0) if hasattr(res.value, "metadata") else getattr(res.value, "duration", 0.0)
                generated_files.append({
                    "stt": stt,
                    "filename": filename,
                    "path": str(target_file),
                    "duration": duration_val,
                })
            else:
                err_msg = f"Lỗi tạo voice cho STT {stt} ({filename}): {res.error}"
                _logger.error(err_msg)
                errors.append(err_msg)

        summary = {
            "total_items": len(rows),
            "success_count": len(generated_files),
            "error_count": len(errors),
            "output_directory": str(out_path.resolve()),
            "files": generated_files,
            "errors": errors,
        }

        # Open output directory in Explorer
        try:
            if os.name == "nt":
                os.startfile(str(out_path.resolve()))
        except Exception:
            pass

        if errors and not generated_files:
            return Result.Err(f"Tạo batch thất bại hoàn toàn. Các lỗi:\n" + "\n".join(errors))

        return Result.Ok(summary)
