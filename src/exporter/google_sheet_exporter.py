# -*- coding: utf-8 -*-
"""
src/exporter/google_sheet_exporter.py

Service for syncing and exporting video analysis, scene timelines, and review scripts
to Google Sheets via Google Apps Script Webhook, gspread API, or CSV export.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.utils.logger import get_logger

_logger = get_logger("google_sheet_exporter")

GOOGLE_APPS_SCRIPT_TEMPLATE = """
// ============================================================================
// GOOGLE APPS SCRIPT WEBHOOK TEMPLATE CHO AI MOVIE REVIEW STUDIO (BATCH CLIPS)
// Hướng dẫn:
// 1. Mở Google Sheet -> Tiện ích mở rộng (Extensions) -> Apps Script
// 2. Dán đoạn mã này vào và bấm Save (Lưu)
// 3. Bấm Deploy (Triển khai) -> New deployment (Triển khai mới)
// 4. Chọn Loại: Web app -> Quyền truy cập: "Anyone" (Bất kỳ ai)
// 5. Sao chép Web App URL và dán vào ô Google Sheet Webhook trong tool.
// ============================================================================

function doPost(e) {
  try {
    var data = JSON.parse(e.postData.contents);
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    
    // Nếu sheet đang trống, tạo Header đúng chuẩn 3 cột
    if (sheet.getLastRow() === 0) {
      sheet.appendRow(["stt video", "nội dung mới viết", "voice"]);
      var headerRange = sheet.getRange(1, 1, 1, 3);
      headerRange.setBackground("#1e293b");
      headerRange.setFontColor("#ffffff");
      headerRange.setFontWeight("bold");
    }
    
    var rows = data.rows || [];
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      sheet.appendRow([
        r.stt_video || (i + 1),
        r.noi_dung_moi_viet || "",
        r.voice || ""
      ]);
    }
    
    return ContentService.createTextOutput(JSON.stringify({status: "success", count: rows.length}))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({status: "error", message: err.toString()}))
      .setMimeType(ContentService.MimeType.JSON);
  }
}
"""


class GoogleSheetExporter:
    """Handles exporting and syncing review scripts and timeline analysis to Google Sheets."""

    @staticmethod
    def sync_to_sheet(
        job_id: str,
        timeline: Any,
        review_text: str,
        webhook_url: Optional[str] = None,
        sheet_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
        video_info: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Sync review data and scenes to Google Sheet and local CSV matching table format:
        [stt video, nội dung mới viết, voice]
        """
        import re

        if output_dir is None:
            out_path = Path("data") / "jobs" / job_id / "sheets"
        else:
            out_path = output_dir
        out_path.mkdir(parents=True, exist_ok=True)

        events = getattr(timeline, "events", []) if timeline else []
        paragraphs = [p.strip() for p in review_text.split("\n\n") if p.strip()]

        clips_list = getattr(video_info, "clips", []) if video_info else []

        sheet_rows: List[Dict[str, Any]] = []
        total_items = max(len(paragraphs), len(events), len(clips_list))

        for i in range(total_items):
            clip_path = clips_list[i] if i < len(clips_list) else None
            stt = str(i + 1)
            
            if clip_path:
                digits = re.findall(r'\d+', clip_path.stem)
                if digits:
                    stt = digits[-1]
            
            para_text = paragraphs[i] if i < len(paragraphs) else ""
            if stt.isdigit():
                voice_name = f"AT-{int(stt):02d}.mp3"
            else:
                voice_name = f"AT-{stt}.mp3"

            sheet_rows.append({
                "stt_video": stt,
                "noi_dung_moi_viet": para_text,
                "voice": voice_name,
            })

        payload = {
            "job_id": job_id,
            "total_rows": len(sheet_rows),
            "rows": sheet_rows,
        }

        # 1. Local CSV Export matching user's exact columns
        csv_file = out_path / f"batch_content_{job_id}.csv"
        with csv_file.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["stt video", "nội dung mới viết", "voice"],
                extrasaction="ignore",
            )
            writer.writeheader()
            for r in sheet_rows:
                writer.writerow({
                    "stt video": r["stt_video"],
                    "nội dung mới viết": r["noi_dung_moi_viet"],
                    "voice": r["voice"],
                })
        _logger.info("Saved local batch clip CSV export to: {}", csv_file)

        # Also write a copy to the main output directory if accessible
        try:
            main_csv = Path("data") / f"batch_content_{job_id}.csv"
            main_csv.write_text(csv_file.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
        except Exception:
            pass

        # 2. Sync to Google Apps Script Webhook if provided
        webhook_status = "not_configured"
        webhook_target = webhook_url or os.getenv("GOOGLE_SHEET_WEBHOOK_URL", "")
        if webhook_target and webhook_target.startswith("http"):
            if "/home/projects/" in webhook_target or "/edit" in webhook_target or not ("/macros/s/" in webhook_target or webhook_target.endswith("/exec")):
                _logger.warning("⚠️ LỖI WEBHOOK URL: Bạn đang nhập link chỉnh sửa Apps Script ({})! Vui lòng bấm nút 'Triển khai' (Deploy) ➔ Chọn 'Ứng dụng web' và lấy link kết thúc bằng '/exec'.", webhook_target)
                webhook_status = "invalid_url_pasted_editor_link"
            else:
                try:
                    _logger.info("Pushing batch clip data to Google Sheet Webhook: {}", webhook_target)
                    res = requests.post(webhook_target, json=payload, timeout=30)
                    if res.status_code == 200:
                        webhook_status = "success"
                        _logger.info("Successfully synced {} rows to Google Sheet via Webhook!", len(sheet_rows))
                    else:
                        webhook_status = f"error_{res.status_code}"
                        _logger.warning("Google Sheet Webhook returned HTTP {}: {}", res.status_code, res.text)
                except Exception as exc:
                    webhook_status = f"failed: {exc}"
                    _logger.warning("Failed to sync to Google Sheet Webhook: {}", exc)

        return {
            "csv_file": csv_file,
            "rows_count": len(sheet_rows),
            "webhook_status": webhook_status,
        }

