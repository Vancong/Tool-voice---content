# -*- coding: utf-8 -*-
"""
src/ui/main_window.py

Giao diện Hiện đại (Dark Studio Pro) cho AI Movie Review Pipeline.
Tích hợp:
- Multi-Agent AI (Gemini Web, Gemini Cloud API, OpenAI ChatGPT-4o)
- Few-Shot Style Adapter (Văn phong mẫu Triệu View, Kịch tính, Tâm lý, Thuyết minh, File mẫu ngoài)
- Studio Text-to-Speech (CapCut API, EdgeTTS, ElevenLabs Multilingual v2 & Voice Cloning)
- Google Sheet Webhook Sync & Structured Analysis Data Export (JSON/CSV/Markdown)
- FFmpeg Ultra-HQ Video Composer
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk
from PIL import Image

from src.config.settings import CONFIG
from src.core.result import Result
from src.core.video_loader import VideoInfo, VideoLoader
from src.core.workflow import WorkflowEngine
from src.gemini_web.gemini_web_provider import GeminiWebProvider
from src.gemini_web.models import SessionStatus
from src.stt.providers.faster_whisper_provider import FasterWhisperProvider
from src.scene.providers.pyscenedetect_provider import PySceneDetectProvider
from src.frame.providers.opencv_frame_provider import OpenCVFrameProvider
from src.vision.providers.gemini_vision_provider import GeminiVisionProvider
from src.timeline.providers.timeline_builder import TimelineBuilderProvider
from src.review.providers.gemini_review_provider import GeminiReviewProvider
from src.review.providers.openai_review_provider import OpenAIReviewProvider
from src.tts.providers.capcut_tts_provider import CapCutTTSProvider
from src.tts.providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from src.composer.providers.ffmpeg_video_composer import FFmpegVideoComposer
from src.agents.multi_agent_provider import MultiAgentReviewProvider
from src.agents.sample_styles import SAMPLE_STYLES

# ---------------------------------------------------------------------------
# Appearance & Theme Tokens
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Theme Palette: Deep Slate & Obsidian Studio
BG_MAIN = "#0a0e17"
BG_CARD = "#111827"
BG_CARD_BORDER = "#1f293d"
BG_SUB_CARD = "#0f172a"
BG_INPUT = "#0b0f19"
BORDER_INPUT = "#334155"

TEXT_PRIMARY = "#f8fafc"
TEXT_SECONDARY = "#94a3b8"
TEXT_ACCENT = "#38bdf8"
TEXT_MUTED = "#64748b"

ACCENT_BLUE = "#2563eb"
ACCENT_BLUE_HOVER = "#1d4ed8"
ACCENT_GREEN = "#10b981"
ACCENT_GREEN_HOVER = "#059669"
ACCENT_RED = "#ef4444"
ACCENT_RED_HOVER = "#dc2626"
ACCENT_SLATE = "#334155"
ACCENT_SLATE_HOVER = "#475569"

_REVIEW_STYLES = ["Documentary", "Storytelling", "Funny", "Shorts"]
_VOICE_OPTIONS = [
    "ElevenLabs: Kat (Nữ - Sharp Educator)",
    "ElevenLabs: Parker (Nam - Professional)",
    "ElevenLabs: Adam (Nam - Free API)",
]
_LANGUAGES = ["Tiếng Việt", "Tiếng Anh"]

_SAMPLE_STYLE_MAP = {
    "🎭 Review Triệu View (Hài hước, Châm biếm, 'Anh em')": "viral_trieu_view",
    "🔥 Kịch Tính & Giật Gân (Shorts / TikTok / Gay cấn)": "suspense_kich_tinh",
    "🧠 Phân Tích Chiều Sâu & Cái Kết (Tâm lý, Triết lý)": "deep_analysis_tam_ly",
    "📖 Thuyết Minh Chuẩn Mực (Tài liệu, Truyền cảm)": "documentary_chuan_muc",
    "📁 Kịch bản mẫu tùy chỉnh từ file (.txt)...": "custom_file",
}

_PRESET_PROMPTS = {
    "-- Chọn mẫu yêu cầu nhanh --": "",
    "🎭 Hài hước & Châm biếm": "Bình luận theo phong cách hài hước, dí dỏm, châm biếm các tình huống trớ trêu, xưng hô 'anh em' thân mật và cuốn hút.",
    "🔥 Kịch tính (TikTok / Shorts)": "Review theo phong cách kịch tính dồn dập, đẩy mạnh sự hồi hộp, đặt nhiều câu hỏi lôi cuốn người xem không thể rời mắt.",
    "🧠 Phân tích tâm lý & Cái kết": "Phân tích sâu diễn biến tâm lý nhân vật, động cơ hành động và giải thích chi tiết ý nghĩa cái kết của bộ phim.",
    "📖 Thuyết minh chi tiết toàn phim": "Thuyết minh chi tiết, mạch lạc toàn bộ cốt truyện theo phong cách review phim tài liệu chuẩn mực của các kênh lớn.",
}

_log_queue: queue.Queue[str] = queue.Queue()


def _update_env_file(updates: dict[str, str]) -> None:
    """Safely update or add key-value pairs in the local .env file."""
    env_path = Path(".env")
    lines = []
    if env_path.exists():
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []

    updated_keys = set()
    new_lines = []
    
    # Sanitize updates to prevent breaking .env with multiline JSON
    sanitized_updates = {k: str(v).replace('\n', '').replace('\r', '') for k, v in updates.items()}

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in sanitized_updates and sanitized_updates[k]:
                val = sanitized_updates[k]
                if '"' in val:
                    new_lines.append(f"{k}='{val}'")
                else:
                    new_lines.append(f'{k}="{val}"')
                updated_keys.add(k)
                continue
        new_lines.append(line)

    for k, v in sanitized_updates.items():
        if k not in updated_keys and v:
            if '"' in v:
                new_lines.append(f"{k}='{v}'")
            else:
                new_lines.append(f'{k}="{v}"')

    try:
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def _enqueue_log(msg: str) -> None:
    _log_queue.put_nowait(msg)


class MainWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("AI Movie Review Studio v3.0 — Professional Studio Pipeline")
        self.geometry("1020x720")
        self.minsize(920, 640)
        self.configure(fg_color=BG_MAIN)

        # Threading & State
        self._worker_thread: Optional[threading.Thread] = None
        self._cancel_token = threading.Event()
        self._current_video_info: Optional[VideoInfo] = None
        self._last_output_file: Optional[Path] = None
        self._last_job_id: Optional[str] = None
        self._custom_sample_text: Optional[str] = None

        # Gemini Web provider & API Key / State Variables
        self._gemini_web_provider = GeminiWebProvider()
        self._gemini_api_key_var = ctk.StringVar(value=os.getenv("GEMINI_API_KEY", ""))
        self._openai_key_var = ctk.StringVar(value=os.getenv("OPENAI_API_KEY", ""))
        self._openai_model_var = ctk.StringVar(value=os.getenv("OPENAI_MODEL", "gpt-4o"))
        self._eleven_key_var = ctk.StringVar(value=os.getenv("ELEVENLABS_API_KEY", ""))
        self._eleven_voice_id_var = ctk.StringVar(value="")
        self._debug_var = ctk.BooleanVar(value=False)

        self._build_ui()
        self._poll_log_queue()
        self._refresh_gemini_status()

    # ------------------------------------------------------------------
    # UI Layout Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. Header Banner
        self._build_header()

        # 2. Main 2-Column Split
        body_frame = ctk.CTkFrame(self, fg_color="transparent")
        body_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(6, 8))
        body_frame.grid_columnconfigure(0, weight=4, minsize=320)  # Left Video Hub
        body_frame.grid_columnconfigure(1, weight=6, minsize=480)  # Right AI Studio
        body_frame.grid_rowconfigure(0, weight=1)

        self._build_left_column(body_frame)
        self._build_right_column(body_frame)

        # 3. Bottom Control Deck & Live Monitor
        self._build_bottom_section()

    def _build_header(self) -> None:
        """Top bar with Studio logo and status badge."""
        header_card = ctk.CTkFrame(
            self,
            fg_color=BG_CARD,
            border_width=1,
            border_color=BG_CARD_BORDER,
            corner_radius=12,
            height=46,
        )
        header_card.grid(row=0, column=0, sticky="ew", padx=20, pady=(8, 4))
        header_card.grid_propagate(False)
        header_card.grid_columnconfigure(1, weight=1)

        left_hdr = ctk.CTkFrame(header_card, fg_color="transparent")
        left_hdr.pack(side="left", padx=16, pady=8)

        ctk.CTkLabel(
            left_hdr,
            text="🎬 AI MOVIE REVIEW STUDIO",
            font=ctk.CTkFont(family="Arial", size=17, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        # Pro Badge
        badge = ctk.CTkFrame(left_hdr, fg_color="#1e293b", corner_radius=6)
        badge.pack(side="left", padx=(10, 0))
        ctk.CTkLabel(
            badge,
            text="⚡ v3.0 PRO",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_ACCENT,
        ).pack(padx=6, pady=2)

        # Subtitle
        ctk.CTkLabel(
            header_card,
            text="Tự động biên kịch ChatGPT/Gemini • Lồng tiếng ElevenLabs • Đồng bộ Google Sheets",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(side="left", padx=20)

        # Right quick backend indicator
        self._lbl_hdr_status = ctk.CTkLabel(
            header_card,
            text="● Sẵn sàng",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT_GREEN,
        )
        self._lbl_hdr_status.pack(side="right", padx=16)

    def _build_left_column(self, parent: ctk.CTkFrame) -> None:
        """Left Column: Video Source, Thumbnail & Technical Metadata."""
        left_card = ctk.CTkFrame(
            parent,
            fg_color=BG_CARD,
            border_width=1,
            border_color=BG_CARD_BORDER,
            corner_radius=14,
        )
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        left_card.grid_columnconfigure(0, weight=1)

        # Title
        hdr_box = ctk.CTkFrame(left_card, fg_color="transparent")
        hdr_box.pack(fill="x", padx=16, pady=(8, 4))
        ctk.CTkLabel(
            hdr_box,
            text="📹 Video Nguồn",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        # Thumbnail Showcase Box
        self._thumb_frame = ctk.CTkFrame(
            left_card,
            width=310,
            height=155,
            fg_color=BG_INPUT,
            border_width=1,
            border_color=BORDER_INPUT,
            corner_radius=10,
        )
        self._thumb_frame.pack(padx=16, pady=(0, 6))
        self._thumb_frame.pack_propagate(False)

        self._thumb_label = ctk.CTkLabel(
            self._thumb_frame,
            text="🎞️ Chưa tải video\nNhấn 'Chọn Video Nguồn' bên dưới",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=13),
            justify="center",
        )
        self._thumb_label.pack(expand=True)

        # Browse Video Buttons (Clips Folder or Single File)
        btn_box = ctk.CTkFrame(left_card, fg_color="transparent")
        btn_box.pack(fill="x", padx=16, pady=(0, 6))
        btn_box.grid_columnconfigure((0, 1), weight=1)

        self._btn_browse_folder = ctk.CTkButton(
            btn_box,
            text="🎬 Chọn Thư Mục Clips (10-30s)",
            font=ctk.CTkFont(size=11, weight="bold"),
            height=30,
            fg_color=ACCENT_GREEN,
            hover_color=ACCENT_GREEN_HOVER,
            corner_radius=8,
            command=self.browse_clip_folder,
        )
        self._btn_browse_folder.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self._btn_browse_vid = ctk.CTkButton(
            btn_box,
            text="📹 Chọn 1 File Video",
            font=ctk.CTkFont(size=11, weight="bold"),
            height=30,
            fg_color=ACCENT_BLUE,
            hover_color=ACCENT_BLUE_HOVER,
            corner_radius=8,
            command=self.browse_video,
        )
        self._btn_browse_vid.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # Metadata Card Grid
        meta_container = ctk.CTkFrame(
            left_card,
            fg_color=BG_SUB_CARD,
            border_width=1,
            border_color=BG_CARD_BORDER,
            corner_radius=10,
        )
        meta_container.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        meta_container.grid_columnconfigure(1, weight=1)

        meta_rows = [
            ("📁 Tên file:", "_lbl_filename", "Chưa có"),
            ("⏱ Thời lượng:", "_lbl_duration", "—"),
            ("📐 Độ phân giải:", "_lbl_resolution", "—"),
            ("🎞 Khung hình (FPS):", "_lbl_fps", "—"),
            ("💾 Dung lượng:", "_lbl_filesize", "—"),
        ]

        for idx, (label_text, attr_name, default_val) in enumerate(meta_rows):
            ctk.CTkLabel(
                meta_container,
                text=label_text,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=TEXT_SECONDARY,
            ).grid(row=idx, column=0, sticky="w", padx=12, pady=3)

            val_label = ctk.CTkLabel(
                meta_container,
                text=default_val,
                font=ctk.CTkFont(size=12, weight="bold" if idx == 1 else "normal"),
                text_color=TEXT_ACCENT if idx == 1 else TEXT_PRIMARY,
                anchor="w",
                wraplength=170,
            )
            val_label.grid(row=idx, column=1, sticky="w", padx=(6, 12), pady=3)
            setattr(self, attr_name, val_label)

    def _build_right_column(self, parent: ctk.CTkFrame) -> None:
        """Right Column: AI Engine, Few-Shot Style, TTS Voices, Google Sheet & Custom Prompt."""
        right_card = ctk.CTkFrame(
            parent,
            fg_color=BG_CARD,
            border_width=1,
            border_color=BG_CARD_BORDER,
            corner_radius=14,
        )
        right_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)
        right_card.grid_columnconfigure(0, weight=1)

        # -------------------------------------------------------------
        # Section 1: AI Engine & Writer Backend
        # -------------------------------------------------------------
        ai_card = ctk.CTkFrame(
            right_card,
            fg_color=BG_SUB_CARD,
            border_width=1,
            border_color=BG_CARD_BORDER,
            corner_radius=10,
        )
        ai_card.pack(fill="x", padx=14, pady=(6, 4))
        ai_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            ai_card,
            text="🤖 AI Engine Biên Kịch (Script Writer):",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_ACCENT,
        ).pack(anchor="w", padx=12, pady=(4, 2))

        # Gemini Web – engine duy nhất
        self._ai_engine_var = ctk.StringVar(value="gemini_web")

        engine_badge = ctk.CTkFrame(ai_card, fg_color="#1a2744", corner_radius=8)
        engine_badge.pack(fill="x", padx=12, pady=(2, 4))
        ctk.CTkLabel(
            engine_badge,
            text="🌐  Gemini Web  —  Playwright Automation (Miễn phí)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_ACCENT,
        ).pack(anchor="w", padx=12, pady=6)

        # Gemini Web Session Bar
        self._web_session_card = ctk.CTkFrame(ai_card, fg_color="transparent")
        self._web_session_card.pack(fill="x", padx=12, pady=(2, 4))

        status_row = ctk.CTkFrame(self._web_session_card, fg_color="transparent")
        status_row.pack(fill="x", pady=(0, 4))

        ctk.CTkLabel(
            status_row,
            text="Trạng thái phiên:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_SECONDARY,
        ).pack(side="left")

        self._lbl_session_status = ctk.CTkLabel(
            status_row,
            text="Đang kiểm tra...",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#f59e0b",
        )
        self._lbl_session_status.pack(side="left", padx=8)

        # 4 Actions (1 row)
        btn_action_row = ctk.CTkFrame(self._web_session_card, fg_color="transparent")
        btn_action_row.pack(fill="x", pady=(0, 2))

        self._btn_import_cookie = ctk.CTkButton(
            btn_action_row,
            text="🍪 Cookie Gemini",
            fg_color="#8b5cf6",
            hover_color="#7c3aed",
            font=ctk.CTkFont(size=11, weight="bold"),
            height=26,
            corner_radius=6,
            command=self._on_import_cookie,
        )
        self._btn_import_cookie.pack(side="left", padx=(0, 4), expand=True, fill="x")

        self._btn_import_chatgpt_cookie = ctk.CTkButton(
            btn_action_row,
            text="🤖 Cookie ChatGPT",
            fg_color="#10a37f",
            hover_color="#0e8c6d",
            font=ctk.CTkFont(size=11, weight="bold"),
            height=26,
            corner_radius=6,
            command=self._on_import_chatgpt_cookie,
        )
        self._btn_import_chatgpt_cookie.pack(side="left", padx=(0, 4), expand=True, fill="x")

        self._btn_test_gemini = ctk.CTkButton(
            btn_action_row,
            text="⚡ Test Kết Nối (Tab 1 & Tab 2)",
            fg_color=ACCENT_GREEN,
            hover_color=ACCENT_GREEN_HOVER,
            font=ctk.CTkFont(size=11, weight="bold"),
            height=26,
            corner_radius=6,
            command=self._on_test_gemini,
        )
        self._btn_test_gemini.pack(side="left", expand=True, fill="x")

        # -------------------------------------------------------------
        # Section 2: Custom Prompt Directives (Optional)
        # -------------------------------------------------------------
        style_card = ctk.CTkFrame(
            right_card,
            fg_color=BG_SUB_CARD,
            border_width=1,
            border_color=BG_CARD_BORDER,
            corner_radius=10,
        )
        style_card.pack(fill="x", padx=14, pady=(0, 4))
        style_card.grid_columnconfigure(0, weight=1)

        hdr_style = ctk.CTkFrame(style_card, fg_color="transparent")
        hdr_style.pack(fill="x", padx=12, pady=(4, 2))

        ctk.CTkLabel(
            hdr_style,
            text="📝 Yêu Cầu Prompt Bổ Sung (Tùy Chọn):",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_ACCENT,
        ).pack(side="left")

        # Custom Prompt Directives Box
        self._custom_prompt_box = ctk.CTkTextbox(
            style_card,
            height=38,
            font=ctk.CTkFont(size=11),
            fg_color=BG_INPUT,
            border_width=1,
            border_color=BORDER_INPUT,
            corner_radius=6,
        )
        self._custom_prompt_box.pack(fill="x", padx=12, pady=(2, 2))

        ctk.CTkLabel(
            style_card,
            text="💡 Để trống nếu dùng Prompt mặc định chuẩn. Nhập vào nếu muốn yêu cầu phong cách riêng.",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_MUTED,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 4))

        # -------------------------------------------------------------
        # Section 3: TTS Voices
        # -------------------------------------------------------------
        tts_card = ctk.CTkFrame(
            right_card,
            fg_color=BG_SUB_CARD,
            border_width=1,
            border_color=BG_CARD_BORDER,
            corner_radius=10,
        )
        tts_card.pack(fill="x", padx=14, pady=(0, 4))
        tts_card.grid_columnconfigure(1, weight=1)

        hdr_tts = ctk.CTkFrame(tts_card, fg_color="transparent")
        hdr_tts.pack(fill="x", padx=12, pady=(4, 4))

        ctk.CTkLabel(
            hdr_tts,
            text="🎙️ Giọng Đọc (TTS) & Studio Voice:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_ACCENT,
        ).pack(side="left")

        self._btn_batch_tts = ctk.CTkButton(
            hdr_tts,
            text="⚡ Batch Voice Sheet",
            fg_color="#059669",
            hover_color="#047857",
            font=ctk.CTkFont(size=11, weight="bold"),
            height=24,
            width=135,
            corner_radius=6,
            command=self._on_batch_sheet_tts,
        )
        self._btn_batch_tts.pack(side="right", padx=(6, 0))

        self._btn_eleven_key = ctk.CTkButton(
            hdr_tts,
            text="🔑 Nhập API Key ElevenLabs",
            fg_color="#8b5cf6",
            hover_color="#7c3aed",
            font=ctk.CTkFont(size=11, weight="bold"),
            height=24,
            width=175,
            corner_radius=6,
            command=self._on_input_elevenlabs_key,
        )
        self._btn_eleven_key.pack(side="right", padx=(6, 0))

        self._voice_var = ctk.StringVar(value=_VOICE_OPTIONS[0])
        ctk.CTkOptionMenu(
            hdr_tts,
            variable=self._voice_var,
            values=_VOICE_OPTIONS,
            width=240,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color="#1e293b",
            button_color=ACCENT_BLUE,
            button_hover_color=ACCENT_BLUE_HOVER,
            command=self._on_voice_changed,
        ).pack(side="right")

        # -------------------------------------------------------------
        # Section 4: Google Sheets Sync & Structured Analysis Export
        # -------------------------------------------------------------
        sheet_card = ctk.CTkFrame(
            right_card,
            fg_color=BG_SUB_CARD,
            border_width=1,
            border_color=BG_CARD_BORDER,
            corner_radius=10,
        )
        sheet_card.pack(fill="x", padx=14, pady=(0, 4))
        sheet_card.grid_columnconfigure(1, weight=1)

        hdr_sheet = ctk.CTkFrame(sheet_card, fg_color="transparent")
        hdr_sheet.pack(fill="x", padx=12, pady=(4, 2))

        self._sync_sheet_var = ctk.BooleanVar(value=True)
        self._skip_media_var = ctk.BooleanVar(value=True)

        ctk.CTkLabel(
            hdr_sheet,
            text="📊 Tự động đẩy kịch bản lên Google Sheet",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_ACCENT,
        ).pack(side="left")



        ctk.CTkButton(
            hdr_sheet,
            text="ℹ Apps Script Setup",
            font=ctk.CTkFont(size=10),
            width=120,
            height=22,
            fg_color=ACCENT_SLATE,
            hover_color=ACCENT_SLATE_HOVER,
            command=self._show_apps_script_guide,
        ).pack(side="right")

        sheet_url_row = ctk.CTkFrame(sheet_card, fg_color="transparent")
        sheet_url_row.pack(fill="x", padx=12, pady=(0, 4))
        sheet_url_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            sheet_url_row,
            text="Webhook URL:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_SECONDARY,
        ).grid(row=0, column=0, sticky="w", padx=(0, 6))

        self._sheet_webhook_var = ctk.StringVar(value=os.getenv("GOOGLE_SHEET_WEBHOOK_URL", ""))
        self._sheet_webhook_entry = ctk.CTkEntry(
            sheet_url_row,
            textvariable=self._sheet_webhook_var,
            placeholder_text="https://script.google.com/macros/s/.../exec",
            fg_color=BG_INPUT,
            border_color=BORDER_INPUT,
            height=24,
        )
        self._sheet_webhook_entry.grid(row=0, column=1, sticky="ew")

        # -------------------------------------------------------------
        # Section 5: Output Folder & Debug Checkbox
        # -------------------------------------------------------------
        out_card = ctk.CTkFrame(
            right_card,
            fg_color=BG_SUB_CARD,
            border_width=1,
            border_color=BG_CARD_BORDER,
            corner_radius=10,
        )
        out_card.pack(fill="x", padx=14, pady=(0, 4))
        out_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            out_card,
            text="📁 Thư mục xuất voice:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_SECONDARY,
        ).grid(row=0, column=0, sticky="w", padx=(12, 6), pady=4)

        self._output_var = ctk.StringVar(value=str(Path("data/output_voices").resolve()))
        out_entry = ctk.CTkEntry(
            out_card,
            textvariable=self._output_var,
            fg_color=BG_INPUT,
            border_color=BORDER_INPUT,
            height=24,
        )
        out_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=4)

        ctk.CTkButton(
            out_card,
            text="Chọn",
            width=60,
            height=24,
            fg_color=ACCENT_SLATE,
            hover_color=ACCENT_SLATE_HOVER,
            command=self._browse_output,
        ).grid(row=0, column=2, padx=(0, 12), pady=4)

        # Initial visibility setup
        self._on_ai_engine_changed()
        self._on_voice_changed(self._voice_var.get())

    def _build_bottom_section(self) -> None:
        """Bottom Section: Live Progress Bar, Terminal Logs & Action Buttons."""
        bottom_card = ctk.CTkFrame(
            self,
            fg_color=BG_CARD,
            border_width=1,
            border_color=BG_CARD_BORDER,
            corner_radius=14,
        )
        bottom_card.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 8))
        bottom_card.grid_columnconfigure(0, weight=1)

        # Progress Info Row
        prog_info = ctk.CTkFrame(bottom_card, fg_color="transparent")
        prog_info.pack(fill="x", padx=16, pady=(6, 2))

        self._lbl_stage = ctk.CTkLabel(
            prog_info,
            text="Trạng thái: Sẵn sàng thực hiện",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        self._lbl_stage.pack(side="left")

        self._lbl_pct = ctk.CTkLabel(
            prog_info,
            text="0%",
            font=ctk.CTkFont(family="Arial", size=14, weight="bold"),
            text_color=TEXT_ACCENT,
            anchor="e",
        )
        self._lbl_pct.pack(side="right")

        # Sleek Progress Bar
        self._progress_bar = ctk.CTkProgressBar(
            bottom_card,
            height=10,
            progress_color=TEXT_ACCENT,
            fg_color=BG_SUB_CARD,
            corner_radius=5,
        )
        self._progress_bar.set(0)
        self._progress_bar.pack(fill="x", padx=16, pady=2)

        # Real-time Terminal Log Box
        self._log_box = ctk.CTkTextbox(
            bottom_card,
            height=90,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#060911",
            border_width=1,
            border_color=BG_CARD_BORDER,
            text_color="#93c5fd",
            corner_radius=8,
        )
        self._log_box.pack(fill="x", padx=16, pady=(4, 6))
        self._log_box.configure(state="disabled")

        # Control Buttons Action Bar
        btn_bar = ctk.CTkFrame(bottom_card, fg_color="transparent")
        btn_bar.pack(fill="x", padx=16, pady=(0, 8))

        self._btn_generate = ctk.CTkButton(
            btn_bar,
            text="🚀 BẮT ĐẦU REVIEW",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=36,
            fg_color=ACCENT_GREEN,
            hover_color=ACCENT_GREEN_HOVER,
            corner_radius=8,
            command=self._on_generate,
        )
        self._btn_generate.pack(side="left", padx=(0, 10), expand=True, fill="x")

        self._btn_cancel = ctk.CTkButton(
            btn_bar,
            text="⏹ Dừng",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=36,
            width=100,
            fg_color=ACCENT_RED,
            hover_color=ACCENT_RED_HOVER,
            corner_radius=8,
            state="disabled",
            command=self._on_cancel,
        )
        self._btn_cancel.pack(side="left", padx=(0, 10))

        self._btn_open_folder = ctk.CTkButton(
            btn_bar,
            text="📂 Mở Thư Mục Voice",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=42,
            width=160,
            fg_color=ACCENT_SLATE,
            hover_color=ACCENT_SLATE_HOVER,
            corner_radius=8,
            state="normal",
            command=self._open_output_folder,
        )
        self._btn_open_folder.pack(side="left", padx=(0, 8))

        self._btn_open_analysis = ctk.CTkButton(
            btn_bar,
            text="📊 Mở Phân Tích & Sheets",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=42,
            width=175,
            fg_color=ACCENT_BLUE,
            hover_color=ACCENT_BLUE_HOVER,
            corner_radius=8,
            state="disabled",
            command=self._open_analysis_folder,
        )
        self._btn_open_analysis.pack(side="right")

    # ------------------------------------------------------------------
    # Dynamic View Handlers
    # ------------------------------------------------------------------

    def _on_ai_engine_changed(self) -> None:
        _enqueue_log("⚙️ AI Engine: Gemini Web (Playwright Automation - Miễn phí).")

    def _on_voice_changed(self, choice: str) -> None:
        _enqueue_log(f"🎙️ TTS: {choice}")

    def _on_sample_style_changed(self, choice: str) -> None:
        _enqueue_log(f"🎭 Đã chọn văn phong: {choice}")

    def _on_browse_sample_file(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Chọn file văn phong mẫu (.txt)",
            filetypes=[("Text File", "*.txt *.md"), ("All Files", "*.*")],
        )
        if path_str:
            p = Path(path_str)
            try:
                self._custom_sample_text = p.read_text(encoding="utf-8")
                self._lbl_custom_sample.configure(text=f"File: {p.name} ({len(self._custom_sample_text.split())} từ)")
                _enqueue_log(f"📄 Đã nạp file văn phong mẫu: {p.name}")
            except Exception as exc:
                messagebox.showerror("Lỗi đọc file", f"Không thể đọc file mẫu:\n{exc}")

    def _apply_preset_chip(self, preset_key: str) -> None:
        if preset_key in _PRESET_PROMPTS and _PRESET_PROMPTS[preset_key]:
            self._custom_prompt_box.delete("1.0", "end")
            self._custom_prompt_box.insert("1.0", _PRESET_PROMPTS[preset_key])
            _enqueue_log(f"📝 Áp dụng mẫu nhanh: {preset_key}")

    def _show_apps_script_guide(self) -> None:
        guide_text = """HƯỚNG DẪN 1-CLICK TẠO GOOGLE SHEET WEBHOOK (BATCH CLIPS):

1. Mở Google Sheet mới -> Chọn Tiện ích mở rộng (Extensions) -> Apps Script.
2. Dán đoạn mã sau vào:

function doPost(e) {
  try {
    var data = JSON.parse(e.postData.contents);
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    
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

3. Nhấn Deploy (Triển khai) -> New deployment -> Web app -> Chọn Who has access: Anyone -> Deploy.
4. Copy Webhook URL nhận được và dán vào ô 'Webhook URL' trên Tool!"""

        msg_box = ctk.CTkToplevel(self)
        msg_box.title("Hướng Dẫn Thiết Lập Google Sheet Webhook")
        msg_box.geometry("640x520")
        msg_box.transient(self)

        txt = ctk.CTkTextbox(msg_box, font=ctk.CTkFont(size=12))
        txt.pack(fill="both", expand=True, padx=16, pady=(16, 8))
        txt.insert("1.0", guide_text)
        txt.configure(state="disabled")

        script_code = """function doPost(e) {
  try {
    var data = JSON.parse(e.postData.contents);
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    
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
}"""

        def _copy_script():
            self.clipboard_clear()
            self.clipboard_append(script_code)
            copy_btn.configure(text="✅ Đã Sao Chép!", fg_color="#16a34a")
            self.after(2000, lambda: copy_btn.configure(text="📋 Sao Chép Mã Apps Script", fg_color="#2563eb"))

        copy_btn = ctk.CTkButton(
            msg_box,
            text="📋 Sao Chép Mã Apps Script",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            height=36,
            command=_copy_script,
        )
        copy_btn.pack(fill="x", padx=16, pady=(0, 16))

    # ------------------------------------------------------------------
    # Gemini Web Session Handlers
    # ------------------------------------------------------------------

    def _refresh_gemini_status(self) -> None:
        sm = self._gemini_web_provider.session_manager
        has_gemini = sm.has_session_file()
        has_chatgpt = sm.has_chatgpt_session()

        status_parts = []
        if has_gemini:
            status_parts.append("Gemini: Ready ✓")
        else:
            status_parts.append("Gemini: Chưa dán ❌")

        if has_chatgpt:
            status_parts.append("ChatGPT: Ready ✓")
        else:
            status_parts.append("ChatGPT: Chưa dán ❌")

        full_status = " | ".join(status_parts)
        if has_gemini and has_chatgpt:
            self._lbl_session_status.configure(text=full_status, text_color=ACCENT_GREEN)
            self._lbl_hdr_status.configure(text="● 2-Tab Ready (Gemini + ChatGPT)", text_color=ACCENT_GREEN)
        elif has_gemini or has_chatgpt:
            self._lbl_session_status.configure(text=full_status, text_color="#f59e0b")
            self._lbl_hdr_status.configure(text="● Đã dán 1/2 Session", text_color="#f59e0b")
        else:
            self._lbl_session_status.configure(text=full_status, text_color=ACCENT_RED)
            self._lbl_hdr_status.configure(text="● Chưa dán Cookie", text_color=ACCENT_RED)

    def _on_import_cookie(self) -> None:
        """Show modal dialog for pasting Gemini cookies."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("🍪 Import Cookie Gemini Web")
        dialog.geometry("620x440")
        dialog.transient(self)
        dialog.grab_set()
        dialog.focus_force()

        ctk.CTkLabel(
            dialog,
            text="🍪 Import Cookie Gemini Web",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_ACCENT,
        ).pack(anchor="w", padx=16, pady=(14, 4))

        ctk.CTkLabel(
            dialog,
            text="Dán Cookie của bạn vào ô bên dưới (Hỗ trợ dạng JSON từ Cookie-Editor/EditThisCookie hoặc dạng text SID=...; HSID=...):",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            wraplength=580,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 8))

        txt_cookie = ctk.CTkTextbox(
            dialog,
            height=240,
            font=ctk.CTkFont(size=11),
            fg_color=BG_INPUT,
            border_width=1,
            border_color=BORDER_INPUT,
            corner_radius=8,
        )
        txt_cookie.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        btn_row = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 14))

        def _do_save():
            raw_text = txt_cookie.get("1.0", "end").strip()
            if not raw_text:
                messagebox.showwarning("Cảnh báo", "Vui lòng dán chuỗi Cookie trước khi lưu!", parent=dialog)
                return

            try:
                count = self._gemini_web_provider.import_cookies(raw_text)
                if count > 0:
                    self._refresh_gemini_status()
                    self._append_log(f"✅ Đã import thành công {count} cookies & tự động nạp vào trình duyệt!")
                    messagebox.showinfo("Thành công", f"Đã lưu và nạp {count} cookies thành công!\nBây giờ bạn có thể bấm Bắt Đầu Review ngay mà không cần tắt/mở lại tool.", parent=dialog)
                    dialog.destroy()
                else:
                    messagebox.showerror("Lỗi Cookie", "Không phân tích được chuỗi Cookie! Vui lòng kiểm tra lại định dạng JSON hoặc Header string.", parent=dialog)
            except Exception as exc:
                messagebox.showerror("Lỗi", f"Lỗi khi import cookie:\n{exc}", parent=dialog)

        ctk.CTkButton(
            btn_row,
            text="✅ Lưu Cookie",
            fg_color=ACCENT_BLUE,
            hover_color=ACCENT_BLUE_HOVER,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=32,
            command=_do_save,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            btn_row,
            text="Hủy",
            fg_color=ACCENT_SLATE,
            hover_color=ACCENT_SLATE_HOVER,
            font=ctk.CTkFont(size=12),
            height=32,
            command=dialog.destroy,
        ).pack(side="right")

    def _on_import_chatgpt_cookie(self) -> None:
        """Show modal dialog for pasting ChatGPT Web cookies."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("🤖 Import Cookie ChatGPT Web (Tab 2)")
        dialog.geometry("620x440")
        dialog.transient(self)
        dialog.grab_set()
        dialog.focus_force()

        ctk.CTkLabel(
            dialog,
            text="🤖 Import Cookie ChatGPT Web (Tab 2)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#10a37f",
        ).pack(anchor="w", padx=16, pady=(14, 4))

        ctk.CTkLabel(
            dialog,
            text="Dán Cookie ChatGPT của bạn vào ô bên dưới (Hỗ trợ dạng JSON từ Cookie-Editor/EditThisCookie hoặc dạng text Header):",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            wraplength=580,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 8))

        txt_cookie = ctk.CTkTextbox(
            dialog,
            height=240,
            font=ctk.CTkFont(size=11),
            fg_color=BG_INPUT,
            border_width=1,
            border_color=BORDER_INPUT,
            corner_radius=8,
        )
        txt_cookie.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        btn_row = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 14))

        def _do_save_chatgpt():
            raw_text = txt_cookie.get("1.0", "end").strip()
            if not raw_text:
                messagebox.showwarning("Cảnh báo", "Vui lòng dán chuỗi Cookie ChatGPT trước khi lưu!", parent=dialog)
                return

            try:
                # Import into session manager targeting .chatgpt.com
                mgr = self._gemini_web_provider.session_manager
                count = mgr.import_cookies_from_raw_string(raw_text, target_domain=".chatgpt.com")
                if count > 0:
                    self._gemini_web_provider.browser_manager.reload_cookies()
                    self._refresh_gemini_status()
                    self._append_log(f"✅ Đã import thành công {count} cookies ChatGPT & tự động nạp vào Tab 2 trình duyệt!")
                    messagebox.showinfo("Thành công", f"Đã lưu và nạp {count} cookies ChatGPT thành công!\nBây giờ Tab 2 sẽ tự động chạy bằng ChatGPT Web.", parent=dialog)
                    dialog.destroy()
                else:
                    messagebox.showerror("Lỗi Cookie", "Không phân tích được chuỗi Cookie ChatGPT! Vui lòng kiểm tra lại định dạng JSON.", parent=dialog)
            except Exception as exc:
                messagebox.showerror("Lỗi", f"Lỗi khi import cookie ChatGPT:\n{exc}", parent=dialog)

        ctk.CTkButton(
            btn_row,
            text="✅ Lưu Cookie ChatGPT",
            fg_color="#10a37f",
            hover_color="#0e8c6d",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=32,
            command=_do_save_chatgpt,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            btn_row,
            text="Hủy",
            fg_color=ACCENT_SLATE,
            hover_color=ACCENT_SLATE_HOVER,
            font=ctk.CTkFont(size=12),
            height=32,
            command=dialog.destroy,
        ).pack(side="right")



    def _on_clear_session(self) -> None:
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn xóa phiên đăng nhập Gemini đã lưu?"):
            self._gemini_web_provider.clear_session()
            self._refresh_gemini_status()
            self._append_log("🗑 Đã xóa toàn bộ session và cookies Gemini.")
            messagebox.showinfo("Thông báo", "Đã xóa session thành công.")

    def _on_test_gemini(self, auto: bool = False) -> None:
        self._append_log("⚡ Đang kiểm tra kết nối Tab 1 (Gemini Web) và Tab 2 (ChatGPT Web)...")
        self._btn_test_gemini.configure(state="disabled")

        def _test_task():
            try:
                bm = self._gemini_web_provider.browser_manager
                # Test Tab 1 (Gemini)
                self._append_log("1️⃣ [Tab 1 - Gemini Web] Đang gửi tin nhắn thử nghiệm...")
                resp1 = bm.send_prompt_to_stage(1, "Xin chào Gemini! Trả lời ngắn gọn 1 câu để xác nhận kết nối thành công.")
                self._append_log(f"✅ Tab 1 (Gemini Web) phản hồi: {resp1.text}")

                # Test Tab 2 (ChatGPT)
                self._append_log("2️⃣ [Tab 2 - ChatGPT Web] Đang gửi tin nhắn thử nghiệm...")
                resp2 = bm.send_prompt_to_stage(2, "Xin chào ChatGPT! Trả lời ngắn gọn 1 câu để xác nhận kết nối thành công.")
                self._append_log(f"✅ Tab 2 (ChatGPT Web) phản hồi: {resp2.text}")

                def _done():
                    self._btn_test_gemini.configure(state="normal")
                    self._refresh_gemini_status()
                    if not auto:
                        messagebox.showinfo(
                            "Kết nối thành công ✓",
                            f"Cả 2 Tab đều kết nối thành công!\n\n"
                            f"🔹 Tab 1 (Gemini Web): {resp1.text}\n\n"
                            f"🟢 Tab 2 (ChatGPT Web): {resp2.text}"
                        )
                self.after(0, _done)
            except Exception as exc:
                err_msg = str(exc)
                def _err():
                    self._btn_test_gemini.configure(state="normal")
                    self._refresh_gemini_status()
                    self._append_log(f"❌ Test kết nối thất bại: {err_msg}")
                    if not auto:
                        messagebox.showerror("Lỗi kết nối", f"Kiểm tra kết nối thất bại:\n{err_msg}")
                self.after(0, _err)

        threading.Thread(target=_test_task, daemon=True).start()

    def _on_toggle_debug(self) -> None:
        from src.utils.logger import set_debug_mode
        enabled = self._debug_var.get()
        set_debug_mode(enabled)
        _enqueue_log(f"⚙️ Debug Mode: {'BẬT' if enabled else 'TẮT'}")

    def _on_input_elevenlabs_key(self) -> None:
        current = self._eleven_key_var.get().strip() or os.getenv("ELEVENLABS_API_KEY", "")
        dialog = ctk.CTkInputDialog(
            text="Nhập API Key ElevenLabs của bạn (bắt đầu bằng sk_...):",
            title="🔑 Nhập API Key ElevenLabs"
        )
        key = dialog.get_input()
        if key is not None and key.strip():
            key = key.strip()
            self._eleven_key_var.set(key)
            os.environ["ELEVENLABS_API_KEY"] = key
            _update_env_file({"ELEVENLABS_API_KEY": key})
            from src.tts.providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
            ElevenLabsTTSProvider.save_cookie_state(key)
            self._append_log("🔑 Đã cập nhật API Key ElevenLabs thành công!")
            messagebox.showinfo("Thành công", "Đã lưu API Key ElevenLabs thành công! ✓")

    def _on_batch_sheet_tts(self) -> None:
        """Launch Batch TTS Generator dialog to create audio files matching STT Video from Google Sheet / CSV."""
        dialog = ctk.CTkInputDialog(
            text="Dán Link Google Sheet (Link ở chế độ 'Bất kỳ ai có liên kết đều có thể xem')\nhoặc đường dẫn tới file CSV (stt video, nội dung mới viết, voice):",
            title="⚡ Tạo Voice Hàng Loạt từ Google Sheet / CSV"
        )
        source = dialog.get_input()
        if source is not None and source.strip():
            source = source.strip()
            self._append_log("⚡ Đang quét dữ liệu kịch bản từ Google Sheet / CSV...")
            
            def _worker():
                try:
                    from src.tts.batch_processor import BatchTTSProcessor
                    from src.tts.providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
                    
                    tts_provider = ElevenLabsTTSProvider(voice_id=self._voice_var.get())
                    
                    def _prog(msg, pct):
                        self._append_log(msg)
                    
                    out_dir = self._output_var.get().strip() or "data/output_voices"
                    r_res = BatchTTSProcessor.process_batch(
                        source=source,
                        output_dir=out_dir,
                        tts_provider=tts_provider,
                        progress_callback=_prog
                    )
                    
                    def _done():
                        if r_res.is_ok:
                            data = r_res.value
                            self._append_log(f"🎉 Hoàn tất tạo {data['success_count']}/{data['total_items']} voice! Đã lưu tại: {data['output_directory']}")
                            messagebox.showinfo(
                                "Thành công!",
                                f"🎉 Đã tạo thành công {data['success_count']}/{data['total_items']} file voice mp3!\n\n"
                                f"Các file âm thanh đã được đặt tên chuẩn theo STT video (08.mp3, 12.mp3, 1.mp3...) "
                                f"và lưu tại thư mục:\n{data['output_directory']}"
                            )
                        else:
                            self._append_log(f"❌ Lỗi tạo voice hàng loạt: {r_res.error}")
                            messagebox.showerror("Lỗi", f"Không thể tạo voice hàng loạt:\n{r_res.error}")
                    
                    self.after(0, _done)
                except Exception as exc:
                    err_txt = str(exc)
                    self.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi hệ thống: {err_txt}"))
            
            threading.Thread(target=_worker, daemon=True).start()

    def _on_voice_changed(self, choice: str) -> None:
        if "ElevenLabs" in choice:
            key = self._eleven_key_var.get().strip() or os.getenv("ELEVENLABS_API_KEY", "")
            if not key:
                self._append_log("⚠️ Lưu ý: Bạn đang chọn giọng ElevenLabs nhưng chưa nhập Cookie/Key. Hãy bấm nút 'Nhập Cookie ElevenLabs' để bổ sung.")

    # ------------------------------------------------------------------
    # Video Loading & Metadata Handlers
    # ------------------------------------------------------------------

    def browse_clip_folder(self) -> None:
        self._append_log("Đang mở hộp thoại chọn thư mục chứa video clips...")
        folder_str = filedialog.askdirectory(title="Chọn thư mục chứa danh sách video clips (10-30s)")
        if not folder_str:
            return

        folder_path = Path(folder_str).resolve()
        self._append_log(f"Đã chọn thư mục clips: {folder_path.name}")
        self._lbl_filename.configure(text=f"📂 {folder_path.name}")
        self._append_log("Đang quét danh sách video clips & trích xuất thumbnail...")
        self._thumb_label.configure(text="⏳ Đang tải danh sách clips...")

        def _probe_worker():
            try:
                info = VideoLoader._probe(folder_path)
                self.after(0, self._on_video_info_loaded, info)
            except Exception as exc:
                self.after(0, self._on_video_info_failed, str(exc))

        threading.Thread(target=_probe_worker, daemon=True).start()

    def browse_video(self) -> None:
        self._append_log("Đang mở hộp thoại chọn video...")
        path_str = filedialog.askopenfilename(
            title="Chọn file video đầu vào",
            filetypes=[
                ("File Video", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm"),
                ("Tất cả file", "*.*"),
            ],
        )
        if not path_str:
            return

        video_path = Path(path_str).resolve()
        self._append_log(f"Đã chọn video: {video_path.name}")
        self._lbl_filename.configure(text=video_path.name)

        try:
            size_bytes = video_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1024 else f"{size_mb:.1f} MB"
        except Exception:
            size_str = "Không xác định"
        self._lbl_filesize.configure(text=size_str)

        self._append_log("Đang trích xuất thông tin video & thumbnail...")
        self._thumb_label.configure(text="⏳ Đang tải thông tin...")

        def _probe_worker():
            try:
                info = VideoLoader._probe(video_path)
                self.after(0, self._on_video_info_loaded, info)
            except Exception as exc:
                self.after(0, self._on_video_info_failed, str(exc))

        threading.Thread(target=_probe_worker, daemon=True).start()

    def _on_video_info_loaded(self, info: VideoInfo) -> None:
        self._current_video_info = info

        mins, secs = divmod(int(info.duration_sec), 60)
        hrs, mins = divmod(mins, 60)
        dur_str = f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs else f"{mins:02d}:{secs:02d}"
        self._lbl_duration.configure(text=dur_str)

        self._lbl_resolution.configure(text=f"{info.width} x {info.height}")
        self._lbl_fps.configure(text=f"{info.fps:.2f} fps")

        if info.is_multi_clip:
            self._lbl_filename.configure(text=f"📂 {info.video_path.name}\n({len(info.clips)} clips pre-cut)")
            total_size = sum(c.stat().st_size for c in info.clips if c.exists())
            size_mb = total_size / (1024 * 1024)
            size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1024 else f"{size_mb:.1f} MB"
            self._lbl_filesize.configure(text=size_str)
            self._append_log(f"🎬 Đã nạp thành công danh sách {len(info.clips)} video clips (Tổng thời lượng: {dur_str})")
        else:
            self._lbl_filename.configure(text=info.video_path.name)
            try:
                size_bytes = info.video_path.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1024 else f"{size_mb:.1f} MB"
                self._lbl_filesize.configure(text=size_str)
            except Exception:
                pass

        try:
            if info.thumbnail_path and info.thumbnail_path.exists():
                pil_img = Image.open(info.thumbnail_path)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(330, 185))
                self._thumb_label.configure(image=ctk_img, text="")
                self._append_log("Đã tải thumbnail thành công.")
            else:
                self._thumb_label.configure(text="Không có hình xem trước")
        except Exception:
            self._thumb_label.configure(text="Lỗi hiển thị hình")

        self._append_log("Sẵn sàng xử lý.")

    def _on_video_info_failed(self, err_msg: str) -> None:
        self._thumb_label.configure(text=f"Lỗi đọc video\n({err_msg})")
        self._append_log(f"❌ Lỗi đọc video: {err_msg}")
        messagebox.showerror("Lỗi đọc video", f"Không thể đọc thông tin nguồn video:\n{err_msg}")

    def _browse_output(self) -> None:
        folder = filedialog.askdirectory(title="Chọn thư mục xuất voice")
        if folder:
            self._output_var.set(folder)

    # ------------------------------------------------------------------
    # Pipeline Execution & Controls
    # ------------------------------------------------------------------

    def _on_generate(self) -> None:
        video_path = None
        if self._current_video_info:
            video_path = getattr(self._current_video_info, "video_path", getattr(self._current_video_info, "path", None))

        if not video_path or not Path(video_path).exists():
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn file video đầu vào hợp lệ trước khi bắt đầu.")
            return

        video_path = Path(video_path)
        engine_mode = "gemini_web"
        is_web_mode = True

        # Kiểm tra session Gemini Web
        status = self._gemini_web_provider.get_session_status()
        if status == SessionStatus.NOT_LOGGED_IN:
            messagebox.showwarning(
                "Chưa có Cookie Gemini",
                "Bạn chưa cấu hình Cookie Gemini Web.\nVui lòng dán Cookie để tiếp tục.",
            )
            self._on_import_cookie()
            return

        # Check TTS ElevenLabs
        selected_voice = self._voice_var.get().strip()
        is_eleven = "ElevenLabs" in selected_voice
        if is_eleven:
            eleven_key = self._eleven_key_var.get().strip() or os.getenv("ELEVENLABS_API_KEY", "")
            if not eleven_key:
                messagebox.showerror("Thiếu Cookie / API Key", "Vui lòng nhập Cookie hoặc API Key ElevenLabs.")
                self._on_input_elevenlabs_key()
                return
            os.environ["ELEVENLABS_API_KEY"] = eleven_key

        out_folder = Path(self._output_var.get().strip() or CONFIG.composer.output_dir)
        out_folder.mkdir(parents=True, exist_ok=True)
        job_id = str(uuid.uuid4())[:8]
        self._last_job_id = job_id
        output_file = out_folder / f"{job_id}.mp4"
        self._last_output_file = output_file

        # UI state transitions
        self._cancel_token.clear()
        self._btn_generate.configure(state="disabled")
        self._btn_cancel.configure(state="normal")
        self._btn_open_folder.configure(state="disabled")
        self._btn_open_analysis.configure(state="disabled")
        self._progress_bar.set(0)
        self._lbl_pct.configure(text="0%")
        self._lbl_stage.configure(text="Trạng thái: 🚀 Đang khởi tạo pipeline...")

        sample_style_key = "default"
        sheet_webhook = self._sheet_webhook_var.get().strip() if self._sync_sheet_var.get() else None

        # Auto-persist UI settings into .env file
        _update_env_file({
            "GOOGLE_SHEET_WEBHOOK_URL": self._sheet_webhook_var.get().strip(),
            "ELEVENLABS_API_KEY": self._eleven_key_var.get().strip(),
        })

        self._append_log(f"\n==========================================")
        self._append_log(f"🚀 BẮT ĐẦU XỬ LÝ REVIEW [Mã: {job_id}]")
        self._append_log(f"   Đầu vào : {video_path}")
        self._append_log(f"   Đầu ra  : {output_file}")
        self._append_log(f"   AI Writer: {engine_mode.upper()}")
        self._append_log(f"   Voice TTS: {selected_voice}")
        if sheet_webhook:
            self._append_log(f"   Google Sheet: Đã bật đồng bộ tự động")
        self._append_log(f"==========================================")

        try:
            # Instantiate Review Generator (Gemini Web only)
            bm = self._gemini_web_provider.browser_manager
            review_gen = MultiAgentReviewProvider(
                browser_mgr=bm,
                sample_style=sample_style_key,
                custom_sample_text=self._custom_sample_text,
                quality_threshold=0.70,
            )

            # Instantiate TTS Provider
            if is_eleven:
                tts_provider = ElevenLabsTTSProvider(
                    api_key=self._eleven_key_var.get().strip() or os.getenv("ELEVENLABS_API_KEY", ""),
                    voice_id=selected_voice,
                )
            else:
                CONFIG.tts.voice_id = selected_voice
                tts_provider = CapCutTTSProvider(config=CONFIG)

            engine = WorkflowEngine(
                video_loader=VideoLoader(),
                stt=FasterWhisperProvider(config=CONFIG),
                scene_detector=PySceneDetectProvider(config=CONFIG),
                frame_extractor=OpenCVFrameProvider(config=CONFIG),
                vision_analyzer=GeminiVisionProvider(config=CONFIG),
                timeline_builder=TimelineBuilderProvider(config=CONFIG),
                review_generator=review_gen,
                tts=tts_provider,
                video_composer=FFmpegVideoComposer(config=CONFIG),
            )
        except Exception as exc:
            messagebox.showerror("Lỗi khởi tạo", f"Không thể khởi tạo các provider:\n{exc}")
            self._reset_ui_state()
            return

        custom_instructions = self._custom_prompt_box.get("1.0", "end").strip()

        self._worker_thread = threading.Thread(
            target=self._run_pipeline,
            args=(
                engine,
                video_path,
                output_file,
                job_id,
                is_web_mode,
                custom_instructions,
                sample_style_key,
                self._custom_sample_text,
                sheet_webhook,
            ),
            daemon=True,
        )
        self._worker_thread.start()

    def _on_cancel(self) -> None:
        self._cancel_token.set()
        self._append_log("⏹ Đã gửi tín hiệu dừng. Đang dừng pipeline...")
        self._btn_cancel.configure(state="disabled")

    def _run_pipeline(
        self,
        engine: WorkflowEngine,
        video_path: Path,
        output_file: Path,
        job_id: str,
        use_gemini_web: bool = True,
        custom_instructions: Optional[str] = None,
        sample_style: Optional[str] = None,
        custom_sample_text: Optional[str] = None,
        google_sheet_webhook_url: Optional[str] = None,
    ) -> None:
        def _progress_cb(stage: str, pct: float) -> None:
            self.after(0, self._update_progress_ui, stage, pct)

        try:
            result: Result = engine.run(
                video_path=video_path,
                output_path=output_file,
                job_id=job_id,
                progress_callback=_progress_cb,
                cancel_token=self._cancel_token,
                debug_mode=self._debug_var.get(),
                use_gemini_web=use_gemini_web,
                custom_instructions=custom_instructions,
                sample_style=sample_style,
                custom_sample_text=custom_sample_text,
                google_sheet_webhook_url=google_sheet_webhook_url,
                skip_media_generation=self._skip_media_var.get(),
            )
        except Exception as exc:
            self.after(0, self._handle_pipeline_complete, Result.Err(exc))
            return

        self.after(0, self._handle_pipeline_complete, result)

    def _update_progress_ui(self, stage: str, pct: float) -> None:
        self._progress_bar.set(pct)
        self._lbl_pct.configure(text=f"{int(pct * 100)}%")
        self._lbl_stage.configure(text=f"Trạng thái: {stage}...")
        _enqueue_log(f"▶ [{int(pct * 100)}%] Tiến trình: {stage}")

    def _handle_pipeline_complete(self, result: Result) -> None:
        self._reset_ui_state()

        if result.is_ok:
            self._progress_bar.set(1.0)
            self._lbl_pct.configure(text="100%")
            self._lbl_stage.configure(text="Trạng thái: Hoàn tất thành công ✓")
            self._btn_open_folder.configure(state="normal")
            self._btn_open_analysis.configure(state="normal")
            if self._skip_media_var.get():
                messagebox.showinfo(
                    "Hoàn tất thành công! 🎉",
                    "🎉 Đã viết kịch bản & đẩy tự động toàn bộ nội dung lên Google Sheet thành công!\n\n"
                    "• Cột A (stt video): STT của clip (8, 12, 1...)\n"
                    "• Cột B (nội dung mới viết): Kịch bản review do AI viết\n"
                    "• Cột C (voice): Tên file mp3 tương ứng (08.mp3, 12.mp3, 1.mp3...)\n\n"
                    "👉 Bạn có thể mở Google Sheet kiểm tra kịch bản, sau đó bấm nút '⚡ Batch Voice Sheet' để tạo voice hàng loạt bất kỳ lúc nào!"
                )
            else:
                messagebox.showinfo("Thành công", f"Đã tạo video review thành công!\n\nĐường dẫn:\n{self._last_output_file}")
        else:
            self._lbl_stage.configure(text="Trạng thái: Xử lý thất bại ❌")
            err_msg = str(result.error)
            self._append_log(f"❌ Pipeline thất bại: {err_msg}")
            messagebox.showerror("Lỗi xử lý", f"Pipeline gặp lỗi:\n\n{err_msg}")

    def _reset_ui_state(self) -> None:
        self._btn_generate.configure(state="normal")
        self._btn_cancel.configure(state="disabled")

    def _open_output_folder(self) -> None:
        target_dir = Path(self._output_var.get().strip()).resolve()
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)

        if sys.platform == "win32":
            os.startfile(str(target_dir))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(target_dir)])
        else:
            subprocess.run(["xdg-open", str(target_dir)])

    def _open_analysis_folder(self) -> None:
        if not self._last_job_id:
            return
        analysis_dir = Path("data") / "jobs" / self._last_job_id / "analysis"
        if not analysis_dir.exists():
            analysis_dir = Path("data") / "jobs" / self._last_job_id
        if sys.platform == "win32":
            os.startfile(str(analysis_dir.resolve()))
        else:
            subprocess.run(["open", str(analysis_dir.resolve())])

    # ------------------------------------------------------------------
    # Logging Queue Handler
    # ------------------------------------------------------------------

    def _poll_log_queue(self) -> None:
        try:
            while True:
                line = _log_queue.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    def _append_log(self, text: str) -> None:
        if not hasattr(self, "_log_box"):
            _enqueue_log(text)
            return
        self._log_box.configure(state="normal")
        self._log_box.insert("end", text.rstrip() + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")


def run_ui() -> None:
    app = MainWindow()
    app.mainloop()
