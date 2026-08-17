"""Clip-by-Clip Review Provider sending each pre-cut video clip to Gemini / ChatGPT / Gemini Web
to generate per-clip narration text for TTS voiceover.
"""

from __future__ import annotations

import base64
import os
import time
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Any, Callable, List

from src.core.result import Result
from src.review.base import BaseReviewGenerator
from src.review.exceptions import ReviewError
from src.review.models import ReviewResult, ReviewMetadata
from src.timeline.models import TimelineResult
from src.gemini_web.browser_manager import BrowserManager
from src.gemini_web.models import SessionStatus
from src.utils.logger import get_logger

_logger = get_logger("clip_review_provider")

# Lazy import PIL & google.generativeai if available
try:
    from PIL import Image
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class MultiAgentReviewProvider(BaseReviewGenerator):
    """Review Generator that processes pre-cut clips sequentially (one-by-one)
    via Gemini, ChatGPT, or Gemini Web to build the final TTS script.

    ============================================================
    GEMINI WEB WORKFLOW (2-Tab Pipeline):
    ============================================================
    SETUP (once per job):
      Tab 1 ← System prompt JOB 1 (Video Observer role)
      Tab 2 ← System prompt JOB 2 (Content Writer role)

    PER CLIP (repeated for each video clip):
      Tab 1 ← Send video file → wait → get description
      Tab 2 ← Send Tab1 description → wait → get final script
      → Push final script to Google Sheets
    ============================================================
    """

    def __init__(
        self,
        browser_mgr: Optional[BrowserManager] = None,
        openai_provider: Optional[Any] = None,
        sample_style: Optional[str] = None,
        custom_sample_text: Optional[str] = None,
        quality_threshold: float = 0.70,
        review_video_engine: str = "gemini_web",
        write_content_engine: str = "chatgpt_web",
        prompt_job1: Optional[str] = None,
        prompt_job2: Optional[str] = None,
        clip_titles: Optional[dict[str, str]] = None,
    ) -> None:
        self._browser_mgr = browser_mgr
        self._openai_provider = openai_provider
        self._sample_style = sample_style
        self._custom_sample_text = custom_sample_text
        self._review_video_engine = review_video_engine
        self._write_content_engine = write_content_engine
        self._prompt_job1 = prompt_job1
        self._prompt_job2 = prompt_job2
        self._clip_titles = clip_titles or {}
        self._tabs_initialized = False  # Track whether Tab 1 & Tab 2 have been setup

    # ------------------------------------------------------------------
    # Gemini Web: 2-Tab Pipeline
    # ------------------------------------------------------------------

    def _setup_gemini_web_tabs(
        self,
        total_clips: int,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        custom_instructions: Optional[str] = None,
    ) -> None:
        """
        STEP 1 (once): Send system prompts to Tab 1 and Tab 2.
        Tab 1 = Video Observer role (Công việc 1)
        Tab 2 = Content Writer role (Công việc 2)
        """
        if self._tabs_initialized:
            return

        if not self._browser_mgr:
            raise ReviewError("Browser Manager chưa được khởi tạo.")

        tab1_label = "Google AI Studio" if self._review_video_engine == "google_ai_studio" else "Gemini Web"
        if self._write_content_engine == "claude_web":
            tab2_label = "Claude Web"
        elif self._write_content_engine == "chatgpt_web":
            tab2_label = "ChatGPT Web"
        else:
            tab2_label = "Gemini Web"

        # ── BƯỚC 1: Kiểm tra Cookie & Phiên đăng nhập live ──
        if progress_cb:
            progress_cb(f"Bước 1: Kiểm tra Cookie & Phiên đăng nhập ({tab1_label} & {tab2_label})...", 0.02)
        _logger.info("⚡ [Bước 1] Verifying {} and {} live session status...", tab1_label, tab2_label)

        session_mgr = self._browser_mgr._session_mgr

        # 1. Check Gemini Session (Tab 1) if gemini_web selected
        if self._review_video_engine == "gemini_web" and not session_mgr.has_session_file():
            raise ReviewError(
                "⚠️ Chưa có Cookie Gemini Web (Tab 1)!\n"
                "Tab 1 cần Cookie Gemini để phân tích video.\n"
                "Vui lòng bấm '🍪 Cookie Gemini' trên giao diện để nạp cookie!"
            )

        # 2. Check ChatGPT Session (Tab 2) if chatgpt_web selected
        if self._write_content_engine == "chatgpt_web" and not session_mgr.has_chatgpt_session():
            raise ReviewError(
                "⚠️ Chưa có Cookie ChatGPT Web (Tab 2)!\n"
                "Tab 2 cần Cookie ChatGPT để biên soạn kịch bản content.\n"
                "Vui lòng bấm '🤖 Cookie ChatGPT' trên giao diện để nạp cookie!"
            )

        # 3. Check Claude Session (Tab 2) if claude_web selected
        if self._write_content_engine == "claude_web" and not session_mgr.has_claude_session():
            raise ReviewError(
                "⚠️ Chưa có Cookie Claude Web (Tab 2)!\n"
                "Tab 2 cần Cookie Claude để biên soạn kịch bản content.\n"
                "Vui lòng bấm '🦙 Cookie Claude' trên giao diện để nạp cookie!"
            )

        # 4. Live browser check: Mở trình duyệt để check phiên đăng nhập
        try:
            self._browser_mgr.verify_live_sessions(
                review_video_engine=self._review_video_engine,
                write_content_engine=self._write_content_engine,
            )
        except Exception as exc:
            try:
                self._browser_mgr.close()
            except Exception:
                pass
            raise ReviewError(str(exc)) from exc

        # 🎯 BƯỚC 1 THÀNH CÔNG: Đóng trình duyệt check theo đúng thiết kế của người dùng
        _logger.info("✅ [Bước 1 Hoàn Tất] Cookie hợp lệ! Đang đóng trình duyệt check để khởi tạo phiên làm việc mới...")
        try:
            self._browser_mgr.close()
        except Exception:
            pass

        # ── BƯỚC 2: Khởi tạo trình duyệt mới & Gửi các công việc tiếp theo ──
        if progress_cb:
            progress_cb(f"Bước 2: Mở trình duyệt mới & khởi tạo công việc ({tab1_label} & {tab2_label})...", 0.04)
        _logger.info("🚀 [Bước 2] Launching fresh browser session for task execution...")

        p1_file = Path("TRỢ LÝ QUAN SÁT VÀ MÔ TẢ VIDEO.txt")
        p2_file = Path("TRỢ LÝ VIẾT CONTENT NGẮN CHUYÊN NGHIỆP.txt")

        if self._prompt_job1 and self._prompt_job1.strip():
            p1_template = self._prompt_job1.strip()
        else:
            p1_template = p1_file.read_text(encoding="utf-8").strip() if p1_file.exists() else (
                "CÔNG VIỆC 1: TRỢ LÝ QUAN SÁT VÀ MÔ TẢ VIDEO\n"
                "Vai trò: Quan sát video và cung cấp thông tin hình ảnh chính xác cho người viết content.\n"
                "Hãy mô tả chi tiết: Mô tả diễn biến, Nhân vật xuất hiện, Hành động chính, Khoảnh khắc đắt giá nhất, "
                "Biểu cảm và ngôn ngữ cơ thể, Mối tương tác trong cảnh, Bối cảnh, Yếu tố an toàn, "
                "Thông tin chưa chắc chắn, Thời lượng cảnh."
            )

        if self._prompt_job2 and self._prompt_job2.strip():
            p2_template = self._prompt_job2.strip()
        else:
            p2_template = p2_file.read_text(encoding="utf-8").strip() if p2_file.exists() else (
                "CÔNG VIỆC 2: TRỢ LÝ VIẾT CONTENT NGẮN CHUYÊN NGHIỆP\n"
                "Vai trò: Chuyên gia viết lời bình tiếng Anh ngắn (10-25 từ) cho video thú cưng & động vật.\n"
                "Cảm xúc: Warm humor hoặc Adorable curiosity.\n"
                "Chỉ xuất ra 1 câu lời bình hoàn chỉnh, không giải thích, không tiêu đề."
            )

        if custom_instructions and custom_instructions.strip():
            p2_template = p2_template + f"\n\nChủ đề cần viết nội dung : {custom_instructions.strip()}"

        # ── Tab 1: Setup Observer Role ──
        if self._review_video_engine == "gemini_web":
            if progress_cb:
                progress_cb(f"Khởi động Tab 1 ({tab1_label}: Quan sát video)...", 0.05)
            _logger.info("Setting up Tab 1 ({}) with JOB 1 system prompt...", tab1_label)
            try:
                self._browser_mgr.send_prompt_to_stage(
                    stage_idx=1,
                    prompt=p1_template,
                    media_path=None,
                    job_id="setup_tab1",
                    progress_callback=progress_cb,
                    review_video_engine=self._review_video_engine,
                    write_content_engine=self._write_content_engine,
                )
                _logger.info("Tab 1 ({}) setup complete.", tab1_label)
            except Exception as exc:
                _logger.error("Tab 1 ({}) setup failed: {}", tab1_label, exc)
                raise ReviewError(f"Lỗi khởi động Tab 1 ({tab1_label}): {exc}") from exc

        # ── Tab 2: Setup Writer Role ──
        if self._write_content_engine in ("chatgpt_web", "claude_web"):
            if progress_cb:
                progress_cb(f"Khởi động Tab 2 ({tab2_label}: Viết content)...", 0.08)
            _logger.info("Setting up Tab 2 ({}) with JOB 2 system prompt...", tab2_label)
            try:
                self._browser_mgr.send_prompt_to_stage(
                    stage_idx=2,
                    prompt=p2_template,
                    media_path=None,
                    job_id="setup_tab2",
                    progress_callback=progress_cb,
                    review_video_engine=self._review_video_engine,
                    write_content_engine=self._write_content_engine,
                )
                _logger.info("Tab 2 ({}) setup complete.", tab2_label)
            except Exception as exc:
                _logger.error("Tab 2 ({}) setup failed: {}", tab2_label, exc)
                raise ReviewError(f"Lỗi khởi động Tab 2 ({tab2_label}): {exc}") from exc

        self._tabs_initialized = True

    def _is_valid_stage1_description(self, text: str) -> bool:
        """Kiểm tra kết quả CV1: Chỉ cần trong đầu ra có chứa cụm từ 'Review thành công'."""
        if not text or not text.strip():
            return False
        return "review thành công" in text.lower()

    def _is_valid_stage2_script(self, text: str) -> bool:
        """Kiểm tra kết quả CV2: Chỉ cần trong đầu ra có chứa cụm từ 'Viết content thành công'."""
        if not text or not text.strip():
            return False
        return "viết content thành công" in text.lower()

    @staticmethod
    def _strip_stage2_tag(text: str) -> str:
        """Xóa tag 'Viết content thành công' khỏi nội dung trước khi lưu/push."""
        if not text:
            return ""
        import re
        cleaned = re.sub(r'(?i)[\s\.,]*viết\s+content\s+thành\s+công[\s\.,]*', '', text).strip()
        return cleaned.strip('"').strip("'").strip()


    def _get_stage1_desc_gemini_web(
        self,
        clip_idx: int,
        total_clips: int,
        job_id: str = "web",
        clip_video_path: Optional[Path] = None,
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> str:
        if not self._browser_mgr:
            return f"Phân cảnh {clip_idx + 1} diễn ra vô cùng cuốn hút."

        clip_num = clip_idx + 1
        base_pct = clip_idx / total_clips
        tab1_label = "Google AI Studio" if self._review_video_engine == "google_ai_studio" else "Gemini Web"

        if progress_cb:
            progress_cb(f"Clip {clip_num}/{total_clips} (Tab 1: Gửi video qua {tab1_label})", base_pct + 0.01)

        clip_title_str = ""
        if clip_video_path and clip_video_path.name in self._clip_titles:
            clip_title_str = self._clip_titles[clip_video_path.name].strip()
        elif f"clip_{clip_num}" in self._clip_titles:
            clip_title_str = self._clip_titles[f"clip_{clip_num}"].strip()
        elif str(clip_num) in self._clip_titles:
            clip_title_str = self._clip_titles[str(clip_num)].strip()

        video_prompt = (
            f"[CLIP #{clip_num}/{total_clips}]\n"
            "CÔNG VIỆC 1: TRỢ LÝ QUAN SÁT VÀ MÔ TẢ VIDEO\n"
            "Hãy quan sát kĩ video được đính kèm và cung cấp đầy đủ thông tin theo đúng định dạng "
            "của Công việc 1 đã được thiết lập.\n"
            "QUAN TRỌNG: Hãy chắc chắn kết thúc bài mô tả của bạn bằng chính xác cụm từ: \"Review thành công\"."
        )
        if clip_title_str:
            video_prompt += f"\n\nTIÊU ĐỀ / CHỦ ĐỀ CỦA CLIP NÀY:\n{clip_title_str}"

        stage1_desc = ""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                _logger.info("Tab 1 ({}): Sending video clip {}/{} (Attempt {}/{}) → {}", tab1_label, clip_num, total_clips, attempt, max_attempts, clip_video_path)
                resp1 = self._browser_mgr.send_prompt_to_stage(
                    stage_idx=1,
                    prompt=video_prompt,
                    media_path=clip_video_path,
                    job_id=f"{job_id}_clip{clip_idx}_tab1_try{attempt}",
                    progress_callback=progress_cb,
                    review_video_engine=self._review_video_engine,
                    write_content_engine=self._write_content_engine,
                )
                raw_desc = resp1.text.strip()
                for prefix in ["gemini đã nói", "gemini said", "gemini:"]:
                    if raw_desc.lower().startswith(prefix):
                        raw_desc = raw_desc[len(prefix):].strip()
                raw_desc = raw_desc.strip('"').strip("'").strip()

                if self._is_valid_stage1_description(raw_desc):
                    stage1_desc = raw_desc
                    _logger.info("Tab 1 valid video description obtained for clip {} ({} chars) on attempt {}!", clip_num, len(stage1_desc), attempt)
                    break
                else:
                    _logger.warning("Tab 1 attempt {}/{} returned invalid response: '{}...'. Retrying...", attempt, max_attempts, raw_desc[:80])
                    time.sleep(3.0)
            except Exception as exc:
                _logger.warning("Tab 1 error on clip {} (Attempt {}/{}): {}", clip_num, attempt, max_attempts, exc)
                time.sleep(2.0)

        if not stage1_desc or not self._is_valid_stage1_description(stage1_desc):
            _logger.error("❌ Tab 1 failed to return valid video description for Clip {}!", clip_num)
            raise ReviewError(f"Clip {clip_num}: Gemini (Tab 1) không thể phân tích video sau {max_attempts} lần thử.")

        return stage1_desc

    def _generate_for_clip_web_stage2(
        self,
        stage1_desc: str,
        clip_idx: int,
        total_clips: int,
        custom_instructions: str = "",
        job_id: str = "web",
        clip_video_path: Optional[Path] = None,
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> str:
        if not self._browser_mgr:
            return f"Phân cảnh {clip_idx + 1} diễn ra vô cùng cuốn hút."

        clip_num = clip_idx + 1
        base_pct = clip_idx / total_clips

        clip_title_str = ""
        if clip_video_path and clip_video_path.name in self._clip_titles:
            clip_title_str = self._clip_titles[clip_video_path.name].strip()
        elif f"clip_{clip_num}" in self._clip_titles:
            clip_title_str = self._clip_titles[f"clip_{clip_num}"].strip()
        elif str(clip_num) in self._clip_titles:
            clip_title_str = self._clip_titles[str(clip_num)].strip()

        if progress_cb:
            progress_cb(f"Clip {clip_num}/{total_clips} (Tab 2: Viết lời bình via {self._write_content_engine})", base_pct + 0.04)

        writer_prompt = (
            f"[CLIP #{clip_num}/{total_clips}]\n\n"
            f"THÔNG TIN ĐẦU VÀO TỪ CÔNG VIỆC 1 (Clip #{clip_num}/{total_clips}):\n"
            f"{stage1_desc}"
        )
        if clip_title_str:
            writer_prompt += f"\n\nTIÊU ĐỀ / CHỦ ĐỀ CLIP:\n{clip_title_str}"
        if custom_instructions.strip():
            writer_prompt += f"\n\nYÊU CẦU BỔ SUNG:\n{custom_instructions}"
        writer_prompt += "\n\nQUAN TRỌNG: Sau khi viết xong nội dung, hãy kết thúc bằng đúng cụm từ: \"Viết content thành công\"."

        max_s2_attempts = 2
        for s2_attempt in range(1, max_s2_attempts + 1):
            try:
                _logger.info(
                    "Tab 2: Sending description for clip {} to content writer ({}) — Attempt {}/{}",
                    clip_num, self._write_content_engine, s2_attempt, max_s2_attempts,
                )
                resp2 = self._browser_mgr.send_prompt_to_stage(
                    stage_idx=2,
                    prompt=writer_prompt,
                    media_path=None,
                    job_id=f"{job_id}_clip{clip_idx}_tab2_try{s2_attempt}",
                    progress_callback=progress_cb,
                    write_content_engine=self._write_content_engine,
                )
                txt = resp2.text.strip()
                for prefix in [
                    "claude responded:", "claude responded", "claude said:", "claude:",
                    "chatgpt responded:", "chatgpt responded", "chatgpt said:", "chatgpt:",
                    "gemini đã nói:", "gemini đã nói", "gemini said:", "gemini said", "gemini:",
                ]:
                    if txt.lower().startswith(prefix):
                        txt = txt[len(prefix):].strip()
                txt = txt.strip('"').strip("'").strip()

                p_lines = [p.strip() for p in txt.splitlines() if p.strip()]
                unique_p = []
                for p in p_lines:
                    p_clean = p.strip("…").strip()
                    p_normalized = p_clean.lower()
                    if any(k in p_normalized for k in ["thông tin đầu vào", "công việc 1", "công việc 2", "tiêu đề / chủ đề"]):
                        continue

                    is_sub = False
                    for idx2, existing in enumerate(unique_p):
                        ex_clean = existing.strip("…").strip()
                        ex_norm = ex_clean.lower()
                        if p_normalized == ex_norm:
                            is_sub = True
                            break
                        elif ex_norm.startswith(p_normalized) and (len(ex_norm) - len(p_normalized) > 10 or p_clean.endswith("…")):
                            is_sub = True
                            break
                        elif p_normalized.startswith(ex_norm) and (len(p_normalized) - len(ex_norm) > 10 or ex_clean.endswith("…")):
                            unique_p[idx2] = p
                            is_sub = True
                            break
                    if not is_sub:
                        unique_p.append(p)

                txt = "\n\n".join(unique_p).strip()

                # ── Validate: dòng cuối phải là 'Viết content thành công' ──
                if self._is_valid_stage2_script(txt):
                    clean_txt = self._strip_stage2_tag(txt)
                    _logger.info("Tab 2 final script (clip {}, attempt {}): {}", clip_num, s2_attempt, clean_txt)
                    return clean_txt
                else:
                    _logger.warning(
                        "Tab 2 attempt {}/{}: Missing 'Viết content thành công' tag for clip {}. Retrying...",
                        s2_attempt, max_s2_attempts, clip_num,
                    )
                    if s2_attempt < max_s2_attempts:
                        time.sleep(3.0)

            except Exception as exc:
                _logger.error("Tab 2 error on clip {} (Attempt {}/{}): {}", clip_num, s2_attempt, max_s2_attempts, exc)
                if s2_attempt < max_s2_attempts:
                    time.sleep(2.0)

        _logger.error("❌ Tab 2 failed to return valid voiceover for Clip {} after {} attempts.", clip_num, max_s2_attempts)
        return f"Phân cảnh {clip_num} tiếp tục với những tình huống nhiều bất ngờ."

    # ------------------------------------------------------------------
    # Gemini API (Cloud): 2-stage pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _compress_video_for_gemini(video_path: Path) -> Path:
        """Tối ưu dung lượng video clip chuẩn 480p bằng FFmpeg (scale 480p, CRF 26, ultrafast) trước khi nạp API (nếu > 2.5MB)."""
        if not video_path.exists():
            return video_path

        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        if file_size_mb <= 2.5:
            return video_path

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            return video_path

        temp_dir = Path(tempfile.gettempdir()) / "cotent_voice_opt"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_compressed = temp_dir / f"{video_path.stem}_opt_{os.getpid()}_{int(time.time())}.mp4"
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", str(video_path),
                "-vf", "scale=-2:480",
                "-c:v", "libx264",
                "-crf", "26",
                "-preset", "ultrafast",
                "-an",
                str(temp_compressed)
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            if res.returncode == 0 and temp_compressed.exists() and temp_compressed.stat().st_size > 0:
                new_size_mb = temp_compressed.stat().st_size / (1024 * 1024)
                _logger.info("⚡ [FFmpeg Auto-Compress] Tối ưu clip {} từ {:.2f}MB ➔ {:.2f}MB!", video_path.name, file_size_mb, new_size_mb)
                return temp_compressed
        except Exception as exc:
            _logger.warning("Không thể nén video bằng FFmpeg, dùng file gốc: {}", exc)

        return video_path

    def _get_stage1_desc_gemini_api(
        self,
        frame_path: Optional[Path],
        clip_idx: int,
        total_clips: int,
        custom_instructions: str = "",
        clip_video_path: Optional[Path] = None,
    ) -> str:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ReviewError("Chưa cấu hình GEMINI_API_KEY hoặc OPENAI_API_KEY.")

        # BẮT BUỘC: Gửi trực tiếp file video clip (MP4/MOV/...) sang Gemini API, KHÔNG bao giờ gửi ảnh tĩnh.
        if not clip_video_path or not Path(clip_video_path).exists():
            _logger.error("❌ Clip #{}: Không tìm thấy file video clip để gửi tới Gemini API!", clip_idx + 1)
            raise ReviewError(f"Clip #{clip_idx + 1}: Không tìm thấy file video clip để gửi tới Gemini API.")

        media_path = self._compress_video_for_gemini(Path(clip_video_path))
        is_temp_opt = (media_path != Path(clip_video_path))

        def _call_shopaikey(prompt: str, video_file: Path, model: str = "gemini-2.5-flash") -> str:
            import requests
            url = f"https://api.shopaikey.com/v1beta/models/{model}:generateContent?key={api_key}"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            parts = [{"text": prompt}]
            ext = video_file.suffix.lower()
            mime_type = "video/quicktime" if ext == ".mov" else "video/mp4"
            file_size_mb = video_file.stat().st_size / (1024 * 1024)
            _logger.info("🎬 [Gemini API] Gửi trực tiếp file video clip {} ({:.2f} MB, {}) sang ShopAIKey...", video_file.name, file_size_mb, mime_type)
            with open(video_file, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            parts.append({"inlineData": {"mimeType": mime_type, "data": b64_data}})

            payload = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"temperature": 0.7}
            }

            for attempt in range(1, 4):
                try:
                    res = requests.post(url, headers=headers, json=payload, timeout=(120, 180))
                    if res.status_code == 200:
                        txt = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                        _logger.info("📥 [DEBUG STAGE 1 GEMINI RESPONSE] Clip {}:\n==================================================\n{}\n==================================================", video_file.name, txt)
                        if self._is_valid_stage1_description(txt):
                            return txt
                        _logger.warning("ShopAIKey Stage 1 attempt {}/3: Thiếu tag 'Review thành công' trong phản hồi. Đang thử lại...", attempt)
                    else:
                        _logger.warning("ShopAIKey Stage 1 attempt {}/3 HTTP error: {} - {}", attempt, res.status_code, res.text[:200])
                        if res.status_code == 503 or "model_not_found" in res.text:
                            _logger.warning("Model {} không khả dụng trên kênh ShopAIKey. Bỏ qua model này.", model)
                            break
                except Exception as exc:
                    _logger.warning("ShopAIKey Stage 1 attempt {}/3 upload/request error: {}", attempt, exc)
                if attempt < 3:
                    time.sleep(3.0)
            raise RuntimeError(f"ShopAIKey Gemini Stage 1 Video upload thất bại: thiếu tag 'Review thành công' sau 3 lần thử.")

        try:
            p1_file = Path("TRỢ LÝ QUAN SÁT VÀ MÔ TẢ VIDEO.txt")
            p1_template = p1_file.read_text(encoding="utf-8").strip() if p1_file.exists() else ""
            ui_prompt_job1 = getattr(self, "_prompt_job1", None)
            prompt_user = ui_prompt_job1.strip() if ui_prompt_job1 and ui_prompt_job1.strip() else custom_instructions.strip()
            prompt_base = prompt_user if prompt_user else p1_template
            prompt_stage1 = prompt_base if prompt_base else (
                "CÔNG VIỆC 1: TRỢ LÝ QUAN SÁT VÀ MÔ TẢ VIDEO/HÌNH ẢNH\n"
                "Mô tả chi tiết: Hành động chính, Nhân vật xuất hiện, Biểu cảm ngôn ngữ cơ thể và Mối tương tác."
            )
            if "review thành công" not in prompt_stage1.lower():
                prompt_stage1 += '\n\nQUAN TRỌNG: Hãy chắc chắn kết thúc bài mô tả của bạn bằng chính xác cụm từ: "Review thành công".'

            _logger.info("📤 [DEBUG STAGE 1 PROMPT] Clip {}:\n==================================================\n{}\n==================================================", media_path.name, prompt_stage1)

            stage1_desc = ""
            if api_key.startswith("sk-"):
                models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
                for m in models_to_try:
                    try:
                        stage1_desc = _call_shopaikey(prompt_stage1, media_path, m)
                        if stage1_desc and self._is_valid_stage1_description(stage1_desc):
                            break
                    except Exception as exc:
                        _logger.warning("ShopAIKey Gemini Stage 1 error on model {}: {}", m, exc)

            if not stage1_desc and genai is not None and not api_key.startswith("sk-"):
                try:
                    genai.configure(api_key=api_key)
                    _logger.info("🎬 [Gemini API] Uploading video clip {} tới Gemini File API...", media_path.name)
                    uploaded_file_ref = genai.upload_file(media_path)
                    while getattr(uploaded_file_ref, "state", None) and uploaded_file_ref.state.name == "PROCESSING":
                        time.sleep(2)
                        uploaded_file_ref = genai.get_file(uploaded_file_ref.name)

                    if getattr(uploaded_file_ref, "state", None) and uploaded_file_ref.state.name == "FAILED":
                        raise RuntimeError(f"Gemini File API processing failed for video clip: {uploaded_file_ref.name}")

                    models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
                    for m in models:
                        try:
                            model = genai.GenerativeModel(m)
                            resp = model.generate_content([uploaded_file_ref, prompt_stage1])
                            txt = resp.text.strip()
                            if txt and self._is_valid_stage1_description(txt):
                                stage1_desc = txt
                                break
                        except Exception as m_err:
                            _logger.warning("Gemini model {} error: {}", m, m_err)
                            continue

                    if uploaded_file_ref:
                        try:
                            genai.delete_file(uploaded_file_ref.name)
                        except Exception:
                            pass

                except Exception as exc:
                    _logger.warning("GenerativeAI SDK error: {}", exc)

            if stage1_desc and self._is_valid_stage1_description(stage1_desc):
                _logger.info("✅ [Gemini API Stage 1] Phân tích video clip thành công ({} ký tự)!", len(stage1_desc))
                return stage1_desc
            _logger.error("Gemini API Stage 1: Không nhận được kết quả hợp lệ từ API.")
            raise ReviewError(f"Clip #{clip_idx + 1}: Không nhận được bài mô tả video hợp lệ có tag 'Review thành công' từ Gemini API.")
        except Exception as exc:
            _logger.error("Stage 1 Gemini API error on clip {}: {}", clip_idx + 1, exc)
            raise ReviewError(f"Clip #{clip_idx + 1}: Lỗi phân tích video từ Gemini API: {exc}") from exc
        finally:
            if is_temp_opt and media_path.exists():
                try:
                    media_path.unlink()
                except Exception:
                    pass

    def _get_stage2_script_gemini_api(
        self, stage1_desc: str, clip_idx: int, total_clips: int, language: str = "vi", custom_instructions: str = ""
    ) -> str:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Chưa cấu hình GEMINI_API_KEY hoặc OPENAI_API_KEY cho Stage 2.")

        def _call_shopaikey(prompt: str, model: str = "gemini-2.5-flash") -> str:
            import requests
            url = f"https://api.shopaikey.com/v1beta/models/{model}:generateContent?key={api_key}"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7}
            }
            for attempt in range(1, 4):
                try:
                    res = requests.post(url, headers=headers, json=payload, timeout=30)
                    if res.status_code == 200:
                        txt = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                        _logger.info("📥 [DEBUG STAGE 2 GEMINI RESPONSE] Clip {}:\n==================================================\n{}\n==================================================", clip_idx + 1, txt)
                        return txt
                    _logger.warning("ShopAIKey Stage 2 attempt {}/3 HTTP error (model {}): {} - {}", attempt, model, res.status_code, res.text[:200])
                    if res.status_code == 503 or "model_not_found" in res.text:
                        _logger.warning("Model {} không khả dụng trên kênh ShopAIKey. Bỏ qua model này.", model)
                        break
                except Exception as exc:
                    _logger.warning("ShopAIKey Stage 2 attempt {}/3 request error (model {}): {}", attempt, model, exc)
                if attempt < 3:
                    time.sleep(3.0)
            raise RuntimeError(f"ShopAIKey REST Error (model {model}) thất bại sau 3 lần thử.")

        try:
            p2_file = Path("TRỢ LÝ VIẾT CONTENT NGẮN CHUYÊN NGHIỆP.txt")
            p2_template = p2_file.read_text(encoding="utf-8").strip() if p2_file.exists() else ""
            ui_prompt_job2 = getattr(self, "_prompt_job2", None)
            prompt_user_j2 = ui_prompt_job2.strip() if ui_prompt_job2 and ui_prompt_job2.strip() else ""
            base_p2 = prompt_user_j2 if prompt_user_j2 else p2_template

            stage1_context = f"\n\nTHÔNG TIN ĐẦU VÀO TỪ CÔNG VIỆC 1:\n{stage1_desc}" if stage1_desc else ""
            custom_addon = f"\n\nYÊU CẦU BỔ SUNG TỪ NGƯỜI DÙNG:\n{custom_instructions}" if custom_instructions.strip() else ""

            prompt_stage2 = f"{base_p2}{stage1_context}{custom_addon}" if base_p2 else (
                "CÔNG VIỆC 2: TRỢ LÝ VIẾT CONTENT NGẮN CHUYÊN NGHIỆP\n"
                f"BẢN MÔ TẢ HÌNH ẢNH:\n{stage1_desc}\n\n"
                f"{custom_instructions}"
            )
            if "viết content thành công" not in prompt_stage2.lower():
                prompt_stage2 += '\n\nQUAN TRỌNG: Hãy chắc chắn kết thúc bài viết của bạn bằng chính xác cụm từ: "Viết content thành công".'

            _logger.info("📤 [DEBUG STAGE 2 GEMINI PROMPT] Clip {}:\n==================================================\n{}\n==================================================", clip_idx + 1, prompt_stage2)

            if api_key.startswith("sk-"):
                models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
                for m in models_to_try:
                    for attempt in range(1, 4):
                        try:
                            txt = _call_shopaikey(prompt_stage2, m)
                            if txt and self._is_valid_stage2_script(txt):
                                return self._strip_stage2_tag(txt).strip('"').strip("'")
                            _logger.warning("ShopAIKey Stage 2 attempt {}/3 (model {}): Thiếu tag 'Viết content thành công'. Đang thử lại...", attempt, m)
                        except Exception as exc:
                            _logger.warning("ShopAIKey Gemini Stage 2 error model {}: {}", m, exc)
                            break

            if genai is not None and not api_key.startswith("sk-"):
                models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro"]
                for m in models:
                    try:
                        model = genai.GenerativeModel(m)
                        resp = model.generate_content(prompt_stage2)
                        txt = resp.text.strip().strip('"').strip("'")
                        if txt and self._is_valid_stage2_script(txt):
                            return self._strip_stage2_tag(txt)
                    except Exception:
                        continue
        except Exception as exc:
            _logger.error("Gemini API Stage 2 error on clip {}: {}", clip_idx + 1, exc)
            raise RuntimeError(f"Clip #{clip_idx + 1}: Lỗi tạo kịch bản từ Gemini API: {exc}") from exc

        raise RuntimeError(f"Clip #{clip_idx + 1}: Không nhận được kịch bản hợp lệ có tag 'Viết content thành công' từ Gemini API.")

    def _get_stage1_desc_openai_api(
        self, frame_path: Optional[Path], clip_idx: int, total_clips: int, custom_instructions: str = ""
    ) -> str:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("ChatGPT API error: Không tìm thấy OPENAI_API_KEY hoặc GEMINI_API_KEY.")

        import requests
        p1_file = Path("TRỢ LÝ QUAN SÁT VÀ MÔ TẢ VIDEO.txt")
        p1_template = p1_file.read_text(encoding="utf-8").strip() if p1_file.exists() else ""
        prompt_stage1 = p1_template if p1_template else (
            "CÔNG VIỆC 1: TRỢ LÝ QUAN SÁT VÀ MÔ TẢ VIDEO/HÌNH ẢNH\n"
            "Mô tả chi tiết: Hành động chính, Nhân vật xuất hiện, Biểu cảm ngôn ngữ cơ thể và Mối tương tác."
        )
        prompt_stage1 += "\nQUAN TRỌNG: Kết thúc bài mô tả bằng đúng cụm từ: \"Review thành công\"."
        content_list: List[dict] = [{"type": "text", "text": prompt_stage1}]
        if frame_path and frame_path.exists():
            with open(frame_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "messages": [{"role": "user", "content": content_list}], "temperature": 0.7}
        url = os.getenv("OPENAI_BASE_URL", "https://api.shopaikey.com/v1")
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"
        for attempt in range(1, 4):
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=120)
                if res.status_code == 200:
                    txt = res.json()["choices"][0]["message"]["content"].strip()
                    if self._is_valid_stage1_description(txt):
                        return txt
                    _logger.warning("OpenAI S1 clip {} attempt {}/3: thiếu tag 'Review thành công'.", clip_idx + 1, attempt)
                else:
                    _logger.warning("OpenAI S1 clip {} attempt {}/3: HTTP {}", clip_idx + 1, attempt, res.status_code)
                if attempt < 3:
                    time.sleep(3.0)
            except Exception as exc:
                _logger.warning("OpenAI S1 clip {} attempt {}/3 error: {}", clip_idx + 1, attempt, exc)
                if attempt < 3:
                    time.sleep(2.0)
        raise RuntimeError(f"ChatGPT API Stage 1 lỗi tại clip {clip_idx + 1}: không có 'Review thành công' sau 3 lần thử.")

    def _get_stage1_desc_claude_api(
        self, frame_path: Optional[Path], clip_idx: int, total_clips: int, custom_instructions: str = ""
    ) -> str:
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        if not api_key:
            raise RuntimeError("Claude API error: Không tìm thấy API Key cho Claude API.")

        import requests
        p1_file = Path("TRỢ LÝ QUAN SÁT VÀ MÔ TẢ VIDEO.txt")
        p1_template = p1_file.read_text(encoding="utf-8").strip() if p1_file.exists() else ""
        prompt_stage1 = p1_template if p1_template else (
            "CÔNG VIỆC 1: TRỢ LÝ QUAN SÁT VÀ MÔ TẢ VIDEO/HÌNH ẢNH\n"
            "Mô tả chi tiết: Hành động chính, Nhân vật xuất hiện, Biểu cảm ngôn ngữ cơ thể và Mối tương tác."
        )
        prompt_stage1 += "\nQUAN TRỌNG: Kết thúc bài mô tả bằng đúng cụm từ: \"Review thành công\"."
        content_list: List[dict] = [{"type": "text", "text": prompt_stage1}]
        if frame_path and frame_path.exists():
            with open(frame_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        model_name = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model_name, "messages": [{"role": "user", "content": content_list}], "temperature": 0.7}
        url = os.getenv("OPENAI_BASE_URL", "https://api.shopaikey.com/v1")
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"
        for attempt in range(1, 4):
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=120)
                if res.status_code == 200:
                    txt = res.json()["choices"][0]["message"]["content"].strip()
                    if self._is_valid_stage1_description(txt):
                        return txt
                    _logger.warning("Claude S1 clip {} attempt {}/3: thiếu tag 'Review thành công'.", clip_idx + 1, attempt)
                else:
                    _logger.warning("Claude S1 clip {} attempt {}/3: HTTP {}", clip_idx + 1, attempt, res.status_code)
                if attempt < 3:
                    time.sleep(3.0)
            except Exception as exc:
                _logger.warning("Claude S1 clip {} attempt {}/3 error: {}", clip_idx + 1, attempt, exc)
                if attempt < 3:
                    time.sleep(2.0)
        raise RuntimeError(f"Claude API Stage 1 lỗi tại clip {clip_idx + 1}: không có 'Review thành công' sau 3 lần thử.")

    def _get_stage2_script_claude_api(
        self, stage1_desc: str, clip_idx: int, total_clips: int, language: str = "vi", custom_instructions: str = ""
    ) -> str:
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        if not api_key:
            raise RuntimeError("Claude API error: Không tìm thấy API Key cho Claude API.")

        import requests
        p2_file = Path("TRỢ LÝ VIẾT CONTENT NGẮN CHUYÊN NGHIỆP.txt")
        p2_template = p2_file.read_text(encoding="utf-8").strip() if p2_file.exists() else ""
        stage1_context = f"\n\nTHÔNG TIN ĐẦU VÀO TỪ CÔNG VIỆC 1:\n{stage1_desc}" if stage1_desc else ""
        custom_addon = f"\n\nYÊU CẦU BỔ SUNG TỪ NGƯỚI DÙNG:\n{custom_instructions}" if custom_instructions.strip() else ""
        prompt_stage2 = f"{p2_template}{stage1_context}{custom_addon}" if p2_template else (
            "CÔNG VIỆC 2: TRỢ LÝ VIẾT CONTENT NGẮN CHUYÊN NGHIỆP\n"
            f"BẢN MÔ TẢ HÌNH ẢNH:\n{stage1_desc}\n\n{custom_instructions}"
        )
        prompt_stage2 += "\nQUAN TRỌNG: Kết thúc nội dung bằng đúng cụm từ: \"Viết content thành công\"."
        _logger.info("📤 [DEBUG STAGE 2 CLAUDE PROMPT] Clip {}:\n==================================================\n{}\n==================================================", clip_idx + 1, prompt_stage2)
        model_name = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model_name, "messages": [{"role": "user", "content": prompt_stage2}], "temperature": 0.7}
        url = os.getenv("OPENAI_BASE_URL", "https://api.shopaikey.com/v1")
        if not url.endswith("/chat/completions"):
            url = url.rstrip("/") + "/chat/completions"
        for attempt in range(1, 4):
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=120)
                if res.status_code == 200:
                    txt = res.json()["choices"][0]["message"]["content"].strip().strip('"').strip("'")
                    _logger.info("📥 [DEBUG STAGE 2 CLAUDE RESPONSE] Clip {}:\n==================================================\n{}\n==================================================", clip_idx + 1, txt)
                    if txt and self._is_valid_stage2_script(txt):
                        return self._strip_stage2_tag(txt)
                    _logger.warning("Claude S2 clip {} attempt {}/3: Thiếu tag 'Viết content thành công'. Đang thử lại...", clip_idx + 1, attempt)
                else:
                    _logger.warning("Claude S2 clip {} attempt {}/3: HTTP {}", clip_idx + 1, attempt, res.status_code)
                if attempt < 3:
                    time.sleep(3.0)
            except Exception as exc:
                _logger.warning("Claude S2 clip {} attempt {}/3 error: {}", clip_idx + 1, attempt, exc)
                if attempt < 3:
                    time.sleep(2.0)
        raise RuntimeError(f"Claude API Stage 2 lỗi tại clip {clip_idx + 1}: không có 'Viết content thành công' sau 3 lần thử.")

    def _get_stage2_script_openai_api(
        self, stage1_desc: str, clip_idx: int, total_clips: int, language: str = "vi", custom_instructions: str = ""
    ) -> str:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        if not api_key:
            raise RuntimeError("ChatGPT API error: Không tìm thấy API Key.")

        try:
            import requests
            p2_file = Path("TRỢ LÝ VIẾT CONTENT NGẮN CHUYÊN NGHIỆP.txt")
            p2_template = p2_file.read_text(encoding="utf-8").strip() if p2_file.exists() else ""
            ui_prompt_job2 = getattr(self, "_prompt_job2", None)
            prompt_user_j2 = ui_prompt_job2.strip() if ui_prompt_job2 and ui_prompt_job2.strip() else ""
            base_p2 = prompt_user_j2 if prompt_user_j2 else p2_template

            stage1_context = f"\n\nTHÔNG TIN ĐẦU VÀO TỪ CÔNG VIỆC 1:\n{stage1_desc}" if stage1_desc else ""
            custom_addon = f"\n\nYÊU CẦU BỔ SUNG TỪ NGƯỜI DÙNG:\n{custom_instructions}" if custom_instructions.strip() else ""

            prompt_stage2 = f"{base_p2}{stage1_context}{custom_addon}" if base_p2 else (
                "CÔNG VIỆC 2: TRỢ LÝ VIẾT CONTENT NGẮN CHUYÊN NGHIỆP\n"
                f"BẢN MÔ TẢ HÌNH ẢNH:\n{stage1_desc}\n\n"
                f"{custom_instructions}"
            )
            if "viết content thành công" not in prompt_stage2.lower():
                prompt_stage2 += '\n\nQUAN TRỌNG: Hãy chắc chắn kết thúc bài viết của bạn bằng chính xác cụm từ: "Viết content thành công".'

            _logger.info("📤 [DEBUG STAGE 2 OPENAI PROMPT] Clip {}:\n==================================================\n{}\n==================================================", clip_idx + 1, prompt_stage2)

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt_stage2}],
                "temperature": 0.7
            }
            url = os.getenv("OPENAI_BASE_URL", "https://api.shopaikey.com/v1")
            if not url.endswith("/chat/completions"):
                url = url.rstrip("/") + "/chat/completions"

            for attempt in range(1, 4):
                try:
                    res = requests.post(url, headers=headers, json=payload, timeout=120)
                    if res.status_code == 200:
                        data = res.json()
                        txt = data["choices"][0]["message"]["content"].strip().strip('"').strip("'")
                        _logger.info("📥 [DEBUG STAGE 2 OPENAI RESPONSE] Clip {}:\n==================================================\n{}\n==================================================", clip_idx + 1, txt)
                        if txt and self._is_valid_stage2_script(txt):
                            return self._strip_stage2_tag(txt)
                        _logger.warning("OpenAI Stage 2 attempt {}/3: Thiếu tag 'Viết content thành công'. Đang thử lại...", attempt)
                    if attempt < 3:
                        time.sleep(3.0)
                except Exception as exc:
                    _logger.warning("OpenAI Stage 2 attempt {}/3 error: {}", attempt, exc)
                    if attempt < 3:
                        time.sleep(2.0)
            raise RuntimeError(f"OpenAI API Stage 2 lỗi tại clip {clip_idx + 1}: không có 'Viết content thành công' sau 3 lần thử.")
        except Exception as exc:
            _logger.error("ChatGPT API Stage 2 error on clip {}: {}", clip_idx + 1, exc)
            raise RuntimeError(f"ChatGPT API Stage 2 lỗi tại clip {clip_idx + 1}: {exc}")

    def _generate_for_clip_gemini_api(
        self, frame_path: Optional[Path], clip_idx: int, total_clips: int, language: str = "vi", custom_instructions: str = ""
    ) -> str:
        desc = self._get_stage1_desc_gemini_api(frame_path, clip_idx, total_clips, custom_instructions)
        return self._get_stage2_script_gemini_api(desc, clip_idx, total_clips, language, custom_instructions)

    def _generate_for_clip_openai(
        self, frame_path: Optional[Path], clip_idx: int, total_clips: int, language: str = "vi", custom_instructions: str = ""
    ) -> str:
        desc = self._get_stage1_desc_openai_api(frame_path, clip_idx, total_clips, custom_instructions)
        return self._get_stage2_script_openai_api(desc, clip_idx, total_clips, language, custom_instructions)

    # ------------------------------------------------------------------
    # Main generate() entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        timeline_result: TimelineResult,
        review_style: str = "documentary",
        language: str = "vi",
        target_duration: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Result[ReviewResult, ReviewError]:
        """Iterate through each clip in the timeline, query AI per-clip, and combine scripts."""
        start_ts = time.time()
        job_id = kwargs.get("job_id", "default")
        custom_instructions = kwargs.get("custom_instructions") or ""
        progress_cb: Optional[Callable[[str, float], None]] = kwargs.get("progress_callback")

        video_info = kwargs.get("video_info")
        clips_list = [
            p for p in getattr(video_info, "clips", [])
            if Path(p).exists() and not Path(p).name.endswith("_opt.mp4") and not Path(p).name.startswith(".")
        ] if video_info else []

        if clips_list and len(clips_list) > 0:
            total_clips = len(clips_list)
        else:
            events = getattr(getattr(timeline_result, "timeline", None), "events", []) if timeline_result else []
            total_clips = len(events) if events else 1

        _logger.info("Generating per-clip review script for {} clips (Review Engine: {}, Write Engine: {})...", total_clips, self._review_video_engine, self._write_content_engine)

        clip_scripts: List[str] = []
        frames_dir = Path("data") / "jobs" / job_id / "frames"
        if not frames_dir.exists():
            all_job_dirs = [p for p in Path("data").glob("jobs/*/frames") if p.is_dir()]
            if all_job_dirs:
                frames_dir = sorted(all_job_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                _logger.info("Using frames directory: {}", frames_dir)

        # Determine AI backend mode
        use_web_browser = bool(
            self._browser_mgr and (
                self._review_video_engine == "gemini_web" or
                self._write_content_engine in ("chatgpt_web", "claude_web")
            )
        )

        if use_web_browser:
            self._setup_gemini_web_tabs(total_clips, progress_cb, custom_instructions=custom_instructions)

        # ── Process each clip ──
        for i in range(total_clips):
            clip_num = i + 1
            if progress_cb:
                pct = 0.80 + (i / total_clips) * 0.18  # range 80%→98%
                progress_cb(f"Clip {clip_num}/{total_clips}", pct)

            clip_video_path = None
            if video_info:
                clips_arr = getattr(video_info, "clips", [])
                if clips_arr and i < len(clips_arr) and Path(clips_arr[i]).exists():
                    clip_video_path = Path(clips_arr[i])
                elif getattr(video_info, "video_path", None) and Path(video_info.video_path).exists():
                    clip_video_path = Path(video_info.video_path)
                elif isinstance(video_info, Path) and video_info.exists():
                    clip_video_path = video_info
                elif getattr(video_info, "path", None) and Path(video_info.path).exists():
                    clip_video_path = Path(video_info.path)

            frame_path = frames_dir / f"scene{i:04d}_frame00.jpg"
            if not frame_path.exists():
                possible = list(frames_dir.glob(f"scene{i:04d}_frame*.jpg"))
                frame_path = possible[0] if possible else None

            _logger.info("Processing review for clip {}/{} (Review Engine: {}, Write Engine: {})...", clip_num, total_clips, self._review_video_engine, self._write_content_engine)

            # --- STAGE 1: Video Description ---
            if self._review_video_engine == "gemini_web" and self._browser_mgr:
                stage1_desc = self._get_stage1_desc_gemini_web(
                    i, total_clips, job_id=job_id, clip_video_path=clip_video_path, progress_cb=progress_cb
                )
            elif self._review_video_engine == "openai_api":
                stage1_desc = self._get_stage1_desc_openai_api(
                    frame_path, i, total_clips, custom_instructions=custom_instructions
                )
            elif self._review_video_engine == "claude_api":
                stage1_desc = self._get_stage1_desc_claude_api(
                    frame_path, i, total_clips, custom_instructions=custom_instructions
                )
            else:
                stage1_desc = self._get_stage1_desc_gemini_api(
                    frame_path, i, total_clips, custom_instructions=custom_instructions, clip_video_path=clip_video_path
                )

            watermark_val = self._extract_watermark(stage1_desc)
            if watermark_val:
                _logger.info("📌 [Watermark/Chữ trên video] Clip {}: '{}'", clip_num, watermark_val)

            # --- STAGE 2: Content Writing ---
            if self._write_content_engine == "openai_api":
                script_part = self._get_stage2_script_openai_api(
                    stage1_desc, i, total_clips, language=language, custom_instructions=custom_instructions
                )
            elif self._write_content_engine == "claude_api":
                script_part = self._get_stage2_script_claude_api(
                    stage1_desc, i, total_clips, language=language, custom_instructions=custom_instructions
                )
            elif self._write_content_engine in ("chatgpt_web", "claude_web") and self._browser_mgr:
                script_part = self._generate_for_clip_web_stage2(
                    stage1_desc, i, total_clips, custom_instructions=custom_instructions, job_id=job_id, clip_video_path=clip_video_path, progress_cb=progress_cb
                )
            else:
                script_part = self._get_stage2_script_gemini_api(
                    stage1_desc, i, total_clips, language=language, custom_instructions=custom_instructions
                )
            
            clip_scripts.append(script_part)
            
            # ── Instantly push THIS clip's content to Google Sheet Webhook ──
            try:
                import re
                stt_val = str(clip_num)
                if clip_video_path:
                    digits = re.findall(r'\d+', Path(clip_video_path).stem)
                    if digits:
                        stt_val = str(int(digits[-1]))
                
                voice_name = f"AT-{int(stt_val):02d}.mp3" if stt_val.isdigit() else f"AT-{stt_val}.mp3"
                sheet_webhook = kwargs.get("google_sheet_webhook_url") or os.getenv("GOOGLE_SHEET_WEBHOOK_URL", "")
                
                if sheet_webhook:
                    from src.exporter.google_sheet_exporter import GoogleSheetExporter
                    GoogleSheetExporter.sync_single_clip_to_sheet(
                        stt=stt_val,
                        content=script_part,
                        voice_filename=voice_name,
                        webhook_url=sheet_webhook,
                        job_id=job_id,
                        watermark=watermark_val,
                    )
            except Exception as live_err:
                _logger.warning("Live push clip {} error: {}", clip_num, live_err)

        final_script = "\n\n".join(clip_scripts)
        total_words = len(final_script.split())
        elapsed_ms = (time.time() - start_ts) * 1000

        res = ReviewResult(
            title=f"Kịch bản Thuyết minh ({total_clips} Clips)",
            hook=clip_scripts[0] if clip_scripts else "",
            script=final_script,
            metadata=ReviewMetadata(
                total_words=total_words,
                estimated_duration=round(total_words / 2.5, 1),
                model_name="multi-agent-2stage",
                processing_time=round(elapsed_ms / 1000.0, 2),
                style=review_style,
                language=language,
                target_duration=target_duration,
                word_count=total_words,
                generation_duration_ms=elapsed_ms,
            ),
        )

        _logger.info("Finished per-clip review generation ({} clips, {} total words)", total_clips, total_words)
        return Result.Ok(res)

    @staticmethod
    def _extract_watermark(stage1_text: str) -> str:
        """Trích xuất chữ / watermark / ID channel ở góc màn hình từ bài phân tích Stage 1 của Gemini."""
        if not stage1_text:
            return ""
        import re
        pattern = r"(?:Watermark|Chữ|Text|Logo)(?:\s*/\s*(?:Text|Chữ))?(?:\s*trên\s*video|\s*ở\s*góc\s*màn\s*hình|\s*màn\s*hình)?\s*:\s*([\s\S]*?)(?=\n\s*\*\*|\n\s*[A-ZÀ-Ỹ0-9\-\.\#\:\_]{3,}\s*:|$)"
        m = re.search(pattern, stage1_text, re.IGNORECASE)
        if m:
            raw_text = m.group(1).strip()
            raw_text = re.sub(r"\*+", "", raw_text).strip()
            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            if not lines:
                return ""
            full_line = " ".join(lines)
            
            if full_line.lower() in ("không có", "khong co", "none", "n/a", "không", "không thấy", "chưa thấy", "không xuất hiện", "trường hợp này không có"):
                return ""
            
            # Look for quotes e.g. "Cr: Terry Noah" or "Cr: Djamel Hadj Aissa"
            quotes = re.findall(r'["\']([^"\']+)["\']', full_line)
            if quotes:
                return quotes[0].strip()
                
            # Look for @username
            user_match = re.search(r'@[A-Za-z0-9_.]+', full_line)
            if user_match:
                return user_match.group(0).strip()
                
            # Remove prefix chatter if followed by text
            full_line = re.sub(r"^(?:Góc\s*(?:dưới|trên)?\s*(?:bên\s*)?(?:trái|phải)?\s*(?:màn\s*hình)?\s*(?:có\s*dòng\s*chữ|có\s*chữ|hiển\s*thị)?\s*:?\s*)", "", full_line, flags=re.IGNORECASE).strip()
            
            if full_line.lower() in ("không có", "khong co", "none", "n/a", "không"):
                return ""
            return full_line
        return ""

    def generate_async(self, *args, **kwargs):
        raise NotImplementedError("Use synchronous generate")

    def health_check(self) -> Result[bool, ReviewError]:
        if self._browser_mgr:
            try:
                session_mgr = self._browser_mgr._session_mgr
                if self._review_video_engine == "gemini_web" and not session_mgr.has_session_file():
                    return Result.Err(ReviewError("❌ [Lỗi Gemini Web] Chưa có Cookie Gemini Web (Tab 1). Vui lòng bấm '🍪 Cookie Gemini' để nạp cookie!"))
                if self._write_content_engine == "chatgpt_web" and not session_mgr.has_chatgpt_session():
                    return Result.Err(ReviewError("❌ [Lỗi ChatGPT Web] Chưa có Cookie ChatGPT Web (Tab 2). Vui lòng bấm '🤖 Cookie ChatGPT' để nạp cookie!"))
                if self._write_content_engine == "claude_web" and not session_mgr.has_claude_session():
                    return Result.Err(ReviewError("❌ [Lỗi Claude Web] Chưa có Cookie Claude Web (Tab 2). Vui lòng bấm '🦙 Cookie Claude' để nạp cookie!"))
                return Result.Ok(True)
            except Exception as exc:
                return Result.Err(ReviewError(f"❌ [Lỗi Browser Session] Lỗi kiểm tra cookie: {exc}"))
        return Result.Ok(True)
