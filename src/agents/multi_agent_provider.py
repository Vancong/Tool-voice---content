"""Clip-by-Clip Review Provider sending each pre-cut video clip to Gemini / ChatGPT / Gemini Web
to generate per-clip narration text for TTS voiceover.
"""

from __future__ import annotations

import base64
import os
import time
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

        # ── Tab 1: Setup Observer Role ──
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
        """Validate if Tab 1 response is complete by checking strictly for 'review thành công' tag."""
        if not text or len(text.strip()) < 100:
            return False
        t_lower = text.strip().lower()

        # Reject explicit refusal or cutoff indicator lines
        invalid_keywords = [
            "tôi đã nắm rõ",
            "tôi sẽ tuân thủ",
            "vui lòng tải lên video",
            "vui lòng cung cấp",
            "chưa có dữ liệu clip",
            "bạn chưa tải",
            "không có dữ liệu",
            "không thể phân tích",
            "bạn đã dừng câu trả lời này",
            "bạn đã dừng",
        ]
        for ik in invalid_keywords:
            if ik in t_lower:
                return False

        # Strictly check for the required 'review thành công' completion phrase
        if "review thành công" in t_lower:
            return True

        _logger.warning("Stage 1 description rejected: missing 'Review thành công' tag at the end ({} chars)", len(text))
        return False

    def _generate_for_clip_gemini_web(
        self,
        clip_idx: int,
        total_clips: int,
        language: str = "vi",
        custom_instructions: str = "",
        job_id: str = "web",
        clip_video_path: Optional[Path] = None,
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> str:
        """
        PER CLIP:
          Tab 1 ← video file → mô tả
          Tab 2 ← mô tả từ Tab 1 → lời bình hoàn chỉnh
        """
        if not self._browser_mgr:
            return f"Phân cảnh {clip_idx + 1} diễn ra vô cùng cuốn hút."

        clip_num = clip_idx + 1
        base_pct = clip_idx / total_clips

        # ── Tab 1: Send video, get description ──
        tab1_label = "Google AI Studio" if self._review_video_engine == "google_ai_studio" else "Gemini Web"
        if progress_cb:
            progress_cb(f"Clip {clip_num}/{total_clips} (Tab 1: Gửi video qua {tab1_label})", base_pct + 0.01)

        # Check if custom clip title exists for this video clip
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
            _logger.info("Attached custom clip title to Tab 1 prompt (clip {}): '{}'", clip_num, clip_title_str)

        custom_addon = f"\n\nYÊU CẦU BỔ SUNG:\n{custom_instructions}" if custom_instructions.strip() else ""
        video_prompt += custom_addon

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
                    _logger.warning("Tab 1 attempt {}/{} returned invalid/boilerplate response ({} chars): '{}...'. Retrying video prompt...", attempt, max_attempts, len(raw_desc), raw_desc[:80])
                    time.sleep(3.0)
            except Exception as exc:
                _logger.warning("Tab 1 error on clip {} (Attempt {}/{}): {}", clip_num, attempt, max_attempts, exc)
                time.sleep(2.0)

        if not stage1_desc or not self._is_valid_stage1_description(stage1_desc):
            _logger.error("❌ Tab 1 failed to return valid video description for Clip {} after {} attempts!", clip_num, max_attempts)
            raise ReviewError(f"Clip {clip_num}: Gemini (Tab 1) không thể phân tích video sau {max_attempts} lần thử (thiếu tag 'Review thành công' hoặc bị ngắt). Đã dừng chương trình để bảo đảm dữ liệu.")

        # ── Tab 2: Send Tab1 description, get final script ──
        if progress_cb:
            progress_cb(f"Clip {clip_num}/{total_clips} (Tab 2: Viết lời bình)", base_pct + 0.04)

        writer_prompt = (
            f"[CLIP #{clip_num}/{total_clips}]\n\n"
            f"THÔNG TIN ĐẦU VÀO TỪ CÔNG VIỆC 1 (Clip #{clip_num}/{total_clips}):\n"
            f"{stage1_desc}"
        )
        if clip_title_str:
            writer_prompt += f"\n\nTIÊU ĐỀ / CHỦ ĐỀ CLIP:\n{clip_title_str}"
        if custom_instructions.strip():
            writer_prompt += f"\n\nYÊU CẦU BỔ SUNG:\n{custom_instructions}"

        try:
            _logger.info("Tab 2: Sending description for clip {} to content writer...", clip_num)
            resp2 = self._browser_mgr.send_prompt_to_stage(
                stage_idx=2,
                prompt=writer_prompt,
                media_path=None,
                job_id=f"{job_id}_clip{clip_idx}_tab2",
                progress_callback=progress_cb,
                write_content_engine=self._write_content_engine,
            )
            txt = resp2.text.strip()
            for prefix in [
                "claude responded:",
                "claude responded",
                "claude said:",
                "claude:",
                "chatgpt responded:",
                "chatgpt responded",
                "chatgpt said:",
                "chatgpt:",
                "gemini đã nói:",
                "gemini đã nói",
                "gemini said:",
                "gemini said",
                "gemini:",
            ]:
                if txt.lower().startswith(prefix):
                    txt = txt[len(prefix):].strip()
            txt = txt.strip('"').strip("'").strip()
            
            # Deduplicate repeated identical lines/paragraphs and nested echos/substrings
            p_lines = [p.strip() for p in txt.splitlines() if p.strip()]
            unique_p = []
            for p in p_lines:
                p_clean = p.strip("…").strip(".").strip()
                p_normalized = p_clean.lower()
                # Skip header/prompt echo lines if Claude echos input prompt
                if any(k in p_normalized for k in ["thông tin đầu vào", "công việc 1", "công việc 2", "tiêu đề / chủ đề"]):
                    continue
                
                # Check if this line is a prefix/substring of an existing line, or vice versa
                is_sub = False
                for idx, existing in enumerate(unique_p):
                    ex_clean = existing.strip("…").strip(".").strip()
                    ex_norm = ex_clean.lower()
                    if p_normalized in ex_norm or ex_norm.startswith(p_normalized[:40]):
                        is_sub = True
                        break
                    elif ex_norm in p_normalized or p_normalized.startswith(ex_norm[:40]):
                        # Current line is longer and contains previous line -> replace previous line
                        unique_p[idx] = p
                        is_sub = True
                        break
                
                if not is_sub:
                    unique_p.append(p)

            txt = "\n\n".join(unique_p).strip()

            _logger.info("Tab 2 final script (clip {}): {}", clip_num, txt)
            return txt
        except Exception as exc:
            _logger.error("Tab 2 error on clip {}: {}", clip_num, exc)
            return f"Phân cảnh {clip_num} tiếp tục với những tình huống nhiều bất ngờ."

    # ------------------------------------------------------------------
    # Gemini API (Cloud): 2-stage pipeline
    # ------------------------------------------------------------------

    def _generate_for_clip_gemini_api(
        self, frame_path: Optional[Path], clip_idx: int, total_clips: int, language: str = "vi", custom_instructions: str = ""
    ) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or genai is None:
            return f"Phân cảnh {clip_idx + 1} diễn ra kịch tính với nhiều tình tiết lôi cuốn."

        try:
            genai.configure(api_key=api_key)
            models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro"]

            p1_file = Path("TRỢ LÝ QUAN SÁT VÀ MÔ TẢ VIDEO.txt")
            p2_file = Path("TRỢ LÝ VIẾT CONTENT NGẮN CHUYÊN NGHIỆP.txt")
            p1_template = p1_file.read_text(encoding="utf-8").strip() if p1_file.exists() else ""
            p2_template = p2_file.read_text(encoding="utf-8").strip() if p2_file.exists() else ""

            prompt_stage1 = p1_template if p1_template else (
                "CÔNG VIỆC 1: TRỢ LÝ QUAN SÁT VÀ MÔ TẢ VIDEO/HÌNH ẢNH\n"
                "Mô tả chi tiết: Hành động chính, Nhân vật xuất hiện, Biểu cảm ngôn ngữ cơ thể và Mối tương tác."
            )

            stage1_desc = ""
            for m in models:
                try:
                    model = genai.GenerativeModel(m)
                    if frame_path and frame_path.exists():
                        img = Image.open(frame_path)
                        resp = model.generate_content([img, prompt_stage1])
                    else:
                        resp = model.generate_content(prompt_stage1)
                    stage1_desc = resp.text.strip()
                    if stage1_desc:
                        break
                except Exception as exc:
                    _logger.warning("Gemini Stage 1 fallback model {} on clip {}: {}", m, clip_idx + 1, exc)
                    continue

            if not stage1_desc:
                stage1_desc = f"Phân cảnh {clip_idx + 1} xuất hiện các nhân vật tương tác tự nhiên với nhau."

            _logger.info("Stage 1 Description (Clip {}): {}", clip_idx + 1, stage1_desc[:120])

            stage1_context = f"\n\nTHÔNG TIN ĐẦU VÀO TỪ CÔNG VIỆC 1:\n{stage1_desc}" if stage1_desc else ""
            custom_addon = f"\n\nYÊU CẦU BỔ SUNG TỪ NGƯỜI DÙNG:\n{custom_instructions}" if custom_instructions.strip() else ""

            prompt_stage2 = f"{p2_template}{stage1_context}{custom_addon}" if p2_template else (
                "CÔNG VIỆC 2: TRỢ LÝ VIẾT CONTENT NGẮN CHUYÊN NGHIỆP\n"
                f"BẢN MÔ TẢ HÌNH ẢNH:\n{stage1_desc}\n\n"
                f"{custom_instructions}"
            )

            for m in models:
                try:
                    model = genai.GenerativeModel(m)
                    resp = model.generate_content(prompt_stage2)
                    txt = resp.text.strip().strip('"').strip("'")
                    if txt:
                        return txt
                except Exception as exc:
                    _logger.warning("Gemini Stage 2 fallback model {} on clip {}: {}", m, clip_idx + 1, exc)
                    continue

        except Exception as exc:
            _logger.error("Gemini API 2-stage review error on clip {}: {}", clip_idx + 1, exc)

        return f"Phân cảnh {clip_idx + 1} mang lại những khoảnh khắc vô cùng ấn tượng."

    def _generate_for_clip_openai(
        self, frame_path: Optional[Path], clip_idx: int, total_clips: int, language: str = "vi", custom_instructions: str = ""
    ) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or OpenAI is None:
            return f"Phân cảnh {clip_idx + 1} làm hâm nóng bầu không khí với những diễn biến nghẹt thở."

        try:
            client = OpenAI(api_key=api_key)

            prompt_stage1 = (
                "CÔNG VIỆC 1: TRỢ LÝ QUAN SÁT VÀ MÔ TẢ VIDEO/HÌNH ẢNH\n"
                "Hãy mô tả chi tiết: Hành động chính, Nhân vật xuất hiện, Biểu cảm ngôn ngữ cơ thể và Mối tương tác.\n"
                "Chỉ mô tả khách quan những gì nhìn thấy trong hình ảnh. Không nhân hóa, không suy đoán cảm xúc."
            )
            content_stage1: List[dict] = [{"type": "text", "text": prompt_stage1}]
            if frame_path and frame_path.exists():
                with open(frame_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content_stage1.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

            resp1 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": content_stage1}],
            )
            stage1_desc = resp1.choices[0].message.content.strip()

            is_english = (
                language.lower() in ("en", "english")
                or "tiếng anh" in custom_instructions.lower()
                or "english" in custom_instructions.lower()
            )
            lang_instruction = (
                "Viết lời bình bằng TIẾNG ANH (Natural American English)."
                if is_english
                else "Viết lời bình bằng TIẾNG VIỆT tự nhiên, gần gũi."
            )
            prompt_stage2 = (
                "CÔNG VIỆC 2: TRỢ LÝ VIẾT CONTENT NGẮN CHUYÊN NGHIỆP\n"
                "Nhiệm vụ: Biến bản mô tả từ công việc 1 thành 1-2 câu LỜI BÌNH THÚ VỊ (10-25 từ).\n"
                f"BẢN MÔ TẢ:\n{stage1_desc}\n\n"
                f"CẢM XÚC: Warm humor (Hài hước ấm áp) hoặc Adorable curiosity (Tò mò đáng yêu).\n"
                f"{lang_instruction}\n"
                "QUY TẮC: Tạo góc nhìn mới (nhân hóa tự nhiên, liên tưởng đời thường). 1-2 câu ngắn (10-25 từ). CHỈ xuất ra duy nhất câu lời bình, không tiêu đề hay giải thích.\n"
                f"{custom_instructions}"
            )
            resp2 = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_stage2}],
            )
            txt = resp2.choices[0].message.content.strip().strip('"').strip("'")
            if txt:
                return txt
        except Exception as exc:
            _logger.error("OpenAI 2-stage review error on clip {}: {}", clip_idx + 1, exc)

        return f"Phân cảnh {clip_idx + 1} tạo điểm nhấn kịch tính cho toàn bộ tập phim."

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
        clips_list = getattr(video_info, "clips", []) if video_info else []

        if video_info and getattr(video_info, "is_multi_clip", False) and len(video_info.clips) > 0:
            total_clips = len(video_info.clips)
        else:
            events = timeline_result.timeline.events
            total_clips = len(events)
            if total_clips == 0:
                total_clips = 1

        _logger.info("Generating per-clip review script for {} clips...", total_clips)

        clip_scripts: List[str] = []
        frames_dir = Path("data") / "jobs" / job_id / "frames"
        if not frames_dir.exists():
            all_job_dirs = [p for p in Path("data").glob("jobs/*/frames") if p.is_dir()]
            if all_job_dirs:
                frames_dir = sorted(all_job_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                _logger.info("Using frames directory: {}", frames_dir)

        # Determine AI backend mode
        use_web_kw = kwargs.get("use_gemini_web")
        if use_web_kw is True:
            use_gemini_web = bool(self._browser_mgr)
            use_openai = False
            use_gemini_api = False
        elif use_web_kw is False:
            use_gemini_web = False
            use_openai = bool(self._openai_provider and os.getenv("OPENAI_API_KEY"))
            use_gemini_api = bool(not use_openai and os.getenv("GEMINI_API_KEY"))
        elif self._browser_mgr and not self._openai_provider:
            use_gemini_web = True
            use_openai = False
            use_gemini_api = False
        else:
            use_openai = bool(self._openai_provider and os.getenv("OPENAI_API_KEY"))
            use_gemini_api = bool(os.getenv("GEMINI_API_KEY"))
            use_gemini_web = bool(self._browser_mgr and not use_openai and not use_gemini_api)

        video_info = kwargs.get("video_info")
        clips_list = getattr(video_info, "clips", []) if video_info else []

        # ── GEMINI WEB: Setup Tab 1 + Tab 2 ONCE before processing clips ──
        if use_gemini_web:
            self._setup_gemini_web_tabs(total_clips, progress_cb)

        # ── Process each clip ──
        for i in range(total_clips):
            clip_num = i + 1
            if progress_cb:
                pct = 0.80 + (i / total_clips) * 0.18  # range 80%→98%
                progress_cb(f"Clip {clip_num}/{total_clips}", pct)

            clip_video_path = clips_list[i] if i < len(clips_list) else getattr(video_info, "video_path", None)

            frame_path = frames_dir / f"scene{i:04d}_frame00.jpg"
            if not frame_path.exists():
                possible = list(frames_dir.glob(f"scene{i:04d}_frame*.jpg"))
                frame_path = possible[0] if possible else None

            _logger.info("Processing review for clip {}/{} (Video: {})...", clip_num, total_clips, clip_video_path)

            if use_openai:
                script_part = self._generate_for_clip_openai(frame_path, i, total_clips, language=language, custom_instructions=custom_instructions)
            elif use_gemini_api:
                script_part = self._generate_for_clip_gemini_api(frame_path, i, total_clips, language=language, custom_instructions=custom_instructions)
            elif use_gemini_web:
                script_part = self._generate_for_clip_gemini_web(
                    i,
                    total_clips,
                    language=language,
                    custom_instructions=custom_instructions,
                    job_id=job_id,
                    clip_video_path=clip_video_path,
                    progress_cb=progress_cb,
                )
            else:
                script_part = self._generate_for_clip_gemini_api(frame_path, i, total_clips, language=language, custom_instructions=custom_instructions)

            clip_scripts.append(script_part)
            _logger.info("Clip {}/{} text: '{}'", clip_num, total_clips, script_part)

            # ── Instantly push THIS clip's content to Google Sheet Webhook ──
            try:
                import re
                stt_val = str(clip_num)
                if clip_video_path:
                    digits = re.findall(r'\d+', Path(clip_video_path).stem)
                    if digits:
                        stt_val = digits[-1]
                
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
