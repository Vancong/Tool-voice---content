# -*- coding: utf-8 -*-
"""
src/agents/sample_styles.py

Predefined few-shot sample writing styles (Văn phong mẫu) for the Review Writer Agent
and ChatGPT (OpenAI) / Gemini.
"""

from __future__ import annotations

from typing import Dict, Optional

# Sample few-shot review excerpts illustrating exact tone, cadence, and vocabulary
SAMPLE_STYLE_FUNNY_TRIEU_VIEW = """
[MẪU VĂN PHONG REVIEW TRIỆU VIEW - HÀI HƯỚC, DÍ DỎM, CHÂM BIẾM]
"Chào mừng toàn thể anh em đã quay trở lại với kênh! Hôm nay tôi sẽ dẫn anh em đến với một siêu phẩm điện ảnh mà xem xong chỉ muốn gọi ngay cho đạo diễn để xin một lời giải thích. 
Mở đầu bộ phim, chúng ta gặp ngay nam chính của chúng ta - một anh chàng số nhọ đến mức chim bay ngang qua cũng phải chọn đầu anh ấy làm điểm hạ cánh. Đang yên đang lành thì bỗng đâu một biến cố từ trên trời rơi xuống..."
"""

SAMPLE_STYLE_KICH_TINH = """
[MẪU VĂN PHONG KỊCH TÍNH - DỒN DẬP, HỒI HỘP, CUỐN HÚT TỪ TỪNG GIÂY]
"Đừng rời mắt khỏi màn hình, bởi vì 10 giây tiếp theo sẽ thay đổi hoàn toàn cục diện của câu chuyện! 
Trong căn phòng tối tăm không một bóng người, tiếng thở gấp gáp là thứ âm thanh duy nhất còn sót lại. Khi cánh cửa sắt từ từ hé mở, một sự thật kinh hoàng mà không ai trong số họ lường trước được chuẩn bị lộ diện..."
"""

SAMPLE_STYLE_TAM_LY_SAU_SAC = """
[MẪU VĂN PHONG PHÂN TÍCH TÂM LÝ & Ý NGHĨA CÁI KẾT]
"Ẩn sau lớp vỏ của một bộ phim hành động kịch tính là những giằng xé nội tâm sâu sắc về bản ngã con người. 
Ánh mắt của nhân vật chính ở phân cảnh này không đơn thuần là sự sợ hãi, mà đó là khoảnh khắc của sự vỡ vụn khi niềm tin duy nhất bị phản bội. Cái kết của bộ phim không cho chúng ta một câu trả lời dễ dãi, mà để lại một khoảng lặng đầy ám ảnh..."
"""

SAMPLE_STYLE_THUYET_MINH_CHUAN = """
[MẪU VĂN PHONG THUYẾT MINH CHUẨN MỰC - MẠCH LẠC, TRUYỀN CẢM, LÔI CUỐN]
"Bộ phim mở ra bối cảnh thành phố vào những năm cuối thập niên 90, nơi nhịp sống ồn ào che giấu những góc khuất tăm tối. 
Nhân vật chính của chúng ta bước vào một hành trình không có đường lui khi quyết định dấn thân vào vụ án này. Từng manh mối dần hé lộ, đưa người xem đi từ bất ngờ này sang bất ngờ khác..."
"""

BUILTIN_SAMPLE_STYLES: Dict[str, Dict[str, str]] = {
    "trieu_view": {
        "name": "🎭 Review Triệu View (Hài hước, Dí dỏm, Châm biếm)",
        "sample": SAMPLE_STYLE_FUNNY_TRIEU_VIEW,
        "instructions": (
            "Viết theo văn phong Review Phim Triệu View: Xưng hô 'anh em' thân thiện, tếu táo, châm biếm "
            "nhẹ nhàng các tình huống trớ trêu nhưng không xuyên tạc nội dung chính, tạo tiếng cười và giữ chân người xem."
        ),
    },
    "kich_tinh": {
        "name": "🔥 Kịch Tính & Giật Gân (Shorts / TikTok / Hành động)",
        "sample": SAMPLE_STYLE_KICH_TINH,
        "instructions": (
            "Viết theo văn phong Kịch tính dồn dập: Nhịp điệu nhanh, sử dụng các câu cảm thán, đặt câu hỏi gợi mở, "
            "nhấn mạnh vào sự nguy hiểm và những bất ngờ dồn dập khiến người xem không thể rời mắt."
        ),
    },
    "tam_ly": {
        "name": "🧠 Phân Tích Tâm Lý & Cái Kết (Sâu sắc, Triết lý)",
        "sample": SAMPLE_STYLE_TAM_LY_SAU_SAC,
        "instructions": (
            "Viết theo văn phong Phân tích điện ảnh: Đi sâu vào động cơ nhân vật, phân tích biểu cảm, ẩn dụ hình ảnh "
            "và diễn giải tầng ý nghĩa sâu xa đằng sau hành động và cái kết của bộ phim."
        ),
    },
    "thuyet_minh": {
        "name": "📖 Thuyết Minh Chuẩn Mực (Tài liệu, Truyền cảm)",
        "sample": SAMPLE_STYLE_THUYET_MINH_CHUAN,
        "instructions": (
            "Viết theo văn phong Thuyết minh điện ảnh chuẩn mực: Giọng điệu tự nhiên, truyền cảm, từ ngữ trau chuốt, "
            "kể lại toàn bộ diễn biến mạch lạc, tôn trọng nguyên tác câu chuyện."
        ),
    },
}

SAMPLE_STYLES = BUILTIN_SAMPLE_STYLES


def get_sample_style_context(style_key: Optional[str] = None, custom_sample: Optional[str] = None) -> str:
    """Build a prompt context block with few-shot sample style instructions."""
    if custom_sample and custom_sample.strip():
        return (
            "\n=========================================================\n"
            "VĂN PHONG MẪU CỦA NGƯỜI DÙNG (BẮT BUỘC BẮT CHƯỚC TONE & GIỌNG ĐIỆU DƯỚI ĐÂY):\n"
            "=========================================================\n"
            f"{custom_sample.strip()}\n"
            "---------------------------------------------------------\n"
            "HÃY BẮT CHƯỚC CHÍNH XÁC cách xưng hô, nhịp điệu, phong cách dùng từ và độ cuốn hút của đoạn mẫu trên "
            "khi viết lại kịch bản review cho video này.\n"
            "=========================================================\n"
        )

    if style_key and style_key in BUILTIN_SAMPLE_STYLES:
        entry = BUILTIN_SAMPLE_STYLES[style_key]
        return (
            "\n=========================================================\n"
            f"HƯỚNG DẪN VĂN PHONG: {entry['name']}\n"
            "=========================================================\n"
            f"{entry['instructions']}\n\n"
            "ĐOẠN VĂN MẪU THAM KHẢO:\n"
            f"{entry['sample']}\n"
            "---------------------------------------------------------\n"
            "HÃY BẮT CHƯỚC CHÍNH XÁC tone giọng, nhịp điệu và phong cách kể chuyện như đoạn mẫu trên.\n"
            "=========================================================\n"
        )

    return ""
