"""
Script import Cookie cho Gemini Web.
Chạy: python import_cookie.py
"""
import sys
from pathlib import Path
from src.gemini_web.session_manager import SessionManager

def main():
    sm = SessionManager()
    print("==================================================")
    print("   CÔNG CỤ IMPORT COOKIE CHO GEMINI WEB")
    print("==================================================")
    print(f"File lưu session: {sm.session_path}")
    print("\nBạn có thể dán Cookie ở 1 trong các dạng sau:")
    print("  1. Dạng JSON từ extension (Cookie-Editor, EditThisCookie...)")
    print("  2. Dạng chuỗi Header Cookie (SID=...; HSID=...; __Secure-1PSID=...)")
    print("\nNhập/Dán nội dung Cookie bên dưới (Kết thúc bằng Enter 2 lần hoặc Ctrl+Z/Ctrl+D):")
    print("--------------------------------------------------")

    lines = []
    try:
        while True:
            line = input()
            if not line and lines and not lines[-1]:
                break
            lines.append(line)
    except EOFError:
        pass

    raw_cookie = "\n".join(lines).strip()
    if not raw_cookie:
        print("❌ Chưa nhập cookie nào!")
        return

    count = sm.import_cookies_from_raw_string(raw_cookie)
    if count > 0:
        print(f"\n✅ ĐÃ IMPORT THÀNH CÔNG {count} COOKIES!")
        print(f"File đã được lưu tại: {sm.session_path}")
        print("\nBây giờ bạn có thể khởi động main.py và chạy pipeline trực tiếp!")
    else:
        print("\n❌ Không thể phân tích chuỗi Cookie. Vui lòng kiểm tra lại định dạng!")

if __name__ == "__main__":
    main()
