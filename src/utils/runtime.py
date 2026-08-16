# -*- coding: utf-8 -*-
"""
src/utils/runtime.py

Helper để detect môi trường chạy: PyInstaller bundle hay dev mode.
Cung cấp đường dẫn chính xác tới ffmpeg, ffprobe, playwright browser.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def get_app_dir() -> Path:
    """Trả về thư mục gốc của ứng dụng (dù đang chạy dev hay exe)."""
    if getattr(sys, "frozen", False):
        # Đang chạy từ PyInstaller bundle
        return Path(sys.executable).parent
    else:
        # Đang chạy từ source
        return Path(__file__).parent.parent.parent


import shutil

def get_ffmpeg_path() -> str:
    """Trả về đường dẫn tới ffmpeg.exe (bundled hoặc system PATH)."""
    app_dir = get_app_dir()
    candidates = [
        app_dir / "ffmpeg" / "ffmpeg.exe",
        app_dir / "ffmpeg" / "ffmpeg-win64.exe",
        app_dir / "bin" / "ffmpeg.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    sys_path = shutil.which("ffmpeg")
    if sys_path:
        return sys_path

    known_paths = [
        Path(os.path.expanduser(r"~\AppData\Local\ms-playwright\ffmpeg-1011\ffmpeg-win64.exe")),
        Path(os.path.expanduser(r"~\AppData\Local\CapCut\Apps\9.2.0.3931\ffmpeg.exe")),
        Path(os.path.expanduser(r"~\AppData\Local\CapCut\Apps\9.1.0.3879\ffmpeg.exe")),
    ]
    for kp in known_paths:
        if kp.exists():
            return str(kp)

    return "ffmpeg"  # fallback: dùng system PATH


def get_ffprobe_path() -> str:
    """Trả về đường dẫn tới ffprobe.exe (bundled hoặc system PATH)."""
    app_dir = get_app_dir()
    candidates = [
        app_dir / "ffmpeg" / "ffprobe.exe",
        app_dir / "bin" / "ffprobe.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    sys_path = shutil.which("ffprobe")
    if sys_path:
        return sys_path

    # If ffprobe is not installed separately, ffmpeg can often serve or we fallback to ffmpeg/ffprobe
    return "ffprobe"  # fallback: dùng system PATH


def setup_playwright_env() -> None:
    """Cấu hình PLAYWRIGHT_BROWSERS_PATH nếu chạy từ bundle."""
    if not getattr(sys, "frozen", False):
        return  # dev mode, không cần làm gì

    app_dir = get_app_dir()
    browsers_dir = app_dir / "playwright_browsers"
    if browsers_dir.exists():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers_dir)
