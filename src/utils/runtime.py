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


def get_ffmpeg_path() -> str:
    """Trả về đường dẫn tới ffmpeg.exe (bundled hoặc system PATH)."""
    app_dir = get_app_dir()
    bundled = app_dir / "ffmpeg" / "ffmpeg.exe"
    if bundled.exists():
        return str(bundled)
    return "ffmpeg"  # fallback: dùng system PATH


def get_ffprobe_path() -> str:
    """Trả về đường dẫn tới ffprobe.exe (bundled hoặc system PATH)."""
    app_dir = get_app_dir()
    bundled = app_dir / "ffmpeg" / "ffprobe.exe"
    if bundled.exists():
        return str(bundled)
    return "ffprobe"  # fallback: dùng system PATH


def setup_playwright_env() -> None:
    """Cấu hình PLAYWRIGHT_BROWSERS_PATH nếu chạy từ bundle."""
    if not getattr(sys, "frozen", False):
        return  # dev mode, không cần làm gì

    app_dir = get_app_dir()
    browsers_dir = app_dir / "playwright_browsers"
    if browsers_dir.exists():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers_dir)
