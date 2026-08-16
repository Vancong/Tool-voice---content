# -*- mode: python ; coding: utf-8 -*-
"""
AI Movie Review Studio – PyInstaller spec file.

Build command:
    python build_exe.py

Output:
    dist/
    └── AI-Movie-Review-Studio/
        ├── AI-Movie-Review-Studio.exe
        ├── _internal/         (Python + libs)
        ├── ffmpeg/
        │   ├── ffmpeg.exe
        │   └── ffprobe.exe
        ├── playwright_browsers/  (Chromium)
        └── config/
            └── config.json
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# ─── Collect data files ───────────────────────────────────────────────────────

# CustomTkinter themes & assets
ctk_datas, ctk_binaries, ctk_hiddenimports = collect_all("customtkinter")

# Playwright driver files
pw_datas, pw_binaries, pw_hiddenimports = collect_all("playwright")

# faster-whisper & ctranslate2 & huggingface_hub
fw_datas, fw_binaries, fw_hiddenimports = collect_all("faster_whisper")
ct_datas, ct_binaries, ct_hiddenimports = collect_all("ctranslate2")
hf_datas, hf_binaries, hf_hiddenimports = collect_all("huggingface_hub")

# pydantic-settings
pydantic_datas = collect_data_files("pydantic_settings")

# project config
project_datas = [
    ("config", "config"),
]

all_datas = (
    ctk_datas
    + pw_datas
    + fw_datas
    + ct_datas
    + hf_datas
    + pydantic_datas
    + project_datas
)
all_binaries = ctk_binaries + pw_binaries + fw_binaries + ct_binaries + hf_binaries
all_hiddenimports = list(set(
    ctk_hiddenimports
    + pw_hiddenimports
    + fw_hiddenimports
    + ct_hiddenimports
    + hf_hiddenimports
    + [
        "playwright",
        "playwright.sync_api",
        "playwright.async_api",
        "playwright._impl._driver",
        "customtkinter",
        "faster_whisper",
        "ctranslate2",
        "huggingface_hub",
        "PIL._tkinter_finder",
        "PIL.ImageTk",
        "tkinter",
        "tkinter.ttk",
        "edge_tts",
        "anyio",
        "anyio._backends._asyncio",
        "anyio._backends._trio",
        "loguru",
        "pydantic",
        "pydantic_settings",
        "dotenv",
        "requests",
        "cv2",
        "scenedetect",
        "src",
        "src.ui",
        "src.core",
        "src.config",
        "src.utils",
        "src.tts",
        "src.tts.providers",
        "src.tts.providers.capcut_tts_provider",
        "src.tts.providers.elevenlabs_tts_provider",
        "src.gemini_web",
        "src.agents",
        "src.review",
        "src.review.providers",
        "src.exporter",
        "src.composer",
        "src.composer.providers",
        "src.scene",
        "src.frame",
        "src.vision",
        "src.timeline",
        "src.stt",
    ]
))

# ─── Analysis ────────────────────────────────────────────────────────────────

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "matplotlib",
        "numpy.tests",
        "scipy",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "unittest",
        "test",
        "tests",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI-Movie-Review-Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # Ẩn console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,              # Có thể thêm icon .ico ở đây
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AI-Movie-Review-Studio",
)
