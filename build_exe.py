"""build_exe.py - Build AI Movie Review Studio thanh goi exe khep kin."""
from __future__ import annotations
import io, os, shutil, subprocess, sys
from pathlib import Path

# Fix Windows terminal encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.resolve()
DIST_DIR = ROOT / "dist" / "AI-Movie-Review-Studio"
SPEC_FILE = ROOT / "AI-Movie-Review-Studio.spec"
PLAYWRIGHT_BASE = Path(os.environ.get(
    "PLAYWRIGHT_BROWSERS_PATH",
    Path.home() / "AppData" / "Local" / "ms-playwright"
))
FFMPEG_SRC = shutil.which("ffmpeg")
FFPROBE_SRC = shutil.which("ffprobe")


def p(msg: str) -> None:
    print(f"  {msg}", flush=True)


def hr(title: str) -> None:
    print(f"\n--- {title} ---", flush=True)


def check_prereqs() -> bool:
    hr("Kiem tra prerequisites")
    ok = True
    try:
        import PyInstaller
        p(f"[OK] PyInstaller {PyInstaller.__version__}")
    except ImportError:
        p("[FAIL] PyInstaller chua cai! Chay: pip install pyinstaller")
        ok = False

    if FFMPEG_SRC:
        p(f"[OK] ffmpeg: {FFMPEG_SRC}")
    else:
        local = ROOT / "ffmpeg" / "ffmpeg.exe"
        if local.exists():
            p(f"[OK] ffmpeg local: {local}")
        else:
            p("[WARN] ffmpeg khong tim thay trong PATH hoac thu muc ffmpeg/")

    chromium_dirs = list(PLAYWRIGHT_BASE.glob("chromium-*")) if PLAYWRIGHT_BASE.exists() else []
    if chromium_dirs:
        p(f"[OK] Playwright Chromium: {chromium_dirs[0].name}")
    else:
        p(f"[FAIL] Playwright Chromium khong tim thay tai: {PLAYWRIGHT_BASE}")
        p("       Chay: playwright install chromium")
        ok = False

    return ok


def run_pyinstaller() -> bool:
    hr("Chay PyInstaller (co the mat 3-5 phut...)")
    cmd = [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", str(SPEC_FILE)]
    p(f"CMD: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        p("[FAIL] PyInstaller that bai!")
        return False
    p("[OK] PyInstaller hoan thanh")
    return True


def copy_ffmpeg() -> None:
    hr("Copy FFmpeg vao dist")
    dest = DIST_DIR / "ffmpeg"
    dest.mkdir(exist_ok=True)

    for name, src_path in [("ffmpeg.exe", FFMPEG_SRC), ("ffprobe.exe", FFPROBE_SRC)]:
        local = ROOT / "ffmpeg" / name
        final = src_path or (str(local) if local.exists() else None)
        if final:
            shutil.copy2(final, dest / name)
            p(f"[OK] Copy {name}")
        else:
            p(f"[WARN] {name} khong tim thay - khach se can cai FFmpeg thu cong")


def copy_playwright_browsers() -> None:
    hr("Copy Playwright Chromium vao dist (~150MB)")
    if not PLAYWRIGHT_BASE.exists():
        p(f"[FAIL] Khong tim thay: {PLAYWRIGHT_BASE}")
        return

    chromium_dirs = sorted(PLAYWRIGHT_BASE.glob("chromium-*"), reverse=True)
    if not chromium_dirs:
        p("[FAIL] Khong tim thay thu muc chromium-*")
        return

    src = chromium_dirs[0]
    dest = DIST_DIR / "playwright_browsers" / src.name
    if dest.exists():
        p(f"[SKIP] Chromium da co: {src.name}")
    else:
        p(f"Dang copy {src.name}...")
        shutil.copytree(src, dest)
        p(f"[OK] Da copy Chromium: {src.name}")


def copy_env() -> None:
    hr("Tao file .env cho khach")
    dest = DIST_DIR / ".env"
    if dest.exists():
        p("[SKIP] .env da ton tai")
        return
    dest.write_text(
        "# AI Movie Review Studio - Cau hinh\n"
        "GOOGLE_SHEET_WEBHOOK_URL=\n"
        "ELEVENLABS_API_KEY=\n"
        "CAPCUT_SESSION_ID=\n",
        encoding="utf-8"
    )
    p("[OK] Da tao .env mac dinh")


def create_zip() -> None:
    hr("Nen thanh .zip phan phoi")
    output = ROOT / "dist" / "AI-Movie-Review-Studio-Release"
    if (ROOT / "dist" / "AI-Movie-Review-Studio-Release.zip").exists():
        (ROOT / "dist" / "AI-Movie-Review-Studio-Release.zip").unlink()
    p("Dang nen... (co the mat vai phut)")
    shutil.make_archive(str(output), "zip", root_dir=ROOT / "dist", base_dir="AI-Movie-Review-Studio")
    size = (ROOT / "dist" / "AI-Movie-Review-Studio-Release.zip").stat().st_size / 1024 / 1024
    p(f"[OK] Da tao: AI-Movie-Review-Studio-Release.zip ({size:.0f} MB)")


def main() -> int:
    print("\n" + "="*55)
    print("  AI Movie Review Studio - Build EXE")
    print("="*55)

    if not check_prereqs():
        print("\n[FAIL] Prerequisites chua du. Vui long cai dat roi chay lai.")
        return 1

    if not run_pyinstaller():
        return 1

    copy_ffmpeg()
    copy_playwright_browsers()
    copy_env()

    do_zip = input("\n  Tao file .zip de phan phoi? [Y/n]: ").strip().lower()
    if do_zip in ("", "y", "yes"):
        create_zip()

    print("\n" + "="*55)
    print("  BUILD XONG!")
    print(f"  Thu muc: {DIST_DIR}")
    print("  Gui toan bo thu muc (hoac .zip) cho khach")
    print("="*55 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
