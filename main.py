# -*- coding: utf-8 -*-
"""
main.py

Application entry point.

With no arguments  → launches the CustomTkinter desktop UI.
With --input flag  → runs the CLI pipeline (headless mode).

Usage:
    python main.py                          # open UI
    python main.py --input movie.mp4        # CLI headless
    python main.py --input movie.mp4 --output out.mp4
"""

from __future__ import annotations

import sys

# ── Khi chạy từ PyInstaller bundle, phải setup Playwright env TRƯỚC mọi import ──
from src.utils.runtime import setup_playwright_env
setup_playwright_env()

# Load .env before any project import so CONFIG is fully populated.
from dotenv import load_dotenv
load_dotenv(override=False)


def _run_cli() -> int:
    """CLI entry-point – used when --input is passed."""
    import argparse
    import threading
    import uuid
    from pathlib import Path

    from src.config.settings import CONFIG
    from src.utils.logger import get_logger
    from src.core.result import Result
    from src.core.video_loader import VideoLoader
    from src.core.workflow import WorkflowEngine
    from src.stt.providers.faster_whisper_provider import FasterWhisperProvider
    from src.scene.providers.pyscenedetect_provider import PySceneDetectProvider
    from src.frame.providers.opencv_frame_provider import OpenCVFrameProvider
    from src.vision.providers.gemini_vision_provider import GeminiVisionProvider
    from src.timeline.providers.timeline_builder import TimelineBuilderProvider
    from src.review.providers.gemini_review_provider import GeminiReviewProvider
    from src.gemini_web.gemini_web_provider import GeminiWebProvider
    from src.agents.multi_agent_provider import MultiAgentReviewProvider
    from src.tts.providers.capcut_tts_provider import CapCutTTSProvider
    from src.composer.providers.ffmpeg_video_composer import FFmpegVideoComposer

    _logger = get_logger("main")

    parser = argparse.ArgumentParser(
        description="AI Movie Review Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, metavar="MOVIE")
    parser.add_argument("--output", "-o", default=None, metavar="OUTPUT")
    parser.add_argument("--job-id", default=None, metavar="JOB_ID")
    parser.add_argument("--web", action="store_true", default=True, help="Use Gemini Web Playwright automation (default)")
    parser.add_argument("--api", action="store_true", help="Use Gemini Cloud API instead of Gemini Web")
    parser.add_argument("--instructions", "-p", default=None, help="Custom prompt or storytelling instructions for the review")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    job_id: str = args.job_id or str(uuid.uuid4())[:8]

    if not input_path.exists():
        print(f"FAILED – path not found: {input_path}", file=sys.stderr)
        return 2
    if input_path.is_file() and input_path.suffix.lower() not in CONFIG.video.supported_formats:
        print(f"FAILED – unsupported format '{input_path.suffix}'", file=sys.stderr)
        return 2

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_dir = Path(CONFIG.composer.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}.mp4"

    use_web = not args.api
    web_prov = GeminiWebProvider(config=CONFIG) if use_web else None
    review_gen = MultiAgentReviewProvider(
        browser_mgr=web_prov.browser_manager if web_prov else None,
        quality_threshold=0.70,
    )

    def _progress(stage: str, pct: float) -> None:
        bar = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))
        print(f"\r  [{bar}] {pct * 100:5.1f}%  {stage:<24}", end="", flush=True)
        if pct >= 1.0:
            print()

    cancel_token = threading.Event()
    engine = WorkflowEngine(
        video_loader=VideoLoader(),
        stt=FasterWhisperProvider(config=CONFIG),
        scene_detector=PySceneDetectProvider(config=CONFIG),
        frame_extractor=OpenCVFrameProvider(config=CONFIG),
        vision_analyzer=GeminiVisionProvider(config=CONFIG),
        timeline_builder=TimelineBuilderProvider(config=CONFIG),
        review_generator=review_gen,
        tts=CapCutTTSProvider(config=CONFIG),
        video_composer=FFmpegVideoComposer(config=CONFIG),
    )

    print(f"\n  Job ID : {job_id}\n  Input  : {input_path}\n  Output : {output_path}\n")
    try:
        result: Result = engine.run(
            video_path=input_path,
            output_path=output_path,
            job_id=job_id,
            progress_callback=_progress,
            cancel_token=cancel_token,
            debug_mode=args.debug,
            custom_instructions=args.instructions,
        )
    except KeyboardInterrupt:
        cancel_token.set()
        print("\nINTERRUPTED", file=sys.stderr)
        return 1
    except Exception as exc:
        _logger.exception("Unexpected error")
        print(f"\nFAILED – {exc}", file=sys.stderr)
        return 3

    if result.is_ok:
        print(f"\nSUCCESS – {result.unwrap()}")
        return 0
    else:
        print(f"\nFAILED – {result.error}", file=sys.stderr)
        return 1


def _run_ui() -> None:
    """UI entry-point – default when no --input is given."""
    from src.ui.main_window import run_ui
    run_ui()


def main() -> int:
    # If --input is anywhere in argv, run CLI mode.
    if "--input" in sys.argv or "-i" in sys.argv:
        return _run_cli()

    # Otherwise open the desktop UI.
    _run_ui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
