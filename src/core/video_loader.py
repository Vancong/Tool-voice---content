import re
import cv2
import subprocess
from pathlib import Path
from typing import NamedTuple, List, Optional
from uuid import UUID

from ..utils.logger import get_logger
from ..utils.thread_pool import ThreadPoolManager

_logger = get_logger("video_loader")

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}


def _natural_sort_key(p: Path):
    """Sort key to order clip filenames naturally (e.g. clip_1, clip_2, clip_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", p.name)]


class VideoInfo(NamedTuple):
    video_path: Path
    duration_sec: float
    fps: float
    width: int
    height: int
    thumbnail_path: Path
    clips: List[Path] = []

    @property
    def duration(self) -> float:
        return self.duration_sec

    @property
    def path(self) -> Path:
        return self.video_path

    @property
    def is_multi_clip(self) -> bool:
        return len(self.clips) > 1


class VideoLoader:
    """Load video metadata and generate a thumbnail without blocking UI.
    Supports single video file or a folder of pre-cut video clips.
    All heavy work is dispatched to a ThreadPoolExecutor.
    """

    @staticmethod
    def _get_clips_from_path(path: Path) -> List[Path]:
        """Resolve a file or directory into a list of video clip paths."""
        if path.is_file():
            return [path]
        elif path.is_dir():
            clips = [
                p for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            clips.sort(key=_natural_sort_key)
            if not clips:
                raise FileNotFoundError(f"Không tìm thấy file video nào trong thư mục: {path}")
            return clips
        else:
            raise FileNotFoundError(f"Đường dẫn không tồn tại: {path}")

    @staticmethod
    def _probe(video_path: Path) -> VideoInfo:
        video_path = Path(video_path).resolve()
        clips = VideoLoader._get_clips_from_path(video_path)

        total_duration = 0.0
        fps = 30.0
        width = 1280
        height = 720

        # Probe each clip to aggregate total duration and verify validity
        for idx, clip in enumerate(clips):
            cap = cv2.VideoCapture(str(clip))
            if not cap.isOpened():
                _logger.warning("Không thể mở clip: {}", clip)
                continue
            clip_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            clip_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            clip_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            clip_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            clip_dur = clip_frames / clip_fps if clip_fps else 0.0
            cap.release()

            total_duration += clip_dur
            if idx == 0:
                fps = clip_fps
                width = clip_w
                height = clip_h

        # Generate thumbnail using ffmpeg from the first clip
        from src.utils.runtime import get_ffmpeg_path
        first_clip = clips[0]
        thumb_dir = video_path if video_path.is_dir() else video_path.parent
        thumb_path = thumb_dir / "thumbnail.jpg"
        ff_cmd = [
            get_ffmpeg_path(),
            "-y",
            "-i", str(first_clip),
            "-vf", "scale=320:180",
            "-vframes", "1",
            str(thumb_path),
        ]
        try:
            subprocess.run(ff_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception:
            pass

        return VideoInfo(
            video_path=video_path,
            duration_sec=total_duration,
            fps=fps,
            width=width,
            height=height,
            thumbnail_path=thumb_path,
            clips=clips,
        )

    @classmethod
    def load(cls, video_path: Path, job_id: str) -> "concurrent.futures.Future[VideoInfo]":
        """Submit a load job to the thread pool. Returns a Future.
        Args:
            video_path: Path to the video file or directory of pre-cut clips.
            job_id: UUID string identifying the current job (used for logging).
        """
        _logger.info("[Job {job}] Loading video source: {path}", job=job_id, path=video_path)
        pool = ThreadPoolManager.get_pool()
        future = pool.submit(cls._probe, video_path)
        return future

