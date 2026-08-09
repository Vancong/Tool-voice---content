import sys, pathlib, importlib
project_root = pathlib.Path('e:/tool-yt/rv-phim')
sys.path.append(str(project_root))
modules = [
    'src.utils.logger',
    'src.utils.thread_pool',
    'src.core.video_loader',
    'src.stt.providers.faster_whisper_provider',
    'src.scene.providers.pyscenedetect_provider',
    'src.frame.providers.opencv_frame_provider',
    'src.vision.providers.gemini_vision_provider',
    'src.timeline.providers.timeline_builder',
    'src.review.providers.gemini_review_provider',
    'src.tts.providers.capcut_tts_provider',
    'src.composer.providers.ffmpeg_video_composer',
    'src.core.workflow',
    'main'
]
for m in modules:
    try:
        importlib.import_module(m)
        print(f'OK {m}')
    except Exception as e:
        print(f'FAIL {m}: {e.__class__.__name__}: {e}')
