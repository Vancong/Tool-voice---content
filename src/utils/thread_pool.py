import concurrent.futures
from typing import Optional
from pathlib import Path
from .logger import get_logger

_logger = get_logger("thread_pool")

class ThreadPoolManager:
    _instance: Optional[concurrent.futures.ThreadPoolExecutor] = None

    @classmethod
    def get_pool(cls, max_workers: int = 8) -> concurrent.futures.ThreadPoolExecutor:
        if cls._instance is None:
            _logger.debug("Creating ThreadPoolExecutor with max_workers={}", max_workers)
            cls._instance = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        return cls._instance

    @classmethod
    def shutdown(cls, wait: bool = True) -> None:
        if cls._instance:
            _logger.debug("Shutting down ThreadPoolExecutor")
            cls._instance.shutdown(wait=wait)
            cls._instance = None
