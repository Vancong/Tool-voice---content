"""
Test script to directly replicate what send_prompt_to_stage does.
Run this while main.py is NOT running.
"""
import threading
import time
from src.gemini_web.browser_manager import BrowserManager

print(f"[TEST] Starting on thread: {threading.current_thread().name}")
print(f"[TEST] Browser executor: {BrowserManager._browser_executor}")

bm = BrowserManager()

print("[TEST] Calling get_stage_page(1) via _run_on_browser_thread...")
print(f"[TEST] Current owner_thread_id: {BrowserManager._owner_thread_id}")

try:
    def _task():
        print(f"[TEST-EXECUTOR] Running in thread: {threading.current_thread().name}")
        page = bm.get_stage_page(1)
        print(f"[TEST-EXECUTOR] Page URL: {page.url}")
        return page

    future = BrowserManager._browser_executor.submit(_task)
    page = future.result(timeout=30)
    print(f"[TEST] Got page: {page.url}")
    print("[TEST] SUCCESS - browser opened!")
    time.sleep(5)
    bm.close()
except Exception as e:
    print(f"[TEST] FAILED: {e}")
    import traceback
    traceback.print_exc()
