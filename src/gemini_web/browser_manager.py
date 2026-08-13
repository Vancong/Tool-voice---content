"""Browser manager for Gemini Web automation using Playwright.

Enterprise-Grade Features:
1. Session Watchdog (auto-detection of expired login & automatic recovery)
2. Persistent Browser Singleton (never creates multiple browser instances)
3. FIFO Concurrency Queue / Lock (serializes prompt executions)
4. Automatic DOM Snapshot (captures page.html, page.png, console.log, network.log on failure)
5. Semantic DOM Auto-Discovery (recovers from unexpected UI changes via JS evaluation)
6. Conversation Cleanup (starts fresh chat before each generation to prevent context leakage)
7. Fine-grained Progress Reporting (reports sub-stage statuses in real-time)
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Any

from playwright.sync_api import (
    sync_playwright,
    Playwright,
    BrowserContext,
    Page,
    Locator,
    TimeoutError as PlaywrightTimeoutError,
)

from src.gemini_web.exceptions import (
    GeminiWebAuthError,
    GeminiWebDOMError,
    GeminiWebGenerationError,
    GeminiWebNavigationError,
    GeminiWebTimeoutError,
)
from src.gemini_web.models import GeminiWebConfig, GeminiWebResponse, SessionStatus
from src.gemini_web.session_manager import SessionManager
from src.utils.logger import get_logger


# Selectors for Gemini Web UI
CHAT_INPUT_SELECTORS = [
    "rich-textarea div[contenteditable='true']",
    "div[contenteditable='true'][role='textbox']",
    "div[contenteditable='true']",
    "rich-textarea p",
    "p[data-placeholder]",
    "div[role='textbox']",
    "div[aria-label*='Prompt' i]",
    "div[aria-label*='prompt' i]",
    "div[aria-label*='Hỏi' i]",
    "div[aria-label*='Nhập' i]",
    "textarea[placeholder*='prompt' i]",
    "textarea[placeholder*='Enter' i]",
    "textarea",
    ".ql-editor",
]

SEND_BUTTON_SELECTORS = [
    "button[aria-label*='Send prompt' i]",
    "button[aria-label*='Send message' i]",
    "button[aria-label*='Send' i]",
    "button[aria-label*='Gửi câu hỏi' i]",
    "button[aria-label*='Gửi tin nhắn' i]",
    "button[aria-label*='Gửi' i]",
    "button.send-button",
    "button[data-test-id='send-button']",
    "button:has(mat-icon[fonticon='send'])",
    "button:has(svg[data-icon='send'])",
    "button:has(.send-icon)",
    "button.mat-mdc-button-base:has(mat-icon)",
    "button:has-text('Send')",
    "button:has-text('Gửi')",
]

STOP_BUTTON_SELECTORS = [
    "button[aria-label*='Stop generating' i]",
    "button[aria-label*='Stop response' i]",
    "button[aria-label*='Stop' i]",
    "button[aria-label*='Dừng tạo' i]",
    "button[aria-label*='Dừng phản hồi' i]",
    "button[aria-label*='Dừng' i]",
    "button:has(mat-icon[fonticon='stop'])",
    "button:has(svg[aria-label*='stop' i])",
    "button[data-test-id='stop-button']",
    ".stop-generating-button",
]

RESPONSE_CONTAINER_SELECTORS = [
    "model-response",
    "message-content",
    ".model-response-text",
    ".response-container",
    "div[data-test-id='model-response']",
    "div.markdown",
    "div.response-text",
    "structured-response",
    "message-item .model-response",
    "div[role='region'][aria-label*='Response' i]",
    "div[role='region'][aria-label*='Phản hồi' i]",
]

NEW_CHAT_SELECTORS = [
    "a[href*='/app']",
    "button[aria-label*='New chat' i]",
    "button[aria-label*='Cuộc trò chuyện mới' i]",
    "div[aria-label*='New chat' i]",
    "button:has-text('New chat')",
    "button:has-text('Cuộc trò chuyện mới')",
    "button:has(mat-icon[fonticon='add'])",
    "button:has(svg[data-icon='plus'])",
]

# Selectors for ChatGPT Web UI
CHATGPT_INPUT_SELECTORS = [
    "#prompt-textarea",
    "div[contenteditable='true'][id='prompt-textarea']",
    "div[contenteditable='true']",
    "textarea#prompt-textarea",
    "textarea[data-id]",
    "textarea",
]

CHATGPT_SEND_SELECTORS = [
    "button[data-testid='send-button']",
    "button[aria-label*='Send' i]",
    "button[aria-label*='Gửi' i]",
    "button:has(svg[data-icon='arrow-up'])",
    "button.mb-1",
]

CHATGPT_STOP_SELECTORS = [
    "button[data-testid='stop-button']",
    "button[aria-label*='Stop' i]",
]

CHATGPT_RESPONSE_SELECTORS = [
    "div[data-message-author-role='assistant'] .markdown",
    "div[data-message-author-role='assistant']",
    "div.article-content",
    "div.markdown",
]

CHATGPT_NEW_CHAT_SELECTORS = [
    "a[href='/']",
    "button:has-text('New chat')",
    "a:has-text('New chat')",
    "button[aria-label*='New chat' i]",
]

# Selectors for Claude Web UI (claude.ai)
CLAUDE_INPUT_SELECTORS = [
    "div[contenteditable='true']",
    "p[data-placeholder]",
    "fieldset div[contenteditable='true']",
    "div.ProseMirror",
    "div[role='textbox']",
    "textarea",
]

CLAUDE_SEND_SELECTORS = [
    "button[aria-label*='Send' i]",
    "button[aria-label*='Gửi' i]",
    "button:has(svg)",
    "button.bg-accent-main-100",
    "button[type='submit']",
]

CLAUDE_STOP_SELECTORS = [
    "button[aria-label*='Stop' i]",
    "button[aria-label*='Dừng' i]",
]

CLAUDE_RESPONSE_SELECTORS = [
    "div.font-claude-message",
    "div.grid-cols-1 .markdown",
    "div[data-is-streaming='false']",
    "div.prose",
    "div.markdown",
]

CLAUDE_NEW_CHAT_SELECTORS = [
    "a[href='/new']",
    "button:has-text('New chat')",
    "button:has-text('Start new chat')",
    "a:has-text('Start new chat')",
    "button[aria-label*='New chat' i]",
]

AISTUDIO_INPUT_SELECTORS = [
    "textarea",
    "div[contenteditable='true']",
    "textarea[placeholder*='prompt' i]",
    "textarea[placeholder*='type' i]",
    "textarea[placeholder*='Insert' i]",
    ".mat-mdc-input-element",
    "div[role='textbox']",
]

AISTUDIO_RUN_SELECTORS = [
    "button:has-text('Run')",
    "button[aria-label*='Run' i]",
    "button:has(mat-icon[fonticon='play_arrow'])",
    "button:has(svg[data-icon='play'])",
    "button[aria-label*='Send' i]",
    "button:has-text('Send')",
    "button.run-button",
]


class BrowserManager:
    """Enterprise-grade manager for persistent Playwright browser automation."""

    # -------------------------------------------------------------
    # Class-level Persistent Browser Singleton & Lock
    # -------------------------------------------------------------
    _singleton_playwright: Optional[Playwright] = None
    _singleton_context: Optional[BrowserContext] = None
    _singleton_page: Optional[Page] = None
    _stage_pages: Dict[int, Page] = {}
    # Use RLock (reentrant) so same thread can re-acquire: get_stage_page calls _ensure_browser
    _singleton_lock: threading.RLock = threading.RLock()
    _job_queue_lock: threading.RLock = threading.RLock()  # FIFO serialization
    # Track which thread OWNS the Playwright process
    _owner_thread_id: Optional[int] = None
    # Dedicated single-worker thread to ensure ALL Playwright calls run on same thread
    _browser_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bw")

    # Log accumulators for DOM snapshots
    _console_logs: List[str] = []
    _network_logs: List[str] = []

    @classmethod
    def _on_browser_thread(cls, fn: Callable[..., Any], *args: Any) -> Any:
        """Run fn on the dedicated browser worker thread. If already on that thread, call directly."""
        t = threading.current_thread().name
        if t.startswith("bw"):
            # Already on browser thread — call directly to avoid recursive executor deadlock
            return fn(*args)
        f = cls._browser_executor.submit(fn, *args)
        return f.result()

    def __init__(
        self,
        config: Optional[GeminiWebConfig] = None,
        session_mgr: Optional[SessionManager] = None,
    ) -> None:
        self._config = config or GeminiWebConfig()
        self._session_mgr = session_mgr or SessionManager(self._config)
        self._logger = get_logger(name="gemini_browser").bind(module="gemini_web", component="browser")


    def _bring_chrome_to_front(self, page: Optional[Page] = None) -> None:
        """Bring page tab to front safely within the tool's Playwright browser context only."""
        if page:
            try:
                page.bring_to_front()
            except Exception:
                pass

    def _kill_orphan_chrome_processes(self) -> None:
        """Kill any background chrome process holding lock on our profile directory."""
        profile_path_str = str(Path(self._session_mgr.profile_dir).resolve()).lower()
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    name = (proc.info.get('name') or '').lower()
                    if 'chrome' in name or 'msedge' in name:
                        cmdline = ' '.join(proc.info.get('cmdline') or []).lower()
                        if 'gemini_session' in cmdline or profile_path_str in cmdline:
                            self._logger.info("Terminating orphan Chrome process (PID {}) holding profile lock...", proc.info['pid'])
                            proc.kill()
                except Exception:
                    pass
        except Exception as exc:
            self._logger.warning("Error checking orphan chrome processes: {}", exc)

    def _ensure_browser(self, headless: Optional[bool] = False) -> Page:
        """Launch or reuse the persistent browser singleton with VISIBLE GUI (headless=False).
        MUST be called from the dedicated browser thread (bw_*) only.
        """
        with BrowserManager._singleton_lock:
            # Check if singleton page is still active — just reuse it
            if (
                BrowserManager._singleton_page is not None
                and not BrowserManager._singleton_page.is_closed()
                and BrowserManager._singleton_context is not None
            ):
                try:
                    BrowserManager._singleton_page.bring_to_front()
                    self._bring_chrome_to_front(BrowserManager._singleton_page)
                except Exception:
                    pass
                return BrowserManager._singleton_page

            self._logger.info("Initializing Persistent Chromium Singleton (headless=False)...")

            # Ensure any background orphan chrome process holding our profile lock is terminated
            self._kill_orphan_chrome_processes()

            if BrowserManager._singleton_playwright is None:
                BrowserManager._singleton_playwright = sync_playwright().start()

            profile_dir_path = Path(self._session_mgr.profile_dir)
            profile_dir = str(profile_dir_path)
            
            # Clean up stale Chromium lock files if previous process crashed or left orphan locks
            for lock_name in ["SingletonLock", "SingletonCookie", "SingletonSocket"]:
                lock_file = profile_dir_path / lock_name
                if lock_file.exists():
                    try:
                        lock_file.unlink()
                    except Exception:
                        pass

            args = [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-infobars",
                "--start-maximized",
            ]

            launch_kwargs = {
                "user_data_dir": profile_dir,
                "headless": False,
                "args": args,
                "viewport": {"width": 1280, "height": 800},
                "user_agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
                "timeout": 20000,
            }

            try:
                # Try launching real installed Google Chrome first for maximum compatibility with Google login
                BrowserManager._singleton_context = BrowserManager._singleton_playwright.chromium.launch_persistent_context(
                    channel="chrome", **launch_kwargs
                )
            except Exception as exc_chrome:
                self._logger.info("Real Chrome launch fallback to Playwright Chromium: {}", exc_chrome)
                BrowserManager._singleton_context = BrowserManager._singleton_playwright.chromium.launch_persistent_context(
                    **launch_kwargs
                )

            # If session file exists, inject cookies into browser context
            if self._session_mgr.session_path.exists():
                try:
                    with open(self._session_mgr.session_path, "r", encoding="utf-8") as f:
                        sdata = json.load(f)
                    cookies_list = sdata.get("cookies", []) if isinstance(sdata, dict) else (sdata if isinstance(sdata, list) else [])
                    if cookies_list:
                        # Clean cookies so Playwright add_cookies won't reject non-standard keys
                        valid_keys = {"name", "value", "url", "domain", "path", "expires", "httpOnly", "secure", "sameSite"}
                        clean_cookies = []
                        for c in cookies_list:
                            if isinstance(c, dict) and "name" in c and "value" in c:
                                cleaned = {k: v for k, v in c.items() if k in valid_keys and v is not None}
                                clean_cookies.append(cleaned)
                        
                        # Add cookies in batches or individually to prevent 1 bad cookie from breaking all
                        success_cnt = 0
                        for ck in clean_cookies:
                            try:
                                BrowserManager._singleton_context.add_cookies([ck])
                                success_cnt += 1
                            except Exception:
                                pass
                        self._logger.info("Injected {}/{} valid cookies into browser context", success_cnt, len(clean_cookies))
                except Exception as exc_cookies:
                    self._logger.warning("Failed injecting session cookies: {}", exc_cookies)

            pages = BrowserManager._singleton_context.pages
            page = pages[0] if pages else BrowserManager._singleton_context.new_page()
            page.set_default_timeout(self._config.timeout_ms)
            page.set_default_navigation_timeout(self._config.navigation_timeout_ms)
            BrowserManager._singleton_page = page
            self._bring_chrome_to_front(page)

            # Listeners for DOM snapshot diagnostics
            BrowserManager._console_logs.clear()
            BrowserManager._network_logs.clear()

            def _on_console(msg):
                BrowserManager._console_logs.append(f"[{time.strftime('%X')}] [{msg.type}] {msg.text}")
                if len(BrowserManager._console_logs) > 500:
                    BrowserManager._console_logs.pop(0)

            def _on_request(req):
                BrowserManager._network_logs.append(f"[{time.strftime('%X')}] >> {req.method} {req.url}")
                if len(BrowserManager._network_logs) > 500:
                    BrowserManager._network_logs.pop(0)

            def _on_response(resp):
                BrowserManager._network_logs.append(f"[{time.strftime('%X')}] << {resp.status} {resp.url}")
                if len(BrowserManager._network_logs) > 500:
                    BrowserManager._network_logs.pop(0)

            try:
                page.on("console", _on_console)
                page.on("request", _on_request)
                page.on("response", _on_response)
            except Exception:
                pass

            # Inject stealth scripts to avoid bot detection
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            BrowserManager._singleton_page = page
            return BrowserManager._singleton_page

    # -------------------------------------------------------------
    # DOM Snapshot Diagnostics
    # -------------------------------------------------------------
    def save_dom_snapshot(self, page: Page, reason: str, job_id: Optional[str] = None) -> Path:
        """Capture page HTML, screenshot, console and network logs for troubleshooting."""
        return self._on_browser_thread(self._save_dom_snapshot_impl, page, reason, job_id)

    def _save_dom_snapshot_impl(self, page: Page, reason: str, job_id: Optional[str] = None) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = Path("logs") / "dom_snapshot" / f"{timestamp}_{job_id or 'gemini'}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._logger.warning("Saving DOM snapshot to: {} (Reason: {})", snapshot_dir, reason)

        # 1. Save HTML
        try:
            html_content = page.content()
            (snapshot_dir / "page.html").write_text(html_content, encoding="utf-8")
        except Exception as exc:
            self._logger.warning("Failed saving page.html: {}", exc)

        # 2. Save Screenshot
        try:
            page.screenshot(path=str(snapshot_dir / "page.png"), full_page=False)
        except Exception as exc:
            self._logger.warning("Failed saving page.png: {}", exc)

        # 3. Save Console Log
        try:
            (snapshot_dir / "console.log").write_text("\n".join(BrowserManager._console_logs), encoding="utf-8")
        except Exception as exc:
            self._logger.warning("Failed saving console.log: {}", exc)

        # 4. Save Network Log
        try:
            (snapshot_dir / "network.log").write_text("\n".join(BrowserManager._network_logs), encoding="utf-8")
        except Exception as exc:
            self._logger.warning("Failed saving network.log: {}", exc)

        # 5. Metadata
        try:
            meta = {
                "timestamp": timestamp,
                "reason": reason,
                "url": page.url,
                "job_id": job_id,
            }
            (snapshot_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

        return snapshot_dir

    # -------------------------------------------------------------
    # Semantic DOM Auto-Discovery
    # -------------------------------------------------------------
    def _auto_discover_element(self, page: Page, target_type: str) -> Tuple[Optional[Locator], Optional[str]]:
        """Automatically inspect page DOM using JavaScript heuristics to discover missing elements."""
        self._logger.info("Executing Semantic DOM Auto-Discovery for '{}'...", target_type)
        try:
            if target_type == "input":
                js_script = """
                (() => {
                    // 1. Check contenteditable elements
                    let editables = Array.from(document.querySelectorAll('[contenteditable="true"]'));
                    for (let el of editables) {
                        let rect = el.getBoundingClientRect();
                        if (rect.width > 50 && rect.height > 20) {
                            return { tag: el.tagName.toLowerCase(), id: el.id, className: el.className, isEditable: true };
                        }
                    }
                    // 2. Check textareas or inputs
                    let textareas = Array.from(document.querySelectorAll('textarea, input[type="text"]'));
                    for (let el of textareas) {
                        let rect = el.getBoundingClientRect();
                        if (rect.width > 50 && rect.height > 20) {
                            return { tag: el.tagName.toLowerCase(), id: el.id, className: el.className };
                        }
                    }
                    return null;
                })()
                """
                info = page.evaluate(js_script)
                if info:
                    if info.get("isEditable"):
                        discovered_sel = f"{info.get('tag', 'div')}[contenteditable='true']"
                    elif info.get("id"):
                        discovered_sel = f"#{info['id']}"
                    else:
                        discovered_sel = info.get("tag", "textarea")

                    loc = page.locator(discovered_sel).first
                    if loc.is_visible(timeout=1000):
                        self._logger.info("Auto-discovered chat input selector: '{}'", discovered_sel)
                        return loc, discovered_sel

            elif target_type == "send_button":
                js_script = """
                (() => {
                    let buttons = Array.from(document.querySelectorAll('button'));
                    for (let b of buttons) {
                        let text = (b.innerText || '').toLowerCase();
                        let aria = (b.getAttribute('aria-label') || '').toLowerCase();
                        let testid = (b.getAttribute('data-testid') || '').toLowerCase();
                        if (aria.includes('send') || aria.includes('gửi') || aria.includes('submit') || testid.includes('send') || text.includes('send') || text.includes('gửi')) {
                            let rect = b.getBoundingClientRect();
                            if (rect.width > 10 && rect.height > 10) {
                                return { id: b.id, aria: b.getAttribute('aria-label'), testid: b.getAttribute('data-testid') };
                            }
                        }
                    }
                    return null;
                })()
                """
                info = page.evaluate(js_script)
                if info:
                    if info.get("testid"):
                        discovered_sel = f"button[data-testid='{info['testid']}']"
                    elif info.get("aria"):
                        discovered_sel = f"button[aria-label='{info['aria']}']"
                    elif info.get("id"):
                        discovered_sel = f"#{info['id']}"
                    else:
                        discovered_sel = "button[data-testid='send-button']"
                    loc = page.locator(discovered_sel).first
                    if loc.is_visible(timeout=1000):
                        self._logger.info("Auto-discovered send button selector: '{}'", discovered_sel)
                        return loc, discovered_sel

            elif target_type == "response":
                js_script = """
                (() => {
                    let candidates = Array.from(document.querySelectorAll('model-response, message-content, [data-test-id*="response"], .markdown'));
                    if (candidates.length > 0) {
                        let el = candidates[candidates.length - 1];
                        return el.tagName.toLowerCase();
                    }
                    return null;
                })()
                """
                tag = page.evaluate(js_script)
                if tag:
                    discovered_sel = str(tag)
                    loc = page.locator(discovered_sel)
                    if loc.count() > 0:
                        self._logger.info("Auto-discovered response container selector: '{}'", discovered_sel)
                        return loc, discovered_sel
        except Exception as exc:
            self._logger.warning("DOM Auto-Discovery failed: {}", exc)

        return None, None

    # -------------------------------------------------------------
    # Selector Search with Fallbacks & Auto-Discovery
    # -------------------------------------------------------------
    def _find_element_with_fallback(
        self,
        page: Page,
        selectors: List[str],
        description: str,
        target_type: Optional[str] = None,
        timeout_per_selector_ms: int = 1500,
        max_retries: int = 3,
        check_enabled: bool = False,
    ) -> Tuple[Optional[Locator], Optional[str]]:
        """Try multiple fallback selectors with retry. If all fail, invokes auto-discovery."""
        for attempt in range(1, max_retries + 1):
            for sel in selectors:
                try:
                    loc = page.locator(sel).first
                    if loc.is_visible(timeout=timeout_per_selector_ms):
                        if check_enabled and not loc.is_enabled():
                            continue
                        self._logger.info(
                            "Found {} using selector: '{}' (attempt {}/{})",
                            description,
                            sel,
                            attempt,
                            max_retries,
                        )
                        return loc, sel
                except Exception:
                    continue

            if attempt < max_retries:
                time.sleep(1.0)

        # Fallback to Semantic Auto-Discovery
        if target_type:
            loc, sel = self._auto_discover_element(page, target_type)
            if loc is not None:
                return loc, sel

        self._logger.warning(
            "Failed to find {} after testing all {} fallback selectors across {} attempts",
            description,
            len(selectors),
            max_retries,
        )
        return None, None

    def _get_response_containers(
        self, page: Page
    ) -> Tuple[Optional[Locator], Optional[str], int]:
        """Find response container locator with fallback selectors & auto-discovery."""
        for sel in RESPONSE_CONTAINER_SELECTORS:
            try:
                loc = page.locator(sel)
                cnt = loc.count()
                if cnt > 0:
                    self._logger.info("Matched response containers using selector: '{}' (count: {})", sel, cnt)
                    return loc, sel, cnt
            except Exception:
                continue

        # Auto-discovery fallback
        loc, sel = self._auto_discover_element(page, "response")
        if loc is not None:
            return loc, sel, loc.count()

        return None, None, 0

    # -------------------------------------------------------------
    # Session Authentication & Watchdog
    # -------------------------------------------------------------
    def is_page_authenticated(self, page: Page) -> bool:
        """Check if Gemini page is genuinely logged into Google account (not anonymous guest mode)."""
        return self._on_browser_thread(self._is_page_authenticated_impl, page)

    def _is_page_authenticated_impl(self, page: Page) -> bool:
        try:
            curr_url = page.url.lower()
            # 1. If on explicit sign-in / service login URLs -> definitely not logged in
            if any(p in curr_url for p in [
                "accounts.google.com/servicelogin",
                "accounts.google.com/signin",
                "accounts.google.com/v3/signin",
                "accounts.google.com/interactivelogin",
            ]):
                return False

            # 2. Check if Sign in / Đăng nhập button is visible in header or top navigation (Unauthenticated Guest Mode)
            header_sign_in_sels = [
                "header button:has-text('Sign in')",
                "header button:has-text('Đăng nhập')",
                "header a:has-text('Sign in')",
                "header a:has-text('Đăng nhập')",
                "a:has-text('Đăng nhập')",
                "button:has-text('Đăng nhập')",
                ".header-sign-in-button",
            ]
            for sel in header_sign_in_sels:
                try:
                    loc = page.locator(sel).first
                    if loc.is_visible(timeout=500):
                        self._logger.info("Detected unauthenticated Sign In button: '{}'", sel)
                        return False
                except Exception:
                    pass

            # 3. Check for logged-in UI components (chat box, side nav, profile avatar)
            profile_sels = [
                "rich-textarea",
                "div[contenteditable='true']",
                "chat-app",
                "bard-sidenav",
                "side-navigation",
                "a[href*='SignOutOptions']",
                "a[href*='myaccount.google.com']",
                "button[aria-label*='Google Account']",
                "button[aria-label*='Tài khoản Google']",
                "img[alt*='Google Account']",
                "img[alt*='Tài khoản Google']",
                "img[src*='googleusercontent.com']",
            ]
            for sel in profile_sels:
                try:
                    loc = page.locator(sel).first
                    if loc.is_visible(timeout=500):
                        return True
                except Exception:
                    pass

            # 4. Fallback: If on gemini.google.com -> authenticated
            if "gemini.google.com" in curr_url:
                return True

            return False
        except Exception as exc:
            self._logger.debug("is_page_authenticated check error: {}", exc)
            return True

    def _watchdog_check_and_recover(self, page: Page, timeout_sec: int = 15) -> bool:
        """Detect expired/anonymous login and prompt automatic re-login / session recovery."""
        return self._on_browser_thread(self._watchdog_check_and_recover_impl, page, timeout_sec)

    def _watchdog_check_and_recover_impl(self, page: Page, timeout_sec: int = 15) -> bool:
        if self._is_page_authenticated_impl(page):
            return True

        self._logger.warning("🚨 [Session Watchdog] Detected unauthenticated or guest session! Waiting for authentication...")

        start_t = time.time()
        while time.time() - start_t < timeout_sec:
            if self._is_page_authenticated_impl(page):
                self._logger.info("✅ [Session Watchdog] Session authenticated successfully!")
                self._save_session_impl()
                return True
            time.sleep(1.5)

        self._logger.error("❌ [Session Watchdog] Session authentication timed out after {}s", timeout_sec)
        return False

    # -------------------------------------------------------------
    # Conversation Cleanup
    # -------------------------------------------------------------
    def start_new_chat(self, page: Page) -> None:
        """Start a fresh chat to avoid context leakage between different movie jobs."""
        return self._on_browser_thread(self._start_new_chat_impl, page)

    def _start_new_chat_impl(self, page: Page) -> None:
        self._logger.info("Starting fresh Gemini conversation...")
        # 1. Try clicking 'New chat' button
        for sel in NEW_CHAT_SELECTORS:
            try:
                btn = page.locator(sel).first
                if btn.is_visible(timeout=800):
                    btn.click()
                    self._logger.info("Clicked 'New chat' button with selector: '{}'", sel)
                    time.sleep(1.5)
                    return
            except Exception:
                continue

        # 2. Fallback: navigate directly to base URL
        try:
            page.goto(self._config.base_url, wait_until="domcontentloaded")
            time.sleep(1.5)
        except Exception as exc:
            self._logger.warning("Navigation during new chat cleanup: {}", exc)

    def new_chat_stage(self, stage_idx: int = 1) -> None:
        """Start a fresh chat on a specific stage tab."""
        def _impl():
            page = self._get_stage_page_impl(stage_idx)
            self._start_new_chat_impl(page)
        self._on_browser_thread(_impl)

    # -------------------------------------------------------------
    # Core Public Methods
    # -------------------------------------------------------------
    def login_interactive(self, timeout_sec: int = 300) -> bool:
        """Open interactive browser window for user to log into Google / Gemini."""
        return self._on_browser_thread(self._login_interactive_impl, timeout_sec)

    def _login_interactive_impl(self, timeout_sec: int = 300) -> bool:
        self._logger.info("Opening interactive browser for Google/Gemini login...")

        page = self._ensure_browser(headless=False)
        login_url = "https://accounts.google.com/ServiceLogin?continue=https://gemini.google.com/"
        self._logger.info("Navigating to Google Login: {}", login_url)

        try:
            page.goto(login_url, wait_until="commit", timeout=12000)
        except Exception as exc:
            self._logger.warning("Initial navigation: {}", exc)

        self._logger.info("Waiting for user to log into Google and open Gemini chat...")
        start_t = time.time()

        while time.time() - start_t < timeout_sec:
            if self.is_page_authenticated(page):
                self._logger.info("Login verified successfully!")
                self.save_session()
                return True
            time.sleep(2.0)

        self._logger.warning("Interactive login timed out after {}s", timeout_sec)
        return False

    def reload_cookies(self) -> bool:
        """Inject latest cookies from session file into live browser context and reload active pages."""
        return self._on_browser_thread(self._reload_cookies_impl)

    def _reload_cookies_impl(self) -> bool:
        with BrowserManager._singleton_lock:
            if not self._session_mgr.session_path.exists():
                return False
            try:
                with open(self._session_mgr.session_path, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                cookies_list = sdata.get("cookies", []) if isinstance(sdata, dict) else (sdata if isinstance(sdata, list) else [])
                if not cookies_list:
                    return False

                # If browser context is running, add cookies and reload page
                if BrowserManager._singleton_context:
                    valid_keys = {"name", "value", "url", "domain", "path", "expires", "httpOnly", "secure", "sameSite"}
                    clean_cookies = []
                    for c in cookies_list:
                        if isinstance(c, dict) and "name" in c and "value" in c:
                            cleaned = {k: v for k, v in c.items() if k in valid_keys and v is not None}
                            clean_cookies.append(cleaned)
                    
                    success_cnt = 0
                    for ck in clean_cookies:
                        try:
                            BrowserManager._singleton_context.add_cookies([ck])
                            success_cnt += 1
                        except Exception:
                            pass
                    self._logger.info("Hot-reloaded {}/{} cookies into active browser context", success_cnt, len(clean_cookies))

                    if BrowserManager._singleton_page and not BrowserManager._singleton_page.is_closed():
                        try:
                            BrowserManager._singleton_page.goto(self._config.base_url, wait_until="domcontentloaded", timeout=15000)
                        except Exception:
                            pass
                    for s_page in list(BrowserManager._stage_pages.values()):
                        if s_page and not s_page.is_closed():
                            try:
                                s_page.goto(self._config.base_url, wait_until="domcontentloaded", timeout=15000)
                            except Exception:
                                pass
                return True
            except Exception as exc:
                self._logger.error("Failed to hot-reload cookies: {}", exc)
                return False

    def save_session(self) -> None:
        """Export storage state to session file only if authenticated."""
        return self._on_browser_thread(self._save_session_impl)

    def _save_session_impl(self) -> None:
        with BrowserManager._singleton_lock:
            if BrowserManager._singleton_context:
                try:
                    page = BrowserManager._singleton_page
                    if page and self._is_page_authenticated_impl(page):
                        state = BrowserManager._singleton_context.storage_state()
                        with open(self._session_mgr.session_path, "w", encoding="utf-8") as f:
                            json.dump(state, f, indent=2)
                        self._logger.info("Saved authenticated browser session state to: {}", self._session_mgr.session_path)
                    else:
                        self._logger.debug("Skipping session save: page is not authenticated (prevent wiping valid cookies)")
                except Exception as exc:
                    self._logger.warning("Failed to save storage state file: {}", exc)

    def check_is_logged_in(self) -> SessionStatus:
        """Check if current browser session is logged in."""
        return self._on_browser_thread(self._check_is_logged_in_impl)

    def _check_is_logged_in_impl(self) -> SessionStatus:
        try:
            page = self._ensure_browser(headless=False)
            page.goto(self._config.base_url, wait_until="domcontentloaded", timeout=15000)
            time.sleep(2.5)

            if self._is_page_authenticated_impl(page):
                return SessionStatus.LOGGED_IN
            return SessionStatus.NOT_LOGGED_IN
        except Exception as exc:
            self._logger.warning("Error checking login status: {}", exc)
            return SessionStatus.NOT_LOGGED_IN

    def get_stage_page(
        self,
        stage_idx: int,
        write_content_engine: str = "chatgpt_web",
        review_video_engine: str = "gemini_web",
    ) -> Page:
        """Get or create persistent tab for Stage 1 (Gemini / Google AI Studio) or Stage 2 (ChatGPT / Gemini Web)."""
        return self._on_browser_thread(self._get_stage_page_impl, stage_idx, write_content_engine, review_video_engine)

    def _get_stage_page_impl(
        self,
        stage_idx: int,
        write_content_engine: str = "chatgpt_web",
        review_video_engine: str = "gemini_web",
    ) -> Page:
        with BrowserManager._singleton_lock:
            first_page = self._ensure_browser(headless=False)
            ctx = BrowserManager._singleton_context

            if stage_idx not in BrowserManager._stage_pages or BrowserManager._stage_pages[stage_idx].is_closed():
                pages = ctx.pages
                if stage_idx == 1:
                    page = pages[0] if len(pages) > 0 else ctx.new_page()
                    if review_video_engine == "google_ai_studio":
                        target_url = "https://aistudio.google.com/app/prompts/new_chat?model=gemini-3.5-flash"
                        domain_check = "aistudio.google.com"
                    else:
                        target_url = self._config.base_url  # https://gemini.google.com
                        domain_check = "gemini.google.com"
                else:
                    page = pages[1] if len(pages) > 1 else ctx.new_page()
                    if write_content_engine == "gemini_web":
                        target_url = self._config.base_url  # https://gemini.google.com
                        domain_check = "gemini.google.com"
                    elif write_content_engine == "claude_web":
                        target_url = "https://claude.ai/new"
                        domain_check = "claude.ai"
                    else:
                        target_url = "https://chatgpt.com"
                        domain_check = "chatgpt.com"

                try:
                    self._bring_chrome_to_front(page)
                    if domain_check not in page.url.lower():
                        page.goto(target_url, wait_until="commit", timeout=15000)
                        time.sleep(1.5)
                except Exception:
                    pass

                BrowserManager._stage_pages[stage_idx] = page

            page_obj = BrowserManager._stage_pages[stage_idx]
            self._bring_chrome_to_front(page_obj)
            return page_obj

    def send_prompt_to_stage(
        self,
        stage_idx: int,
        prompt: str,
        media_path: Optional[Path] = None,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        write_content_engine: str = "chatgpt_web",
        review_video_engine: str = "gemini_web",
    ) -> GeminiWebResponse:
        """Send prompt to Stage 1 (Gemini Web / AI Studio) or Stage 2 (ChatGPT Web / Gemini Web)."""
        return self._on_browser_thread(
            self._send_prompt_to_stage_impl,
            stage_idx,
            prompt,
            media_path,
            job_id,
            progress_callback,
            write_content_engine,
            review_video_engine,
        )

    def _send_prompt_to_stage_impl(
        self,
        stage_idx: int,
        prompt: str,
        media_path: Optional[Path] = None,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        write_content_engine: str = "chatgpt_web",
        review_video_engine: str = "gemini_web",
    ) -> GeminiWebResponse:
        """Internal implementation executing strictly on dedicated browser thread."""
        start_time = time.time()
        def _notify(stage_name: str, pct: float) -> None:
            if progress_callback:
                try:
                    progress_callback(stage_name, pct)
                except Exception:
                    pass

        with BrowserManager._job_queue_lock:
            _notify(f"Tab {stage_idx} (Đang mở trình duyệt)", 0.05)
            page = self.get_stage_page(stage_idx, write_content_engine=write_content_engine, review_video_engine=review_video_engine)

            self._bring_chrome_to_front(page)

            if stage_idx == 2 and write_content_engine == "chatgpt_web":
                # -------------------------------------------------------------
                # STAGE 2: CHATGPT WEB WORKFLOW
                # -------------------------------------------------------------
                _notify("Tab 2 (ChatGPT Web: Đang mở trang...)", 0.10)
                try:
                    if "chatgpt.com" not in page.url.lower():
                        page.goto("https://chatgpt.com", wait_until="domcontentloaded", timeout=15000)
                        time.sleep(2.0)
                except Exception as exc:
                    self.save_dom_snapshot(page, f"ChatGPT navigation failure: {exc}", job_id)
                    raise GeminiWebNavigationError(f"Failed to load ChatGPT Web: {exc}") from exc

                # Locate ChatGPT Chat Input
                input_box, input_sel = self._find_element_with_fallback(
                    page,
                    CHATGPT_INPUT_SELECTORS,
                    description="ChatGPT Input",
                    target_type="input",
                    timeout_per_selector_ms=1200,
                    max_retries=4,
                )

                # Check if ChatGPT shows Log In / Sign Up buttons indicating session expired
                login_btn = page.locator("button[data-testid='login-button'], a[href*='login'], button:has-text('Log in')")
                if login_btn.count() > 0 and login_btn.first.is_visible():
                    self.save_dom_snapshot(page, "ChatGPT session expired / login required", job_id)
                    raise GeminiWebAuthError("Phiên đăng nhập ChatGPT (Tab 2) đã hết hạn hoặc chưa đăng nhập. Vui lòng dán Cookie ChatGPT mới!")

                if not input_box:
                    snapshot_dir = self.save_dom_snapshot(page, "ChatGPT input not found", job_id)
                    raise GeminiWebDOMError(f"Không tìm thấy ô nhập câu hỏi ChatGPT. Hướng dẫn: Bấm nút 'Cookie ChatGPT' để dán lại cookie mới. (Snapshot: {snapshot_dir})")

                # Baseline count of assistant response elements in ChatGPT
                baseline_count = page.locator("div[data-message-author-role='assistant']").count()

                # Type prompt into ChatGPT
                _notify("Tab 2 (ChatGPT Web: Đang nhập prompt...)", 0.40)
                try:
                    try:
                        input_box.focus()
                    except Exception:
                        pass
                    try:
                        input_box.click(timeout=1000)
                    except Exception:
                        pass
                    page.keyboard.insert_text(prompt)
                    time.sleep(0.3)
                    try:
                        page.evaluate("el => el.dispatchEvent(new Event('input', { bubbles: true }))", input_box.element_handle())
                    except Exception:
                        pass
                except Exception as exc:
                    # Fallback to direct fill if click/keyboard fails
                    try:
                        input_box.fill(prompt)
                        time.sleep(0.3)
                    except Exception as fill_exc:
                        self.save_dom_snapshot(page, f"Failed typing prompt to ChatGPT: {exc}", job_id)
                        raise GeminiWebDOMError(f"Failed entering text into ChatGPT input: {fill_exc}") from fill_exc

                # Fast Send Submission for ChatGPT
                sent = False
                for s_sel in [
                    "button[data-testid='send-button']",
                    "button[aria-label*='Send' i]",
                    "button[aria-label*='Gửi' i]",
                    "button[aria-label*='gửi' i]",
                ]:
                    try:
                        btn = page.locator(s_sel).first
                        if btn.is_visible(timeout=300) and btn.is_enabled():
                            btn.click(timeout=1000)
                            sent = True
                            self._logger.info("[Tab 2] Clicked ChatGPT Send Button via '{}'", s_sel)
                            break
                    except Exception:
                        continue

                if not sent:
                    self._logger.info("[Tab 2] Pressing Enter to send ChatGPT prompt...")
                    page.keyboard.press("Enter")

                # Wait for ChatGPT response
                _notify("Tab 2 (ChatGPT Web: Đang chờ phản hồi...)", 0.65)
                time.sleep(3.0)

                max_wait_start = time.time()
                target_response = None

                while time.time() - max_wait_start < 45.0:
                    loc = page.locator("div[data-message-author-role='assistant']")
                    if loc.count() > baseline_count:
                        target_response = loc.last
                        break
                    time.sleep(1.0)

                if target_response is None:
                    loc = page.locator("div[data-message-author-role='assistant']")
                    if loc.count() > 0:
                        target_response = loc.last
                    else:
                        snapshot_dir = self.save_dom_snapshot(page, "ChatGPT response timeout", job_id)
                        raise GeminiWebTimeoutError(f"Timed out waiting for ChatGPT response. Snapshot saved to: {snapshot_dir}")

                # Stream stabilization for ChatGPT
                _notify("Tab 2 (ChatGPT Web: Đang nhận kết quả...)", 0.85)
                last_text = ""
                stable_count = 0
                gen_start = time.time()

                while time.time() - gen_start < 120.0:
                    try:
                        # Prefer inner .markdown / .prose container to avoid action buttons like 'Sửa', 'Sao chép'
                        prose_el = target_response.locator(".markdown, .prose, div.flex-col").first
                        if prose_el.count() > 0 and prose_el.is_visible():
                            current_text = prose_el.inner_text().strip()
                        else:
                            current_text = target_response.inner_text().strip()
                        
                        # Clean UI button labels if captured
                        lines = [line.strip() for line in current_text.splitlines()]
                        filtered_lines = [l for l in lines if l not in {"Sửa", "Edit", "Copy", "Sao chép", "Chia sẻ", "Share"}]
                        current_text = "\n".join(filtered_lines).strip()
                    except Exception:
                        current_text = ""

                    if current_text and current_text == last_text:
                        stable_count += 1
                        if stable_count >= 2:
                            break
                    else:
                        stable_count = 0
                        last_text = current_text

                    time.sleep(0.8)

                elapsed = time.time() - start_time
                self.save_session()
                return GeminiWebResponse(
                    text=last_text,
                    processing_time=elapsed,
                    model_name="chatgpt-web-playwright",
                )

            if stage_idx == 2 and write_content_engine == "claude_web":
                # -------------------------------------------------------------
                # STAGE 2: CLAUDE WEB WORKFLOW (claude.ai)
                # -------------------------------------------------------------
                _notify("Tab 2 (Claude Web: Đang mở trang...)", 0.10)
                try:
                    if "claude.ai" not in page.url.lower():
                        page.goto("https://claude.ai/new", wait_until="domcontentloaded", timeout=15000)
                        time.sleep(2.0)
                except Exception as exc:
                    self.save_dom_snapshot(page, f"Claude navigation failure: {exc}", job_id)
                    raise GeminiWebNavigationError(f"Failed to load Claude Web: {exc}") from exc

                # Locate Claude Chat Input
                input_box, input_sel = self._find_element_with_fallback(
                    page,
                    CLAUDE_INPUT_SELECTORS,
                    description="Claude Input",
                    target_type="input",
                    timeout_per_selector_ms=1200,
                    max_retries=4,
                )

                # Check if Claude shows Log In / Sign In buttons indicating session expired
                login_btn = page.locator("a[href*='login'], button:has-text('Log in'), button:has-text('Sign in'), button:has-text('Continue with Email')")
                if login_btn.count() > 0 and login_btn.first.is_visible() and not input_box:
                    self.save_dom_snapshot(page, "Claude session expired / login required", job_id)
                    raise GeminiWebAuthError("Phiên đăng nhập Claude (Tab 2) đã hết hạn hoặc chưa đăng nhập. Vui lòng dán Cookie Claude mới!")

                if not input_box:
                    snapshot_dir = self.save_dom_snapshot(page, "Claude input not found", job_id)
                    raise GeminiWebDOMError(f"Không tìm thấy ô nhập câu hỏi Claude. Hướng dẫn: Bấm nút 'Cookie Claude' để dán lại cookie mới. (Snapshot: {snapshot_dir})")

                # Baseline count of response messages in Claude
                baseline_count = page.locator("div.font-claude-message, div.prose, [data-is-streaming]").count()

                # Type prompt into Claude
                _notify("Tab 2 (Claude Web: Đang nhập prompt...)", 0.40)
                try:
                    try:
                        input_box.focus()
                    except Exception:
                        pass
                    try:
                        input_box.click(timeout=1000)
                    except Exception:
                        pass
                    page.keyboard.insert_text(prompt)
                    time.sleep(0.3)
                    try:
                        page.evaluate("el => el.dispatchEvent(new Event('input', { bubbles: true }))", input_box.element_handle())
                    except Exception:
                        pass
                except Exception as exc:
                    try:
                        input_box.fill(prompt)
                        time.sleep(0.3)
                    except Exception as fill_exc:
                        self.save_dom_snapshot(page, f"Failed typing prompt to Claude: {exc}", job_id)
                        raise GeminiWebDOMError(f"Failed entering text into Claude input: {fill_exc}") from fill_exc

                # Click Send Button for Claude
                sent = False
                for s_sel in [
                    "button[aria-label*='Send' i]",
                    "button[aria-label*='Gửi' i]",
                    "button[type='submit']",
                    "button.bg-accent-main-100",
                ]:
                    try:
                        btn = page.locator(s_sel).first
                        if btn.is_visible(timeout=300) and btn.is_enabled():
                            btn.click(timeout=1000)
                            sent = True
                            self._logger.info("[Tab 2] Clicked Claude Send Button via '{}'", s_sel)
                            break
                    except Exception:
                        continue

                if not sent:
                    self._logger.info("[Tab 2] Pressing Enter to send Claude prompt...")
                    page.keyboard.press("Enter")

                # Wait for Claude response
                _notify("Tab 2 (Claude Web: Đang chờ phản hồi...)", 0.65)
                time.sleep(3.0)

                max_wait_start = time.time()
                target_response = None

                while time.time() - max_wait_start < 45.0:
                    loc = page.locator("div.font-claude-message, div.prose, [data-is-streaming]")
                    if loc.count() > baseline_count:
                        target_response = loc.last
                        break
                    time.sleep(1.0)

                if target_response is None:
                    loc = page.locator("div.font-claude-message, div.prose, [data-is-streaming]")
                    if loc.count() > 0:
                        target_response = loc.last
                    else:
                        snapshot_dir = self.save_dom_snapshot(page, "Claude response timeout", job_id)
                        raise GeminiWebTimeoutError(f"Timed out waiting for Claude response. Snapshot saved to: {snapshot_dir}")

                # Stream stabilization for Claude
                _notify("Tab 2 (Claude Web: Đang nhận kết quả...)", 0.85)
                last_text = ""
                stable_count = 0
                gen_start = time.time()

                while time.time() - gen_start < 120.0:
                    try:
                        current_text = target_response.inner_text().strip()
                        lines = [line.strip() for line in current_text.splitlines()]
                        filtered_lines = [l for l in lines if l not in {"Sửa", "Edit", "Copy", "Sao chép", "Retry", "Thử lại"}]
                        current_text = "\n".join(filtered_lines).strip()
                    except Exception:
                        current_text = ""

                    if current_text and current_text == last_text:
                        stable_count += 1
                        if stable_count >= 2:
                            break
                    else:
                        stable_count = 0
                        last_text = current_text

                    time.sleep(0.8)

                elapsed = time.time() - start_time
                self.save_session()
                return GeminiWebResponse(
                    text=last_text,
                    processing_time=elapsed,
                    model_name="claude-web-playwright",
                )

            # -------------------------------------------------------------
            # STAGE 1: GOOGLE AI STUDIO WEB WORKFLOW
            # -------------------------------------------------------------
            if stage_idx == 1 and review_video_engine == "google_ai_studio":
                _notify("Tab 1 (Google AI Studio: Đang mở trang...)", 0.10)
                try:
                    if "aistudio.google.com" not in page.url.lower():
                        page.goto("https://aistudio.google.com/app/prompts/new_chat?model=gemini-3.5-flash", wait_until="domcontentloaded", timeout=15000)
                        time.sleep(2.5)
                except Exception as exc:
                    self.save_dom_snapshot(page, f"Google AI Studio navigation failure: {exc}", job_id)
                    raise GeminiWebNavigationError(f"Failed to load Google AI Studio Web: {exc}") from exc

                # Auto select working model if preview/unsupported model is selected
                try:
                    for model_picker in [
                        "button:has-text('Gemini 3')",
                        "button:has-text('Preview')",
                        "button:has-text('gemini-3')",
                        ".model-selector-button",
                        "mat-select",
                    ]:
                        m_btn = page.locator(model_picker).first
                        if m_btn.is_visible(timeout=1000):
                            m_btn.click(timeout=1000)
                            time.sleep(0.8)
                            for target_option in [
                                "mat-option:has-text('3.5 Flash')",
                                "mat-option:has-text('2.0 Flash')",
                                "mat-option:has-text('1.5 Flash')",
                                "div:has-text('Gemini 3.5 Flash')",
                                "div:has-text('Gemini 2.0 Flash')",
                            ]:
                                opt = page.locator(target_option).first
                                if opt.is_visible(timeout=1000):
                                    opt.click(timeout=1000)
                                    self._logger.info("[Tab 1] Switched AI Studio model via '{}'", target_option)
                                    time.sleep(1.0)
                                    break
                            break
                except Exception as exc_m:
                    self._logger.debug("[Tab 1] Auto model selection attempt: {}", exc_m)

                # Upload Video File if provided
                if media_path and media_path.exists():
                    _notify("Tab 1 (Google AI Studio: Đang đính kèm video...)", 0.20)
                    self._logger.info("[Tab 1] Uploading video to AI Studio: {} ({:.1f} MB)", media_path.name, media_path.stat().st_size / (1024 * 1024))
                    upload_ok = False
                    media_path_str = str(media_path.resolve())

                    try:
                        file_inputs = page.locator("input[type='file']")
                        if file_inputs.count() > 0:
                            for f_idx in range(file_inputs.count()):
                                try:
                                    file_inputs.nth(f_idx).set_input_files(media_path_str, timeout=4000)
                                    upload_ok = True
                                    self._logger.info("[Tab 1] AI Studio video file input upload succeeded")
                                    break
                                except Exception:
                                    continue
                    except Exception:
                        pass

                    if not upload_ok:
                        # Try Insert file button in Google AI Studio
                        for insert_sel in [
                            "button:has-text('Insert')",
                            "button[aria-label*='Insert' i]",
                            "button:has(mat-icon[fonticon='add'])",
                            "button:has-text('+')",
                        ]:
                            try:
                                btn = page.locator(insert_sel).first
                                if btn.is_visible(timeout=1000):
                                    btn.click()
                                    time.sleep(0.5)
                                    file_inputs = page.locator("input[type='file']")
                                    if file_inputs.count() > 0:
                                        file_inputs.first.set_input_files(media_path_str, timeout=4000)
                                        upload_ok = True
                                        self._logger.info("[Tab 1] AI Studio Insert button upload succeeded")
                                        break
                            except Exception:
                                continue

                    if upload_ok:
                        time.sleep(5.0)  # Wait for AI Studio video processing

                # Find AI Studio Input Box
                input_box, input_sel = self._find_element_with_fallback(
                    page,
                    AISTUDIO_INPUT_SELECTORS,
                    description="Google AI Studio Input",
                    target_type="input",
                    timeout_per_selector_ms=1200,
                    max_retries=4,
                )
                if not input_box:
                    snapshot_dir = self.save_dom_snapshot(page, "AI Studio input not found", job_id)
                    raise GeminiWebDOMError(f"Không tìm thấy ô nhập câu hỏi trên Google AI Studio. (Snapshot: {snapshot_dir})")

                _notify("Tab 1 (Google AI Studio: Đang nhập prompt...)", 0.40)
                try:
                    input_box.focus()
                    input_box.click(timeout=1000)
                except Exception:
                    pass
                try:
                    page.keyboard.insert_text(prompt)
                    time.sleep(0.3)
                except Exception:
                    input_box.fill(prompt)

                # Click Run Button in AI Studio
                run_sent = False
                for r_sel in AISTUDIO_RUN_SELECTORS:
                    try:
                        r_btn = page.locator(r_sel).first
                        if r_btn.is_visible(timeout=500) and r_btn.is_enabled():
                            r_btn.click(timeout=1000)
                            run_sent = True
                            self._logger.info("[Tab 1] Clicked AI Studio Run button via '{}'", r_sel)
                            break
                    except Exception:
                        continue

                if not run_sent:
                    page.keyboard.press("Control+Enter")
                    self._logger.info("[Tab 1] Pressed Ctrl+Enter to run AI Studio prompt")

                # Wait for response in AI Studio
                _notify("Tab 1 (Google AI Studio: Đang chờ AI phân tích video...)", 0.65)
                time.sleep(3.0)

                gen_start = time.time()
                last_text = ""
                stable_count = 0

                while time.time() - gen_start < 120.0:
                    try:
                        turns = page.locator("ms-chat-turn, div.turn-content, div.markdown")
                        if turns.count() > 0:
                            current_text = turns.last.inner_text().strip()
                            if current_text and current_text == last_text:
                                stable_count += 1
                                if stable_count >= 2:
                                    break
                            else:
                                stable_count = 0
                                last_text = current_text
                    except Exception:
                        pass
                    time.sleep(1.0)

                elapsed = time.time() - start_time
                self.save_session()
                return GeminiWebResponse(
                    text=last_text,
                    processing_time=elapsed,
                    model_name="google-ai-studio-web-playwright",
                )

            # -------------------------------------------------------------
            # STAGE 1: GEMINI WEB WORKFLOW
            # -------------------------------------------------------------
            if "gemini.google.com" not in page.url.lower():
                try:
                    _notify(f"Tab {stage_idx} (Đang truy cập Gemini Web)", 0.10)
                    page.goto(self._config.base_url, wait_until="commit", timeout=10000)
                    time.sleep(1.5)
                except Exception as exc:
                    self.save_dom_snapshot(page, f"Tab {stage_idx} navigation failure: {exc}", job_id)

            if not self._watchdog_check_and_recover(page):
                self.save_dom_snapshot(page, f"Tab {stage_idx} session watchdog recovery failed", job_id)
                raise GeminiWebAuthError(f"Phiên đăng nhập Gemini Web (Tab {stage_idx}) đã hết hạn hoặc chưa đăng nhập. Vui lòng bấm nút '🍪 Cookie Gemini' trên phần mềm để dán Cookie mới!")

            # --- UPLOAD ORIGINAL VIDEO FILE IF PROVIDED ---
            if media_path and media_path.exists():
                _notify(f"Tab {stage_idx} (Đang đính kèm video clip...)", 0.20)
                upload_ok = False
                media_path_str = str(media_path.resolve())

                # Clean up any open popups or dropdown menus
                try:
                    page.keyboard.press("Escape")
                    time.sleep(0.3)
                except Exception:
                    pass

                # ── STRATEGY 1: Direct set_input_files on hidden input[type='file'] ───────────────
                try:
                    file_inputs = page.locator("input[type='file']")
                    if file_inputs.count() > 0:
                        for f_idx in range(file_inputs.count()):
                            try:
                                inp = file_inputs.nth(f_idx)
                                inp.set_input_files(media_path_str, timeout=4000)
                                try:
                                    page.evaluate("""(el) => {
                                        if (el) {
                                            el.dispatchEvent(new Event('change', { bubbles: true }));
                                            el.dispatchEvent(new Event('input', { bubbles: true }));
                                        }
                                    }""", inp.element_handle())
                                except Exception:
                                    pass
                                upload_ok = True
                                self._logger.info(
                                    "[Tab {}] Uploading video via direct file input injection: {} ({:.1f} MB)",
                                    stage_idx, media_path.name, media_path.stat().st_size / (1024 * 1024)
                                )
                                break
                            except Exception:
                                continue
                except Exception as exc_dir:
                    self._logger.debug("[Tab {}] Direct input upload attempt: {}", stage_idx, exc_dir)

                # ── STRATEGY 2: 2-step Gemini Web menu upload with File Chooser ─────────────────────────────
                if not upload_ok:
                    UPLOAD_MENU_BUTTON_SELECTORS = [
                        "button[aria-label*='t\u1ea3i l\u00ean']",   # "Nội dung tải lên và công cụ"
                        "button[aria-label*='T\u1ea3i l\u00ean']",
                        "button[aria-label*='upload']",
                        "button[aria-label*='Upload']",
                        "button[aria-label*='Attach']",
                        "button[aria-label*='Th\u00eam']",
                    ]
                    MENU_ITEM_SELECTORS = [
                        "button:has-text('T\u1ec7p')",
                        "button:has-text('t\u1ec7p')",
                        "button:has-text('Video')",
                        "button:has-text('Upload')",
                        "button:has-text('File')",
                        "[role='menuitem']:has-text('T\u1ec7p')",
                        "[role='menuitem']:has-text('Upload')",
                        "mat-option",
                        "[role='menuitem']",
                        "[role='option']",
                        ".mat-mdc-menu-item",
                    ]

                    for menu_btn_sel in UPLOAD_MENU_BUTTON_SELECTORS:
                        try:
                            bottom_area = page.locator("rich-textarea ~ *, form, [class*='input-container'], [class*='bottom-container'], [class*='chat-bar']").last
                            if bottom_area.count() > 0 and bottom_area.locator(menu_btn_sel).count() > 0:
                                menu_btn = bottom_area.locator(menu_btn_sel).last
                            else:
                                menu_btn = page.locator(menu_btn_sel).last

                            if not menu_btn.is_visible(timeout=1000):
                                continue

                            self._logger.info("[Tab {}] Opening upload menu button: {}", stage_idx, menu_btn_sel)
                            menu_btn.click()
                            time.sleep(0.8)

                            # Re-check direct set_input_files after opening upload menu
                            file_inputs = page.locator("input[type='file']")
                            if file_inputs.count() > 0:
                                for f_idx in range(file_inputs.count()):
                                    try:
                                        inp = file_inputs.nth(f_idx)
                                        inp.set_input_files(media_path_str, timeout=4000)
                                        try:
                                            page.evaluate("""(el) => {
                                                if (el) {
                                                    el.dispatchEvent(new Event('change', { bubbles: true }));
                                                    el.dispatchEvent(new Event('input', { bubbles: true }));
                                                }
                                            }""", inp.element_handle())
                                        except Exception:
                                            pass
                                        upload_ok = True
                                        self._logger.info("[Tab {}] Uploaded video via direct file input after menu open", stage_idx)
                                        break
                                    except Exception:
                                        continue
                            if upload_ok:
                                break

                            # Search for menu items that trigger file chooser
                            for item_sel in MENU_ITEM_SELECTORS:
                                try:
                                    items = page.locator(item_sel).all()
                                    for item in items:
                                        if item.is_visible(timeout=400):
                                            try:
                                                with page.expect_file_chooser(timeout=4000) as fc_info:
                                                    item.click()
                                                fc_info.value.set_files(media_path_str)
                                                self._logger.info("[Tab {}] Uploaded via menu item chooser '{}'", stage_idx, item_sel)
                                                upload_ok = True
                                                break
                                            except Exception:
                                                pass
                                    if upload_ok:
                                        break
                                except Exception:
                                    continue

                            if upload_ok:
                                break

                            # Try expect_file_chooser directly on menu button
                            try:
                                with page.expect_file_chooser(timeout=3000) as fc_info2:
                                    menu_btn.click()
                                fc_info2.value.set_files(media_path_str)
                                self._logger.info("[Tab {}] Uploaded via file chooser from upload button", stage_idx)
                                upload_ok = True
                                break
                            except Exception:
                                pass

                            # Press Escape if menu stayed open
                            page.keyboard.press("Escape")
                            time.sleep(0.3)

                        except Exception as exc_menu:
                            self._logger.debug("[Tab {}] Menu strategy failed for '{}': {}", stage_idx, menu_btn_sel, exc_menu)
                            try:
                                page.keyboard.press("Escape")
                            except Exception:
                                pass
                            continue

                if upload_ok:
                    self._logger.info("[Tab {}] File video đã được gắn chặt cố định vào phiên làm việc! Đang chờ hoàn tất nạp...", stage_idx)
                    wait_start = time.time()
                    while time.time() - wait_start < 15.0:
                        spinners = page.locator("mat-progress-spinner, mat-spinner, [role='progressbar'], [class*='spinner'], [class*='uploading']")
                        if spinners.count() > 0 and any(s.is_visible() for s in spinners.all()):
                            time.sleep(1.0)
                        else:
                            break
                    time.sleep(2.0)
                else:
                    self._logger.error("[Tab {}] All upload strategies failed — video not attached", stage_idx)

            input_box, input_sel = self._find_element_with_fallback(
                page,
                CHAT_INPUT_SELECTORS,
                description=f"Tab {stage_idx} Chat Input",
                target_type="input",
                timeout_per_selector_ms=1200,
                max_retries=4,
            )

            if not input_box:
                snapshot_dir = self.save_dom_snapshot(page, f"Tab {stage_idx} input box not found", job_id)
                raise GeminiWebDOMError(f"Could not locate chat input box on Tab {stage_idx}. Snapshot: {snapshot_dir}")

            _, resp_sel_initial, baseline_count = self._get_response_containers(page)

            self._logger.info(
                "[Tab {}] Sending prompt ({} chars, {} words) via input selector '{}'...",
                stage_idx,
                len(prompt),
                len(prompt.split()),
                input_sel,
            )
            try:
                input_box.click()
                time.sleep(0.3)
                page.keyboard.insert_text(prompt)
                time.sleep(0.8)
            except Exception as exc:
                self.save_dom_snapshot(page, f"Tab {stage_idx} typing error: {exc}", job_id)
                raise GeminiWebDOMError(f"Failed to enter text into chat input on Tab {stage_idx}: {exc}") from exc

            # Submit prompt
            sent = False
            send_btn, send_sel = self._find_element_with_fallback(
                page,
                SEND_BUTTON_SELECTORS,
                description=f"Tab {stage_idx} Send Button",
                target_type="send_button",
                timeout_per_selector_ms=800,
                max_retries=2,
                check_enabled=False,
            )

            if send_btn is not None:
                try:
                    if send_btn.is_visible(timeout=500):
                        send_btn.click()
                        sent = True
                        self._logger.info("[Tab {}] Clicked send button: '{}'", stage_idx, send_sel)
                except Exception as exc_send:
                    self._logger.debug("[Tab {}] Click send button failed: {}", stage_idx, exc_send)

            if not sent:
                input_box.focus()
                page.keyboard.press("Enter")
                self._logger.info("[Tab {}] Sent prompt via Enter key press", stage_idx)

            time.sleep(1.5)

            # Verification: if text is still remaining in chat input, force Enter key submission!
            try:
                remaining_text = input_box.inner_text().strip()
                if len(remaining_text) > 5:
                    self._logger.warning("[Tab {}] Text still in input box after send! Retrying force Enter key...", stage_idx)
                    input_box.focus()
                    page.keyboard.press("Enter")
                    time.sleep(1.5)
            except Exception:
                pass
            max_wait_start = time.time()
            active_resp_locator = None
            active_resp_sel = None

            while time.time() - max_wait_start < 45.0:
                loc, sel, cnt = self._get_response_containers(page)
                if loc is not None and cnt > baseline_count:
                    active_resp_locator = loc
                    active_resp_sel = sel
                    break
                time.sleep(1.0)

            if active_resp_locator is None:
                loc, sel, cnt = self._get_response_containers(page)
                if loc is not None and cnt > 0:
                    active_resp_locator = loc
                    active_resp_sel = sel
                else:
                    snapshot_dir = self.save_dom_snapshot(page, f"Tab {stage_idx} response timeout", job_id)
                    raise GeminiWebTimeoutError(f"Timed out waiting for response on Tab {stage_idx}. Snapshot: {snapshot_dir}")

            target_response = active_resp_locator.last
            last_text = ""
            stable_count = 0
            gen_start = time.time()

            while time.time() - gen_start < 120.0:
                try:
                    current_text = target_response.inner_text().strip()
                except Exception:
                    current_text = ""

                # Fast check if stop button is present (generation in progress)
                stop_active = False
                try:
                    stop_loc = page.locator("button[aria-label*='Stop' i], button[aria-label*='Dừng' i], button[aria-label*='stop' i]")
                    stop_active = stop_loc.count() > 0 and stop_loc.first.is_visible()
                except Exception:
                    pass

                if current_text and current_text == last_text and not stop_active:
                    stable_count += 1
                    if stable_count >= 2:
                        self._logger.info("[Tab {}] Response generation finished & stabilized ({:.1f}s)", stage_idx, time.time() - gen_start)
                        break
                else:
                    stable_count = 0
                    last_text = current_text

                time.sleep(0.4)

            if not last_text:
                snapshot_dir = self.save_dom_snapshot(page, f"Tab {stage_idx} empty response text", job_id)
                raise GeminiWebTimeoutError(f"Timed out on Tab {stage_idx}. Snapshot: {snapshot_dir}")

            elapsed = time.time() - start_time
            self.save_session()

            return GeminiWebResponse(
                text=last_text,
                processing_time=elapsed,
                model_name=f"gemini-web-tab{stage_idx}",
            )

    def send_prompt(
        self,
        prompt: str,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> GeminiWebResponse:
        return self._on_browser_thread(self._send_prompt_impl, prompt, job_id, progress_callback)

    def _send_prompt_impl(
        self,
        prompt: str,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> GeminiWebResponse:
        """Send prompt to Gemini Web with FIFO serialization, session watchdog, fallback selectors, and DOM snapshots."""
        start_time = time.time()

        def _notify(substage: str, pct: float) -> None:
            self._logger.info("[Progress] {} ({:.0%})", substage, pct)
            if progress_callback:
                try:
                    progress_callback(substage, pct)
                except Exception:
                    pass

        # -------------------------------------------------------------
        # FIFO Concurrency Queue: Only 1 Gemini prompt executes at a time
        # -------------------------------------------------------------
        with BrowserManager._job_queue_lock:
            _notify("Opening browser", 0.05)
            page = self._ensure_browser(headless=self._config.headless)

            _notify("Loading session", 0.15)
            try:
                _notify("Opening Gemini", 0.25)
                self._logger.info("Navigating to {}", self._config.base_url)
                page.goto(self._config.base_url, wait_until="domcontentloaded")
                time.sleep(2.0)
            except Exception as exc:
                self.save_dom_snapshot(page, f"Navigation failure: {exc}", job_id)
                raise GeminiWebNavigationError(f"Failed to load Gemini Web: {exc}") from exc

            # Session Watchdog check
            if not self._watchdog_check_and_recover(page):
                self.save_dom_snapshot(page, "Session expired / Watchdog recovery failed", job_id)
                raise GeminiWebAuthError("Phiên đăng nhập Gemini Web đã hết hạn hoặc chưa đăng nhập. Vui lòng bấm nút '🍪 Cookie Gemini' trên phần mềm để dán Cookie mới!")

            # Conversation Cleanup: Start fresh chat to avoid context leakage
            _notify("Starting fresh conversation", 0.35)
            self.start_new_chat(page)

            # -------------------------------------------------------------
            # 1. Locate Chat Input
            # -------------------------------------------------------------
            input_box, input_sel = self._find_element_with_fallback(
                page,
                CHAT_INPUT_SELECTORS,
                description="Chat Input",
                target_type="input",
                timeout_per_selector_ms=1200,
                max_retries=4,
            )

            if not input_box:
                snapshot_dir = self.save_dom_snapshot(page, "Chat input not found", job_id)
                raise GeminiWebDOMError(f"Could not locate Gemini chat input box. DOM snapshot saved to: {snapshot_dir}")

            # Baseline count of existing response elements
            _, resp_sel_initial, baseline_count = self._get_response_containers(page)
            self._logger.info(
                "Baseline response count: {} (selector: '{}')",
                baseline_count,
                resp_sel_initial or "none",
            )

            # -------------------------------------------------------------
            # 2. Focus and Type / Insert Prompt
            # -------------------------------------------------------------
            _notify("Typing prompt", 0.50)
            self._logger.info(
                "Sending prompt ({} chars, {} words) via input selector '{}'...",
                len(prompt),
                len(prompt.split()),
                input_sel,
            )
            try:
                input_box.click()
                time.sleep(0.3)
                page.keyboard.insert_text(prompt)
                time.sleep(0.8)
            except Exception as exc:
                self.save_dom_snapshot(page, f"Failed typing prompt: {exc}", job_id)
                raise GeminiWebDOMError(f"Failed to enter text into chat input: {exc}") from exc

            # -------------------------------------------------------------
            # 3. Locate & Click Send Button
            # -------------------------------------------------------------
            send_btn, send_sel = self._find_element_with_fallback(
                page,
                SEND_BUTTON_SELECTORS,
                description="Send Button",
                target_type="send_button",
                timeout_per_selector_ms=1000,
                max_retries=2,
                check_enabled=True,
            )

            if send_btn is not None:
                try:
                    send_btn.click()
                    self._logger.info("Successfully clicked Send button using selector: '{}'", send_sel)
                except Exception as e:
                    self._logger.warning("Clicking send button failed ({}). Falling back to Enter key.", e)
                    page.keyboard.press("Enter")
            else:
                self._logger.info("Send button not found or enabled; submitting via Enter key.")
                page.keyboard.press("Enter")

            # -------------------------------------------------------------
            # 4. Wait for Gemini Response Stream & Stabilization
            # -------------------------------------------------------------
            _notify("Waiting response", 0.65)
            self._logger.info("Waiting for Gemini response stream to begin...")
            time.sleep(3.0)

            # Wait for response container to appear
            max_wait_start = time.time()
            active_resp_locator = None
            active_resp_sel = None

            while time.time() - max_wait_start < 45.0:
                loc, sel, cnt = self._get_response_containers(page)
                if loc is not None and cnt > baseline_count:
                    active_resp_locator = loc
                    active_resp_sel = sel
                    self._logger.info(
                        "New response detected! Total responses: {} (using selector: '{}')",
                        cnt,
                        active_resp_sel,
                    )
                    break
                time.sleep(1.0)

            if active_resp_locator is None:
                loc, sel, cnt = self._get_response_containers(page)
                if loc is not None and cnt > 0:
                    active_resp_locator = loc
                    active_resp_sel = sel
                else:
                    snapshot_dir = self.save_dom_snapshot(page, "Response container timeout", job_id)
                    raise GeminiWebTimeoutError(f"Timed out waiting for Gemini response. Snapshot saved to: {snapshot_dir}")

            _notify("Receiving response stream", 0.85)
            target_response = active_resp_locator.last
            last_text = ""
            stable_count = 0
            gen_start = time.time()

            while time.time() - gen_start < 120.0:
                stop_btn, _ = self._find_element_with_fallback(
                    page,
                    STOP_BUTTON_SELECTORS,
                    description="Stop Button",
                    timeout_per_selector_ms=300,
                    max_retries=1,
                )
                stop_active = (stop_btn is not None)

                try:
                    current_text = target_response.inner_text().strip()
                except Exception:
                    current_text = ""

                if current_text and current_text == last_text and not stop_active:
                    stable_count += 1
                    if stable_count >= 3:  # stable for ~3 seconds
                        self._logger.info(
                            "Gemini finished generating response (stable length: {} chars, matched via '{}').",
                            len(current_text),
                            active_resp_sel,
                        )
                        break
                else:
                    stable_count = 0
                    last_text = current_text

                time.sleep(1.0)

            if not last_text:
                snapshot_dir = self.save_dom_snapshot(page, "Empty response received", job_id)
                raise GeminiWebTimeoutError(f"Timed out waiting for complete response text. Snapshot saved to: {snapshot_dir}")

            _notify("Parsing response", 0.95)
            elapsed = time.time() - start_time
            self._logger.info(
                "Response completed in {:.2f}s ({} chars) via selector '{}'",
                elapsed,
                len(last_text),
                active_resp_sel,
            )

            # Persist session state
            self.save_session()

            return GeminiWebResponse(
                text=last_text,
                processing_time=elapsed,
                model_name="gemini-web-playwright",
            )

    def close(self) -> None:
        """Cleanly close persistent browser singleton."""
        return self._on_browser_thread(self._close_impl)

    def _close_impl(self) -> None:
        with BrowserManager._singleton_lock:
            try:
                if BrowserManager._singleton_context:
                    try:
                        page = BrowserManager._singleton_page
                        if page and self.is_page_authenticated(page):
                            state = BrowserManager._singleton_context.storage_state()
                            with open(self._session_mgr.session_path, "w", encoding="utf-8") as f:
                                json.dump(state, f, indent=2)
                    except Exception:
                        pass
                    BrowserManager._singleton_context.close()
                    BrowserManager._singleton_context = None

                if BrowserManager._singleton_playwright:
                    BrowserManager._singleton_playwright.stop()
                    BrowserManager._singleton_playwright = None

                BrowserManager._singleton_page = None
                BrowserManager._stage_pages.clear()
                self._logger.info("Persistent browser singleton closed cleanly.")
            except Exception as exc:
                self._logger.warning("Error during browser close: {}", exc)
