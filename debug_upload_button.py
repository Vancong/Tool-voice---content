"""
FINAL TEST: After logging in via main app, run this to test video upload.
This uses the profile directory (persistent cookies) to stay logged in.
"""
import time
from pathlib import Path
from src.gemini_web.browser_manager import BrowserManager

TEST_VIDEO = r"E:\tool-yt\Tet\Nested Sequence 08.mp4"

bm = BrowserManager()

def safe(s):
    return str(s).encode("ascii", "replace").decode("ascii")

def _task():
    page = bm.get_stage_page(1)
    time.sleep(3)
    url = page.url
    print(f"URL: {url}")

    # If redirected to login, session is expired
    if "accounts.google" in url or "signin" in url:
        print("SESSION EXPIRED - Please login via python main.py → Đăng nhập button")
        return

    # Inspect page for upload button
    all_btns = page.evaluate("""
        () => document.querySelectorAll('button').length
    """)
    print(f"Total buttons on page: {all_btns}")

    # Try to find upload button
    upload_sel = "button[aria-label*='t\u1ea3i l\u00ean']"
    upload_btn = page.locator(upload_sel).first
    try:
        visible = upload_btn.is_visible(timeout=2000)
        aria = upload_btn.get_attribute("aria-label") or ""
        print(f"Upload button visible: {visible}, aria: {safe(aria)}")
    except Exception as e:
        print(f"Upload button check error: {e}")
        visible = False

    if not visible:
        print("\nScanning all visible buttons...")
        result = page.evaluate("""
            () => {
                const out = [];
                document.querySelectorAll('button').forEach(el => {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        out.push({
                            aria: el.getAttribute('aria-label') || '',
                            text: (el.innerText||'').trim().substring(0, 30),
                            cls: el.className.substring(0, 50),
                        });
                    }
                });
                return out;
            }
        """)
        for r in result:
            print(safe(f"  {r}"))
        return

    # Try clicking and inspect menu
    print("\nClicking upload button...")
    upload_btn.click()
    time.sleep(1.5)

    menu_result = page.evaluate("""
        () => {
            const out = [];
            const sels = ['mat-option', '[role=menuitem]', '[role=option]', '.mat-mdc-menu-item', 'mat-menu button', '[class*=menu-item]'];
            sels.forEach(sel => {
                document.querySelectorAll(sel).forEach(el => {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        out.push({sel, text: (el.innerText||'').trim().substring(0,40), aria: el.getAttribute('aria-label')||''});
                    }
                });
            });
            // Also all visible buttons after menu open
            document.querySelectorAll('button').forEach(el => {
                const rect = el.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0) {
                    out.push({sel:'button', text:(el.innerText||'').trim().substring(0,40), aria:el.getAttribute('aria-label')||''});
                }
            });
            return out;
        }
    """)
    print(f"\nAfter menu click ({len(menu_result)} visible elements):")
    for r in menu_result:
        print(safe(f"  {r}"))

bm._on_browser_thread(_task)
